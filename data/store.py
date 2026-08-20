"""
Append-only JSONL record stores.

Why JSONL and not a database: campaign records are written from many
concurrent processes inside one PBS allocation, on a parallel filesystem, and
must survive a job being killed mid-write. An append-only text file with
one self-contained record per line degrades gracefully -- a truncated final
line costs one record, not the store. SQLite over Lustre/DAOS does not
degrade gracefully.

Concurrency: appends use `fcntl.flock` on POSIX. Single `write()` calls of
lines under the pipe-buffer size are effectively atomic anyway, but labels
carry full force arrays and routinely exceed that, so the lock is real and
necessary.

Corrupt lines are skipped on read with a counted warning rather than raising.
A campaign that dies at hour 5 of 6 should still yield its first 5 hours.
"""

import json
import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:                       # non-POSIX
    _HAVE_FCNTL = False

from data.schema import CPDFTLabel, MDResult, StatePoint, state_point_id


class JSONLStore:
    """Append-only newline-delimited JSON."""

    def __init__(self, path, create: bool = True):
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
        self.n_corrupt_lines = 0

    # -- write -------------------------------------------------------------

    def append(self, record: Any) -> None:
        self.append_many([record])

    def append_many(self, records: Iterable[Any]) -> None:
        payload = "".join(json.dumps(self._as_dict(r), sort_keys=True) + "\n"
                          for r in records)
        if not payload:
            return
        with open(self.path, "a") as fh:
            if _HAVE_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _as_dict(record: Any) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            return record.to_dict()
        if is_dataclass(record):
            from dataclasses import asdict
            return asdict(record)
        if isinstance(record, dict):
            return record
        raise TypeError(
            f"Cannot serialise {type(record).__name__}; provide a dict, a "
            "dataclass, or an object with .to_dict()."
        )

    # -- read --------------------------------------------------------------

    def iter_dicts(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        self.n_corrupt_lines = 0
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Almost always the last line of a killed job.
                    self.n_corrupt_lines += 1
                    continue

    def read_dicts(self) -> List[Dict[str, Any]]:
        return list(self.iter_dicts())

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_dicts())

    def compact(self, key: Callable[[Dict[str, Any]], str],
                keep: str = "last") -> int:
        """Rewrite the file keeping one record per key.

        Append-only means re-running a state point leaves both records. This
        collapses them. Writes to a temp file and renames, so an interrupted
        compaction cannot lose the store. Returns the surviving record count.
        """
        if keep not in ("first", "last"):
            raise ValueError("keep must be 'first' or 'last'.")
        selected: Dict[str, Dict[str, Any]] = {}
        for record in self.iter_dicts():
            k = key(record)
            if keep == "last" or k not in selected:
                selected[k] = record
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            for record in selected.values():
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)
        return len(selected)


class LabelStore(JSONLStore):
    """CP-DFT labels, keyed by state-point id (+ frame index)."""

    def append_label(self, label: CPDFTLabel) -> None:
        self.append(label)

    def labels(self) -> List[CPDFTLabel]:
        return [CPDFTLabel.from_dict(d) for d in self.iter_dicts()]

    def usable_labels(self, mu_tol_ev: float = 0.05) -> List[CPDFTLabel]:
        """Only labels fit for training: converged AND on-target in mu."""
        return [l for l in self.labels() if l.is_usable(mu_tol_ev=mu_tol_ev)]

    def labelled_ids(self) -> set:
        """State points already labelled -- the acquisition policy's
        do-not-repeat set."""
        return {l.state_point_id for l in self.labels()}

    def compact_labels(self) -> int:
        return self.compact(
            key=lambda d: f"{d.get('state_point_id')}:{d.get('frame_index')}")

    def summary(self) -> Dict[str, Any]:
        labels = self.labels()
        usable = [l for l in labels if l.is_usable()]
        mu_errors = [abs(e) for e in (l.mu_error_ev for l in labels) if e is not None]
        return {
            "n_labels": len(labels),
            "n_usable": len(usable),
            "n_unconverged": sum(1 for l in labels if not l.converged),
            "n_state_points": len({l.state_point_id for l in labels}),
            "max_abs_mu_error_ev": max(mu_errors) if mu_errors else None,
            "total_atoms": sum(l.n_atoms for l in labels),
            "corrupt_lines": self.n_corrupt_lines,
        }


class RecordStore(JSONLStore):
    """MD results plus the state points that produced them.

    State points are stored alongside results so a campaign can be resumed,
    or analysed, from the store alone -- without re-deriving what was run.
    """

    def __init__(self, path, state_points_path=None, create: bool = True):
        super().__init__(path, create=create)
        sp_path = (Path(state_points_path) if state_points_path
                   else self.path.with_name(self.path.stem + "_state_points.jsonl"))
        self.state_points = JSONLStore(sp_path, create=create)

    def append_result(self, result: MDResult,
                      state_point: Optional[StatePoint] = None) -> None:
        self.append(result)
        if state_point is not None:
            payload = state_point.to_dict()
            payload["state_point_id"] = state_point_id(state_point)
            self.state_points.append(payload)

    def results(self) -> List[MDResult]:
        return [MDResult.from_dict(d) for d in self.iter_dicts()]

    def state_point_map(self) -> Dict[str, StatePoint]:
        out: Dict[str, StatePoint] = {}
        for d in self.state_points.iter_dicts():
            sid = d.get("state_point_id")
            if sid:
                out[sid] = StatePoint.from_dict(d)
        return out

    def completed_ids(self) -> set:
        return {r.state_point_id for r in self.results() if r.converged}

    def attempted_ids(self) -> set:
        return {r.state_point_id for r in self.results()}

    def to_simulation_records(self) -> List:
        """Everything the exploration agents can reason over.

        Results with no matching state point, or with missing observables, are
        skipped rather than defaulted -- a fabricated facet or charge would
        poison `find_local_abnormalities`, whose whole job is comparing a
        record against its neighbours.
        """
        sp_map = self.state_point_map()
        records = []
        for result in self.results():
            sp = sp_map.get(result.state_point_id)
            if sp is None:
                continue
            if result.barrier_ev is None or result.reaction_energy_ev is None:
                continue
            records.append(result.to_simulation_record(sp))
        return records

    def compact_results(self) -> int:
        self.state_points.compact(key=lambda d: str(d.get("state_point_id")))
        return self.compact(key=lambda d: str(d.get("state_point_id")))

    def summary(self) -> Dict[str, Any]:
        results = self.results()
        return {
            "n_results": len(results),
            "n_converged": sum(1 for r in results if r.converged),
            "n_state_points": len(self.state_point_map()),
            "total_sampled_ns": round(sum(r.sampled_ns for r in results), 4),
            "corrupt_lines": self.n_corrupt_lines,
        }
