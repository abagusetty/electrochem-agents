"""
Pre-registration and scoring of the acquisition comparison.

A baseline chosen after seeing results is not a baseline. This module writes
the comparison down BEFORE the first adaptive round -- grid resolution,
budget, tolerance, target values -- hashes it, and refuses to score against a
registration that was edited afterwards.

The metric, from the research plan:

    cost to reproduce the anchor paper's reported values within tolerance,
    counted in (i) grand-canonical DFT single points and (ii) ns of biased MD.

Scored for B0 grid, B1 random, B2 force-only, A1 sigma_mu. The publishable
quantity is A1 - B2. Everything else is context.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Anchor-paper reference values (arXiv:2509.17862). The reproduction target.
ANCHOR_TARGETS: Dict[str, Dict[str, float]] = {
    "cu100_neutral": {"barrier_ev": 0.64, "reaction_energy_ev": 0.375},
    "cu310_neutral": {"barrier_ev": 0.57, "reaction_energy_ev": 0.088},
    "cu310_m23": {"barrier_ev": 0.49, "reaction_energy_ev": -0.037},
}


@dataclass
class PreRegistration:
    """Frozen specification of the comparison."""

    campaign: str
    created_utc: str                       # caller supplies; no clock here
    facets: List[str]
    controls: List[float]
    control_kind: str                      # 'target_mu_ev' | 'surface_charge_uc_cm2'
    cations: List[Optional[str]]
    dft_budget_total: int                  # max grand-canonical single points
    md_budget_ns_total: float
    rounds: int
    budget_per_round: int
    tolerance_ev: float = 0.05             # |ours - anchor| to count as reproduced
    targets: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: dict(ANCHOR_TARGETS))
    policies: List[str] = field(default_factory=lambda: [
        "B0_grid", "B1_random", "B2_force_uncertainty", "A1_sigma_mu"])
    primary_comparison: str = "A1_sigma_mu - B2_force_uncertainty"
    notes: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def write(self, path) -> Path:
        """Write once. Refuses to overwrite -- that is the whole point."""
        path = Path(path)
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("fingerprint") != self.fingerprint():
                raise RuntimeError(
                    f"A DIFFERENT pre-registration already exists at {path}\n"
                    f"  on disk:  {existing.get('fingerprint')}\n"
                    f"  proposed: {self.fingerprint()}\n"
                    "Refusing to overwrite. Changing the registration after "
                    "results exist invalidates the comparison. Start a new "
                    "campaign directory instead."
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path) -> "PreRegistration":
        d = json.loads(Path(path).read_text())
        recorded = d.pop("fingerprint", None)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        registration = cls(**known)
        if recorded and recorded != registration.fingerprint():
            raise RuntimeError(
                f"Pre-registration at {path} has been EDITED since it was "
                f"written (recorded {recorded}, recomputed "
                f"{registration.fingerprint()}). Results scored against it are "
                "not pre-registered."
            )
        return registration


@dataclass
class PolicyRun:
    """One policy's realised cost and accuracy."""
    policy_name: str
    n_dft_calls: int = 0
    md_ns: float = 0.0
    reproduced: Dict[str, bool] = field(default_factory=dict)
    best_error_ev: Dict[str, float] = field(default_factory=dict)
    rounds_used: int = 0
    notes: str = ""

    @property
    def n_reproduced(self) -> int:
        return sum(1 for v in self.reproduced.values() if v)

    @property
    def all_reproduced(self) -> bool:
        return bool(self.reproduced) and all(self.reproduced.values())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"n_reproduced": self.n_reproduced,
                  "all_reproduced": self.all_reproduced})
        return d


def score_reproduction(observed: Dict[str, Dict[str, float]],
                       registration: PreRegistration) -> Dict[str, Any]:
    """Compare observed barriers/reaction energies against the frozen targets.

    A target counts as reproduced only when BOTH the barrier and the reaction
    energy are within tolerance. Scoring them separately would let a policy
    claim a hit on the easier of the two.
    """
    reproduced: Dict[str, bool] = {}
    errors: Dict[str, float] = {}
    for key, target in registration.targets.items():
        got = observed.get(key)
        if not got:
            reproduced[key] = False
            errors[key] = float("inf")
            continue
        deltas = [abs(got.get(field_name, float("inf")) - value)
                  for field_name, value in target.items()]
        worst = max(deltas) if deltas else float("inf")
        errors[key] = worst
        reproduced[key] = worst <= registration.tolerance_ev
    return {"reproduced": reproduced, "worst_error_ev": errors,
            "n_reproduced": sum(reproduced.values()),
            "n_targets": len(registration.targets)}


def compare_policies(runs: Sequence[PolicyRun],
                     registration: PreRegistration) -> Dict[str, Any]:
    """Rank policies by cost-to-reproduce and report the A1 - B2 gap.

    A policy that did not reproduce all targets is NOT ranked by cost -- it
    did not finish the task, and comparing its cost to one that did would
    reward stopping early.
    """
    by_name = {r.policy_name: r for r in runs}
    complete = [r for r in runs if r.all_reproduced]
    incomplete = [r for r in runs if not r.all_reproduced]
    ranked = sorted(complete, key=lambda r: (r.n_dft_calls, r.md_ns))

    a1, b2 = by_name.get("A1_sigma_mu"), by_name.get("B2_force_uncertainty")
    verdict, gap = "indeterminate", None
    if a1 is None or b2 is None:
        verdict = ("MISSING ABLATION: both A1_sigma_mu and B2_force_uncertainty "
                   "must be run. Without B2 there is no evidence that sigma_mu "
                   "adds anything over ordinary force uncertainty.")
    elif not (a1.all_reproduced and b2.all_reproduced):
        verdict = ("INCONCLUSIVE: at least one of A1/B2 did not reproduce all "
                   "targets within tolerance, so their costs are not comparable.")
    else:
        gap = b2.n_dft_calls - a1.n_dft_calls
        relative = gap / b2.n_dft_calls if b2.n_dft_calls else 0.0
        if relative > 0.20:
            verdict = (f"A1 BEATS B2: {gap} fewer DFT calls "
                       f"({relative:.0%} cheaper). sigma_mu carries information "
                       "beyond force uncertainty -- the C3 claim stands.")
        elif relative < -0.20:
            verdict = (f"A1 LOSES TO B2 by {-gap} DFT calls. Report it. The "
                       "Fermi-level signal is worse than force uncertainty here.")
        else:
            verdict = (f"NO MEANINGFUL DIFFERENCE ({gap:+d} DFT calls, "
                       f"{relative:+.0%}). Publishable negative result: generic "
                       "MLIP uncertainty suffices; sigma_mu adds nothing. "
                       "Retitle the paper accordingly.")

    return {
        "registration_fingerprint": registration.fingerprint(),
        "tolerance_ev": registration.tolerance_ev,
        "primary_comparison": registration.primary_comparison,
        "a1_minus_b2_dft_calls": (-gap if gap is not None else None),
        "verdict": verdict,
        "ranking": [r.to_dict() for r in ranked],
        "did_not_reproduce": [r.to_dict() for r in incomplete],
    }


def write_comparison(path, comparison: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2, default=str))
    return path
