"""
Read finished run directories and turn them into records.

This module is the ONLY place that decides whether a calculation succeeded.
Nothing upstream may set `converged` -- not the launcher, not the exit
status, not the absence of an error message. A JDFTx run can exit 0 without
converging, and a PBS job can be killed after writing a plausible-looking
partial COLVAR. Success is a property of the output, so it is determined
here by parsing the output.

Harvesting is idempotent and side-effect free: it reads, it does not write,
and running it twice on the same directory yields the same record.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.schema import (
    CPDFTLabel, CommitteeStats, MDResult, RunStatus, StatePoint,
    HARTREE_EV, state_point_id,
)

# PLUMED's native units are kJ/mol and nm. ASE's Plumed calculator negotiates
# unit conversion with the PLUMED kernel at setup, so the numbers appearing in
# COLVAR depend on how that calculator was configured.
#
# >>> VERIFY ON FIRST REAL RUN <<<
# Print the first few COLVAR bias values and check the magnitude against the
# OPES BARRIER you requested. If BARRIER was 5 eV and the bias column tops out
# near 480, COLVAR is in kJ/mol; if it tops out near 5, it is in eV. Do not
# leave this to inference -- a silent 96.485x error in the bias rescales every
# reweighted free energy in the campaign, and the resulting barriers would
# still look superficially plausible.
KJ_PER_MOL_TO_EV = 1.0 / 96.48533212
BIAS_UNIT_FACTORS = {"ev": 1.0, "kj/mol": KJ_PER_MOL_TO_EV}


# ---------------------------------------------------------------------------
# JDFTx -> CPDFTLabel
# ---------------------------------------------------------------------------

def harvest_jdftx_run(run_dir,
                      state_point: Optional[StatePoint] = None,
                      sp_id: Optional[str] = None,
                      output_filename: str = "out",
                      target_mu_ev: Optional[float] = None,
                      frame_index: Optional[int] = None,
                      parent_trajectory: Optional[str] = None,
                      pseudo_set: Optional[str] = None,
                      n_nodes: Optional[int] = None) -> CPDFTLabel:
    """Parse one grand-canonical JDFTx run into a CPDFTLabel.

    Uses pymatgen.io.jdftx.JDFTXOutfile, whose public attributes (`.e`,
    `.forces`, `.mu`, `.converged`, `.structure`, `.is_gc`) were confirmed
    against the pymatgen API docs.

    A missing or unparseable output yields a label with the appropriate
    RunStatus and `converged=False` rather than an exception -- one failed
    single point out of hundreds must not abort a harvest sweep.
    """
    run_dir = Path(run_dir)
    identifier = sp_id or (state_point_id(state_point) if state_point else run_dir.name)
    out_path = run_dir / output_filename

    def _empty(status: RunStatus) -> CPDFTLabel:
        return CPDFTLabel(
            state_point_id=identifier, energy_ev=float("nan"), mu_ev=float("nan"),
            forces_ev_per_angstrom=[], positions_angstrom=[], species=[],
            cell_angstrom=[], converged=False, status=status,
            target_mu_ev=target_mu_ev, run_dir=str(run_dir),
            frame_index=frame_index, parent_trajectory=parent_trajectory,
            pseudo_set=pseudo_set, n_nodes=n_nodes,
        )

    if not out_path.exists():
        return _empty(RunStatus.MISSING)

    try:
        from pymatgen.io.jdftx.outputs import JDFTXOutfile
    except ImportError as exc:
        raise ImportError(
            "pymatgen>=2025.4 is required to harvest JDFTx output "
            "(pymatgen.io.jdftx). Install/upgrade pymatgen."
        ) from exc

    try:
        outfile = JDFTXOutfile.from_file(str(out_path))
    except Exception:
        # Truncated output from a walltime kill is the common case.
        return _empty(RunStatus.FAILED)

    structure = getattr(outfile, "structure", None)
    species = ([str(site.specie) for site in structure] if structure is not None else [])
    positions = ([list(map(float, site.coords)) for site in structure]
                 if structure is not None else [])
    cell = ([list(map(float, v)) for v in structure.lattice.matrix]
            if structure is not None else [])

    forces = getattr(outfile, "forces", None)
    forces_list = ([list(map(float, f)) for f in np.asarray(forces)]
                   if forces is not None else [])

    converged = bool(getattr(outfile, "converged", False))
    # `.etype` -- NOT `.eopt_type`, which records the minimiser ("ElecMinimize").
    # atomate2 itself conflates these; do not copy that.
    etype = getattr(outfile, "etype", None)
    mu_value = float(getattr(outfile, "mu", float("nan")))
    # `is_gc` only reports that the target-mu TAG was present, not that a
    # grand-canonical SCF converged at that mu. Combine all three signals.
    tag_is_gc = bool(getattr(outfile, "is_gc", target_mu_ev is not None))
    on_target = (target_mu_ev is None or
                 (mu_value == mu_value and abs(mu_value - target_mu_ev) <= 0.05))

    label = CPDFTLabel(
        state_point_id=identifier,
        energy_ev=float(getattr(outfile, "e", float("nan"))),
        mu_ev=mu_value,
        etype=(str(etype) if etype is not None else None),
        efermi_ev=_maybe_float(getattr(outfile, "efermi", None)),
        gc_converged=bool(tag_is_gc and converged and on_target),
        forces_ev_per_angstrom=forces_list,
        positions_angstrom=positions,
        species=species,
        cell_angstrom=cell,
        n_electrons=_parse_n_electrons(out_path),
        is_gc=tag_is_gc,
        converged=converged,
        target_mu_ev=target_mu_ev,
        fluid_model=_parse_tag(out_path, "fluid"),
        status=RunStatus.DONE if converged else RunStatus.UNCONVERGED,
        source="jdftx",
        code_version=_parse_jdftx_version(out_path),
        pseudo_set=pseudo_set,
        run_dir=str(run_dir),
        walltime_s=_run_walltime(run_dir),
        n_nodes=n_nodes,
        frame_index=frame_index,
        parent_trajectory=parent_trajectory,
    )
    return label


def _maybe_float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse_n_electrons(out_path: Path) -> Optional[float]:
    """Electron count JDFTx settled on under grand-canonical control.

    JDFTx reports this as `nElectrons` in the energy-component dump; the last
    occurrence is the converged value. Needed for CP-MACE's `electron=` tag,
    and pymatgen does not expose it as a first-class attribute.
    """
    try:
        text = out_path.read_text(errors="ignore")
    except OSError:
        return None
    matches = re.findall(r"nElectrons:\s*([-+0-9.eEdD]+)", text)
    if not matches:
        matches = re.findall(r"nElectrons\s*=\s*([-+0-9.eEdD]+)", text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _parse_jdftx_version(out_path: Path) -> Optional[str]:
    try:
        with open(out_path, errors="ignore") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                m = re.search(r"JDFTx\s+([0-9][^\s*]*)", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def _parse_tag(out_path: Path, tag: str) -> Optional[str]:
    """Echoed input tag value from the JDFTx output header."""
    try:
        text = out_path.read_text(errors="ignore")
    except OSError:
        return None
    m = re.search(rf"^{re.escape(tag)}\s+(\S+)", text, flags=re.MULTILINE)
    return m.group(1) if m else None


def _run_walltime(run_dir: Path) -> Optional[float]:
    """Crude but useful: span between oldest and newest file mtimes."""
    try:
        times = [p.stat().st_mtime for p in run_dir.iterdir() if p.is_file()]
    except OSError:
        return None
    if len(times) < 2:
        return None
    return float(max(times) - min(times))


# ---------------------------------------------------------------------------
# COLVAR / MD -> MDResult
# ---------------------------------------------------------------------------

def read_colvar(path, bias_units: str = "ev") -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Parse a PLUMED COLVAR file.

    Returns (cv, bias_ev, all_columns). Column names come from the
    `#! FIELDS ...` header, so this does not depend on column order.
    `bias_units` must match how PLUMED was configured -- see the note at the
    top of this module. Passing the wrong value silently rescales every free
    energy derived from this trajectory.
    """
    path = Path(path)
    factor = BIAS_UNIT_FACTORS.get(bias_units.lower())
    if factor is None:
        raise ValueError(
            f"bias_units={bias_units!r} not understood; expected one of "
            f"{sorted(BIAS_UNIT_FACTORS)}."
        )

    fields: List[str] = []
    rows: List[List[float]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#! FIELDS"):
                fields = line.split()[2:]
                continue
            if line.startswith("#") or not line.strip():
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue          # partial final line from a killed job

    if not rows:
        raise ValueError(f"No data rows parsed from {path}.")
    width = min(len(r) for r in rows)
    array = np.array([r[:width] for r in rows], dtype=float)
    if not fields:
        fields = [f"col{i}" for i in range(width)]
    fields = fields[:width]
    columns = {name: array[:, i] for i, name in enumerate(fields)}

    cv_name = next((f for f in fields if f not in ("time",) and "bias" not in f), None)
    bias_name = next((f for f in fields if f.endswith(".bias") or f == "bias"), None)
    if cv_name is None or bias_name is None:
        raise ValueError(
            f"Could not identify CV and bias columns in {path}; fields={fields}. "
            "Expected a PLUMED PRINT line of the form "
            "`PRINT ARG=<cv>,opes.bias ... FILE=COLVAR`."
        )
    return columns[cv_name], columns[bias_name] * factor, columns


def harvest_md_run(run_dir,
                   state_point: Optional[StatePoint] = None,
                   sp_id: Optional[str] = None,
                   colvar_name: str = "COLVAR",
                   bias_units: str = "ev",
                   temperature_k: float = 300.0,
                   timestep_fs: float = 0.5,
                   convergence_tol_ev: float = 0.02,
                   n_blocks: int = 5,
                   committee: Optional[CommitteeStats] = None,
                   mlip: str = "esen-oc25",
                   ensemble: str = "constant_charge",
                   opes_barrier_ev: Optional[float] = None,
                   opes_pace: Optional[int] = None,
                   n_nodes: Optional[int] = None) -> MDResult:
    """Parse one biased-MD run directory into an MDResult.

    Free energies come from analysis.free_energy, so the harvest path and the
    interactive-analysis path cannot drift apart.
    """
    run_dir = Path(run_dir)
    identifier = sp_id or (state_point_id(state_point) if state_point else run_dir.name)
    colvar_path = run_dir / colvar_name

    result = MDResult(
        state_point_id=identifier, run_dir=str(run_dir),
        temperature_k=temperature_k, timestep_fs=timestep_fs,
        committee=committee, mlip=mlip, ensemble=ensemble,
        opes_barrier_ev=opes_barrier_ev, opes_pace=opes_pace,
        colvar_path=str(colvar_path), n_nodes=n_nodes,
        walltime_s=_run_walltime(run_dir),
    )

    if not colvar_path.exists():
        result.status = RunStatus.MISSING
        return result

    try:
        cv, bias_ev, _ = read_colvar(colvar_path, bias_units=bias_units)
    except (ValueError, OSError):
        result.status = RunStatus.FAILED
        return result

    result.n_steps = int(len(cv))
    result.sampled_ns = _sampled_ns(run_dir, colvar_path, cv, timestep_fs)

    from analysis.free_energy import (
        block_free_energy_convergence, extract_barrier_and_reaction_energy,
    )

    observables = extract_barrier_and_reaction_energy(
        cv, bias_ev, temperature_k=temperature_k)
    if observables.get("barrier_ev") is None:
        # Too few populated bins: the CV never explored enough of the range.
        result.status = RunStatus.UNCONVERGED
        return result

    stats = block_free_energy_convergence(
        cv, bias_ev, n_blocks=n_blocks, temperature_k=temperature_k)

    result.barrier_ev = observables["barrier_ev"]
    result.reaction_energy_ev = observables["reaction_energy_ev"]
    result.barrier_ev_std = stats["barrier_ev_std"]
    result.reaction_energy_ev_std = stats["reaction_energy_ev_std"]
    result.converged = bool(
        stats["barrier_ev_std"] is not None
        and stats["reaction_energy_ev_std"] is not None
        and stats["barrier_ev_std"] < convergence_tol_ev
        and stats["reaction_energy_ev_std"] < convergence_tol_ev
    )
    result.status = RunStatus.DONE if result.converged else RunStatus.UNCONVERGED

    mu_mean, mu_std = _read_mu_log(run_dir)
    result.achieved_mu_mean_ev, result.achieved_mu_std_ev = mu_mean, mu_std

    traj = next((p for p in (run_dir / "traj.traj", run_dir / "traj.xyz",
                             run_dir / "trajectory.traj") if p.exists()), None)
    if traj is not None:
        result.trajectory_path = str(traj)
    return result


def _sampled_ns(run_dir: Path, colvar_path: Path, cv: np.ndarray,
                timestep_fs: float) -> float:
    """Prefer PLUMED's own time column; fall back to steps x stride x dt.

    The fallback needs the PRINT stride, which is read back out of plumed.dat
    rather than assumed -- assuming stride=1 would understate sampled time by
    the stride factor and make every cost comparison wrong.
    """
    try:
        _, _, columns = read_colvar(colvar_path)
        if "time" in columns and len(columns["time"]) > 1:
            # ASE-driven PLUMED reports time in ASE time units unless
            # reconfigured; treat the span as ps only if it is plausible.
            span = float(columns["time"][-1] - columns["time"][0])
            if span > 0:
                return span / 1000.0
    except (ValueError, OSError):
        pass

    stride = 1
    plumed_dat = run_dir / "plumed.dat"
    if plumed_dat.exists():
        m = re.search(r"STRIDE=(\d+)", plumed_dat.read_text(errors="ignore"))
        if m:
            stride = int(m.group(1))
    return len(cv) * stride * timestep_fs * 1e-6


def _read_mu_log(run_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """Mean and spread of the achieved Fermi level over a constant-potential
    run, from the `mu.log` written by md.cp_md_driver.

    `achieved_mu_std_ev` is the potentiostat tracking error. A large value
    means the run was not actually at constant potential, whatever `targetmu`
    said, and the resulting free energies are not constant-potential results.
    """
    mu_log = run_dir / "mu.log"
    if not mu_log.exists():
        return None, None
    values = []
    for line in mu_log.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            values.append(float(parts[-1]))
        except (ValueError, IndexError):
            continue
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def harvest_all(runs_root, state_points: Optional[Dict[str, StatePoint]] = None,
                kind: str = "auto", **kwargs) -> Tuple[List[CPDFTLabel], List[MDResult]]:
    """Walk every run directory under `runs_root` and harvest what is there.

    kind: 'auto' | 'jdftx' | 'md'. 'auto' decides per directory by looking for
    a COLVAR (MD) or a JDFTx `out` file. Directories containing neither are
    skipped silently -- they are normally job scratch, not failed science.
    """
    runs_root = Path(runs_root)
    state_points = state_points or {}
    labels: List[CPDFTLabel] = []
    results: List[MDResult] = []
    if not runs_root.exists():
        return labels, results

    md_kwargs = {k: v for k, v in kwargs.items()
                 if k in harvest_md_run.__code__.co_varnames}
    dft_kwargs = {k: v for k, v in kwargs.items()
                  if k in harvest_jdftx_run.__code__.co_varnames}

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        sp = state_points.get(run_dir.name)
        has_colvar = (run_dir / "COLVAR").exists()
        has_out = (run_dir / "out").exists()

        do_md = kind == "md" or (kind == "auto" and has_colvar)
        do_dft = kind == "jdftx" or (kind == "auto" and has_out and not has_colvar)

        if do_md:
            results.append(harvest_md_run(run_dir, state_point=sp,
                                          sp_id=run_dir.name, **md_kwargs))
        if do_dft:
            labels.append(harvest_jdftx_run(
                run_dir, state_point=sp, sp_id=run_dir.name,
                target_mu_ev=(sp.target_mu_ev if sp else None), **dft_kwargs))
    return labels, results
