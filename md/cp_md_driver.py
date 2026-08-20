"""
Constant-potential MD driver.

Replaces the single `dyn.run(steps)` call in mlip.cp_mace_simulation with the
`integrator.run(1)`-inside-a-loop pattern that CP-MACE's own simulate.py
uses. Three things depend on that loop existing:

  1. SAFETY. CP-MACE's simulate.py checks `force_threshold` and
     `fermi_threshold` between steps. A single run(steps) call never checks
     them, so a grand-canonical run whose electron degree of freedom runs
     away produces a full-length trajectory of garbage instead of stopping.
  2. UNCERTAINTY. sigma_mu must be sampled along the trajectory, not at the
     end. The committee calculator records per-step statistics only if
     something reads them per step.
  3. AGENT CONTROL. The sampling-control agent adjusts OPES BARRIER/PACE
     mid-run. The between-steps boundary is the only place it can act.

Architecture, matching CP-MACE exactly:

    NoseHoover integrator   steps positions/velocities + electron DOF
            v calls
    Plumed-wrapped calc     adds CV bias forces        (biased runs only)
            v calls
    CommitteeCalculator     mean forces + sigma_F, sigma_mu, get_mu()

PLUMED wraps the CALCULATOR; it never replaces the integrator.

Nothing here has been executed.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from data.schema import CommitteeStats, MDResult, RunStatus, StatePoint, state_point_id


@dataclass
class CPMDConfig:
    """Constant-potential MD settings.

    Field names follow CP-MACE's inputs.yml so a config can be diffed against
    their published examples. Their slow-growth example uses
    targetmu=-3.36, Mne=660.74, eta_length=2, ttime=40, timestep=1 fs.
    """

    # Integrator (CP-MACE NoseHoover)
    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    ttime: float = 40.0
    mne: float = 660.74               # fictitious electron mass
    eta_length: int = 2               # Nose-Hoover chain length
    targetmu_ev: float = -3.36        # target electrode potential
    constraints: List[List[float]] = field(default_factory=list)
    increm: float = 0.0               # slow-growth CV increment per step

    # Run control
    n_steps: int = 100_000
    check_every: int = 1              # steps per integrator.run(1) batch
    record_every: int = 10            # committee/mu sampling stride
    trajectory_every: int = 100

    # Safety rails (CP-MACE simulate.py defaults)
    force_threshold: float = 0.15     # eV/A; abort above this
    fermi_threshold: float = 0.04     # eV; abort if |mu - targetmu| exceeds
    abort_on_threshold: bool = True

    # Bias
    use_plumed: bool = False
    plumed_input_lines: Optional[List[str]] = None

    # Output
    save_dir: str = "."
    trajectory_name: str = "traj.traj"
    mu_log_name: str = "mu.log"
    sigma_log_name: str = "sigma.jsonl"
    read_velocity: bool = False
    t_init_k: Optional[float] = None  # None -> use temperature_k

    def to_cp_mace_integrator_kwargs(self) -> Dict[str, Any]:
        """Exactly the eight keyword arguments CP-MACE's NoseHoover requires.

        Verified against its constructor:
            NoseHoover(atoms, timestep, constraints, increm, temperature,
                       ttime, Mne, eta_length, targetmu, f0=None, ...)
        Unit conversion (timestep *= units.fs, temperature *= units.kB) is
        applied by the caller, matching their simulate.py.
        """
        return {
            "timestep": self.timestep_fs,
            "temperature": self.temperature_k,
            "ttime": self.ttime,
            "constraints": self.constraints,
            "increm": self.increm,
            "Mne": self.mne,
            "eta_length": self.eta_length,
            "targetmu": self.targetmu_ev,
        }


@dataclass
class StepContext:
    """Passed to hooks each check interval. Hooks may mutate `request`."""
    step: int
    n_steps: int
    mu: Optional[float]
    sigma_mu: float
    sigma_force: float
    max_force: float
    energy: float
    elapsed_s: float
    request: Dict[str, Any] = field(default_factory=dict)


# A hook returns None, or sets keys in ctx.request:
#   'stop': True                     -- terminate early (converged / diverged)
#   'opes_barrier_ev': float         -- retune the bias
#   'opes_pace': int
#   'audit_frame': True              -- flag this frame for CP-DFT audit
Hook = Callable[[StepContext], None]


class ConstantPotentialMD:
    """Run constant-potential MD with per-step inspection and control."""

    def __init__(self, atoms, config: CPMDConfig,
                 cp_mace_repo: Optional[str] = None,
                 integrator_class: Optional[type] = None,
                 hooks: Optional[List[Hook]] = None):
        self.atoms = atoms
        self.config = config
        self.cp_mace_repo = cp_mace_repo
        self._integrator_class = integrator_class
        self.hooks: List[Hook] = list(hooks or [])

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, float]] = []
        self.audit_frames: List[int] = []
        self.stop_reason: Optional[str] = None
        self.steps_completed: int = 0

    # -- setup -------------------------------------------------------------

    def _load_integrator_class(self):
        if self._integrator_class is not None:
            return self._integrator_class
        from mlip.cp_mace_simulation import CPMACERunConfig, load_cp_mace_integrator_class
        if not self.cp_mace_repo:
            raise ValueError(
                "cp_mace_repo must point at a local CP-MACE checkout so the "
                "real NoseHoover integrator can be imported. It is not vendored "
                "here: that repository ships no LICENSE, and its constant-mu "
                "equations of motion must not be reimplemented from guesses."
            )
        mode = "metadynamics" if self.config.use_plumed else "slow_growth"
        return load_cp_mace_integrator_class(
            CPMACERunConfig(cp_mace_repo_path=self.cp_mace_repo, model_path="",
                            init_xyz_path="", mode=mode))

    def _wrap_with_plumed(self):
        config = self.config
        if not config.use_plumed:
            return
        if not config.plumed_input_lines:
            raise ValueError("use_plumed=True requires plumed_input_lines.")
        from ase import units
        from ase.calculators.plumed import Plumed

        base = self.atoms.calc
        if base is None:
            raise ValueError("atoms.calc must be set before wrapping with PLUMED.")
        self.atoms.calc = Plumed(
            calc=base,
            input=list(config.plumed_input_lines),
            timestep=config.timestep_fs * units.fs,
            atoms=self.atoms,
            kT=config.temperature_k * units.kB,
        )

    def _build_integrator(self):
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        config = self.config
        if not config.read_velocity:
            MaxwellBoltzmannDistribution(
                self.atoms, temperature_K=config.t_init_k or config.temperature_k)

        NoseHoover = self._load_integrator_class()
        kwargs = config.to_cp_mace_integrator_kwargs()
        # Same conversions CP-MACE's simulate.py applies.
        kwargs["timestep"] = kwargs["timestep"] * units.fs
        kwargs["temperature"] = kwargs["temperature"] * units.kB
        return NoseHoover(self.atoms, **kwargs)

    # -- inspection --------------------------------------------------------

    def _committee(self):
        """Find the CommitteeCalculator, through a PLUMED wrapper if present."""
        calc = self.atoms.calc
        if hasattr(calc, "sigma_mu"):
            return calc
        inner = getattr(calc, "calc", None)
        return inner if inner is not None and hasattr(inner, "sigma_mu") else None

    def _current_mu(self) -> Optional[float]:
        calc = self.atoms.calc
        for candidate in (calc, getattr(calc, "calc", None)):
            fn = getattr(candidate, "get_mu", None)
            if callable(fn):
                try:
                    return float(fn())
                except Exception:            # noqa: BLE001
                    continue
        return None

    def _apply_hooks(self, ctx: StepContext) -> Dict[str, Any]:
        for hook in self.hooks:
            hook(ctx)
        return ctx.request

    def _retune_plumed(self, barrier_ev: Optional[float], pace: Optional[int]) -> bool:
        """Sampling-control agent's lever.

        PLUMED cannot retune OPES in place through the ASE wrapper, so this
        records the request and returns True to signal that the caller should
        restart with new bias settings. Pretending to retune in place would be
        worse than admitting the restart -- a bias whose parameters silently
        did not change would corrupt the reweighting.
        """
        if barrier_ev is None and pace is None:
            return False
        self.stop_reason = (
            f"retune_requested(barrier_ev={barrier_ev}, pace={pace})")
        return True

    # -- main loop ---------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        from ase.io.trajectory import Trajectory

        config = self.config
        self._wrap_with_plumed()
        integrator = self._build_integrator()
        committee = self._committee()

        traj_path = self.save_dir / config.trajectory_name
        mu_log_path = self.save_dir / config.mu_log_name
        sigma_log_path = self.save_dir / config.sigma_log_name

        trajectory = Trajectory(str(traj_path), "w", self.atoms)
        mu_log = open(mu_log_path, "w")
        mu_log.write("# step  target_mu_ev  mu_ev  sigma_mu_ev\n")
        sigma_log = open(sigma_log_path, "w")

        start = time.time()
        step = 0
        try:
            while step < config.n_steps:
                batch = min(config.check_every, config.n_steps - step)
                integrator.run(batch)          # the run(1) loop, batched
                step += batch
                self.steps_completed = step

                if step % config.record_every != 0 and step < config.n_steps:
                    continue

                forces = self.atoms.get_forces()
                max_force = float(np.abs(forces).max())
                energy = float(self.atoms.get_potential_energy())
                mu = self._current_mu()
                sigma_mu = float(getattr(committee, "sigma_mu", 0.0)) if committee else 0.0
                sigma_force = float(getattr(committee, "sigma_force", 0.0)) if committee else 0.0

                record = {
                    "step": step, "energy_ev": energy, "max_force": max_force,
                    "mu_ev": mu, "sigma_mu_ev": sigma_mu,
                    "sigma_force_ev_per_angstrom": sigma_force,
                    "mu_error_ev": (None if mu is None else mu - config.targetmu_ev),
                }
                self.records.append(record)
                sigma_log.write(json.dumps(record) + "\n")
                mu_log.write(
                    f"{step:10d}  {config.targetmu_ev:+.6f}  "
                    f"{(mu if mu is not None else float('nan')):+.6f}  "
                    f"{sigma_mu:.6f}\n")

                if step % config.trajectory_every == 0:
                    trajectory.write()

                # -- safety rails, CP-MACE simulate.py semantics -------------
                if config.abort_on_threshold:
                    if max_force > config.force_threshold:
                        self.stop_reason = (
                            f"force_threshold exceeded at step {step}: "
                            f"{max_force:.4f} > {config.force_threshold} eV/A")
                        break
                    if mu is not None:
                        mu_error = abs(mu - config.targetmu_ev)
                        if mu_error > config.fermi_threshold:
                            self.stop_reason = (
                                f"fermi_threshold exceeded at step {step}: "
                                f"|mu - targetmu| = {mu_error:.4f} > "
                                f"{config.fermi_threshold} eV. The run was NOT "
                                "at constant potential past this point.")
                            break
                    if not math.isfinite(energy):
                        self.stop_reason = f"non-finite energy at step {step}"
                        break

                # -- agent hooks ---------------------------------------------
                ctx = StepContext(
                    step=step, n_steps=config.n_steps, mu=mu, sigma_mu=sigma_mu,
                    sigma_force=sigma_force, max_force=max_force, energy=energy,
                    elapsed_s=time.time() - start,
                )
                request = self._apply_hooks(ctx)
                if request.get("audit_frame"):
                    self.audit_frames.append(step)
                if request.get("stop"):
                    self.stop_reason = request.get("stop_reason", "hook requested stop")
                    break
                if self._retune_plumed(request.get("opes_barrier_ev"),
                                       request.get("opes_pace")):
                    break
        finally:
            trajectory.write()
            trajectory.close()
            mu_log.close()
            sigma_log.close()

        return self.summary(time.time() - start, traj_path, mu_log_path)

    # -- reporting ---------------------------------------------------------

    def summary(self, elapsed_s: float, traj_path: Path,
                mu_log_path: Path) -> Dict[str, Any]:
        mus = [r["mu_ev"] for r in self.records if r["mu_ev"] is not None]
        sigma_mus = [r["sigma_mu_ev"] for r in self.records]
        return {
            "steps_completed": self.steps_completed,
            "requested_steps": self.config.n_steps,
            "completed": self.steps_completed >= self.config.n_steps,
            "stop_reason": self.stop_reason,
            "elapsed_s": elapsed_s,
            "sampled_ns": self.steps_completed * self.config.timestep_fs * 1e-6,
            "targetmu_ev": self.config.targetmu_ev,
            "mu_mean_ev": float(np.mean(mus)) if mus else None,
            "mu_std_ev": float(np.std(mus)) if mus else None,
            "max_abs_mu_error_ev": (
                max(abs(m - self.config.targetmu_ev) for m in mus) if mus else None),
            "sigma_mu_mean_ev": float(np.mean(sigma_mus)) if sigma_mus else 0.0,
            "sigma_mu_max_ev": float(np.max(sigma_mus)) if sigma_mus else 0.0,
            "audit_frames": list(self.audit_frames),
            "trajectory_path": str(traj_path),
            "mu_log_path": str(mu_log_path),
        }

    def committee_stats(self) -> CommitteeStats:
        committee = self._committee()
        if committee is None:
            return CommitteeStats(sigma_force_ev_per_angstrom=0.0, n_members=0,
                                  n_samples=0)
        return committee.window_stats(aggregate="mean")


def run_constant_potential_md(atoms, config: CPMDConfig,
                              state_point: Optional[StatePoint] = None,
                              cp_mace_repo: Optional[str] = None,
                              hooks: Optional[List[Hook]] = None) -> MDResult:
    """Run one constant-potential trajectory and return an MDResult.

    Free energies are NOT computed here -- data.harvest.harvest_md_run reads
    the COLVAR afterwards, so there is exactly one code path that turns a
    trajectory into a barrier.
    """
    driver = ConstantPotentialMD(atoms, config, cp_mace_repo=cp_mace_repo, hooks=hooks)
    summary = driver.run()

    sp_id = state_point_id(state_point) if state_point else Path(config.save_dir).name
    status = RunStatus.DONE if summary["completed"] else RunStatus.FAILED
    if summary["stop_reason"] and "threshold" in str(summary["stop_reason"]):
        status = RunStatus.FAILED

    return MDResult(
        state_point_id=sp_id,
        status=status,
        converged=False,          # only harvest_md_run may set this True
        sampled_ns=summary["sampled_ns"],
        n_steps=summary["steps_completed"],
        timestep_fs=config.timestep_fs,
        temperature_k=config.temperature_k,
        ensemble="constant_potential",
        achieved_mu_mean_ev=summary["mu_mean_ev"],
        achieved_mu_std_ev=summary["mu_std_ev"],
        committee=driver.committee_stats(),
        engine="cp-mace-nosehoover" + ("+plumed" if config.use_plumed else ""),
        trajectory_path=summary["trajectory_path"],
        run_dir=str(config.save_dir),
        walltime_s=summary["elapsed_s"],
    )
