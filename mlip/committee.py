"""
MLIP committee calculator: the sigma_mu machinery.

WHY THIS MODULE EXISTS
----------------------
Reading CP-MACE's own metadynamics driver
(simulation/metadynamics/simulate.py) shows it runs TWO trained models
through an `AverageForceCalculator` that averages forces and energies and
reports both the force standard deviation AND the variation in chemical
potential across members.

That second quantity is the point. Call it sigma_mu: committee disagreement
about the FERMI LEVEL -- i.e. about the electronic boundary condition itself.
A constant-charge committee cannot produce it, because in a constant-charge
model there is no mu to disagree about. It is therefore an uncertainty signal
that is native to constant-potential simulation, and it is what this project
uses to decide when a state point has earned an expensive grand-canonical
DFT label (see acquisition.sigma_mu).

INTERFACE CONTRACT
------------------
CP-MACE's NoseHoover integrator calls, verbatim from its step():

    self.Vne[1] = self.Vne[0] + (self.targetmu
                                 - self.atoms.get_calculator().get_mu()) \
                  * self.delT / 2 / self.Mne

so ANY calculator driven by that integrator MUST implement `get_mu()`.
CommitteeCalculator does, returning the committee mean. That single method is
also the only thing a Route-A eSEN mu-head has to provide in order to be
usable as a drop-in constant-potential surrogate.

Nothing here has been run against a trained model.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from data.schema import CommitteeStats


def _base_calculator_class():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError as exc:
        raise ImportError(
            "ASE is required for CommitteeCalculator (it is an ASE Calculator "
            "subclass). Install ase>=3.23.0."
        ) from exc
    return Calculator, all_changes


@dataclass
class CommitteeConfig:
    """How the committee reduces member disagreement to a scalar.

    force_reduction:
        'max'  -- the largest per-atom force disagreement. Sensitive to a
                  single atom in a bad local environment, which is usually
                  exactly what you want to catch. Default.
        'mean' -- average over atoms. Smoother, but a reactive event on two
                  atoms out of 800 is diluted into invisibility.
    """
    force_reduction: str = "max"
    require_mu: bool = True
    # If a member raises, drop it and continue with the rest rather than
    # killing an MD run mid-trajectory. Below min_members, that is fatal --
    # a "committee" of one reports zero uncertainty, which would silently
    # switch the acquisition policy off.
    tolerate_member_failure: bool = True
    min_members: int = 2

    def __post_init__(self):
        if self.force_reduction not in ("max", "mean"):
            raise ValueError("force_reduction must be 'max' or 'mean'.")
        if self.min_members < 2:
            raise ValueError(
                "min_members must be >= 2. A single-member committee reports "
                "zero disagreement, which would disable the acquisition signal "
                "while appearing to work."
            )


def make_committee_calculator(members: Sequence[Any],
                              config: Optional[CommitteeConfig] = None,
                              member_names: Optional[Sequence[str]] = None):
    """Build a CommitteeCalculator.

    A factory rather than a plain class because ASE's Calculator base class
    must be imported lazily -- the rest of this package stays importable
    without ASE installed, which is what lets the unit tests run anywhere.
    """
    Calculator, all_changes = _base_calculator_class()
    config = config or CommitteeConfig()

    if len(members) < config.min_members:
        raise ValueError(
            f"Need at least {config.min_members} committee members, got "
            f"{len(members)}. Train additional members with different seeds; "
            "sigma_mu is meaningless without genuine ensemble spread."
        )

    class CommitteeCalculator(Calculator):
        """Averages member predictions; exposes sigma_F, sigma_E, sigma_mu.

        Mirrors CP-MACE's AverageForceCalculator, with the uncertainty
        bookkeeping promoted from a diagnostic to a first-class output.
        """

        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.members = list(members)
            self.member_names = list(member_names or
                                     [f"member{i}" for i in range(len(members))])
            self.config = config
            # Latest step
            self.sigma_force: float = 0.0
            self.sigma_energy: float = 0.0
            self.sigma_mu: float = 0.0
            self.mean_mu: Optional[float] = None
            self.member_mu: List[float] = []
            self.n_active_members: int = len(self.members)
            # Running history, for per-window aggregation by the driver
            self._history: List[Dict[str, float]] = []

        # -- ASE interface --------------------------------------------------

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            target = self.atoms if atoms is None else atoms

            energies: List[float] = []
            forces: List[np.ndarray] = []
            mus: List[float] = []
            failures: List[str] = []

            for name, member in zip(self.member_names, self.members):
                try:
                    scratch = target.copy()
                    scratch.calc = member
                    energies.append(float(scratch.get_potential_energy()))
                    forces.append(np.asarray(scratch.get_forces(), dtype=float))
                    mu = _member_mu(member, scratch)
                    if mu is not None:
                        mus.append(float(mu))
                except Exception as exc:            # noqa: BLE001
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
                    if not self.config.tolerate_member_failure:
                        raise

            n_ok = len(energies)
            if n_ok < self.config.min_members:
                raise RuntimeError(
                    f"Only {n_ok} of {len(self.members)} committee members "
                    f"evaluated successfully; need {self.config.min_members}. "
                    "Failures:\n  " + "\n  ".join(failures)
                )
            self.n_active_members = n_ok

            energy_array = np.asarray(energies, dtype=float)
            force_stack = np.stack(forces, axis=0)          # (members, atoms, 3)

            mean_energy = float(energy_array.mean())
            mean_forces = force_stack.mean(axis=0)

            # Per-atom disagreement: std over members of each force component,
            # then the vector norm per atom, then reduced over atoms.
            per_component_std = force_stack.std(axis=0, ddof=0)   # (atoms, 3)
            per_atom_sigma = np.linalg.norm(per_component_std, axis=1)
            self.sigma_force = float(
                per_atom_sigma.max() if self.config.force_reduction == "max"
                else per_atom_sigma.mean())
            self.sigma_energy = float(energy_array.std(ddof=0))

            self.member_mu = list(mus)
            if mus:
                mu_array = np.asarray(mus, dtype=float)
                self.mean_mu = float(mu_array.mean())
                self.sigma_mu = float(mu_array.std(ddof=0))
            else:
                self.mean_mu, self.sigma_mu = None, 0.0
                if self.config.require_mu:
                    raise RuntimeError(
                        "No committee member reported a Fermi level, so "
                        "sigma_mu cannot be computed -- the acquisition signal "
                        "this project depends on would be silently zero. Use "
                        "FermiMACE members (or an eSEN mu-head exposing "
                        "get_mu()), or set CommitteeConfig(require_mu=False) "
                        "to run deliberately without it."
                    )

            self._history.append({
                "sigma_force": self.sigma_force,
                "sigma_energy": self.sigma_energy,
                "sigma_mu": self.sigma_mu,
                "mean_mu": self.mean_mu if self.mean_mu is not None else float("nan"),
                "energy": mean_energy,
            })

            self.results["energy"] = mean_energy
            self.results["free_energy"] = mean_energy
            self.results["forces"] = mean_forces

        # -- constant-potential contract ------------------------------------

        def get_mu(self) -> float:
            """Committee-mean Fermi level.

            REQUIRED by CP-MACE's NoseHoover integrator, which reads it every
            step to drive the electron degree of freedom toward `targetmu`.
            """
            if self.mean_mu is None:
                raise RuntimeError(
                    "get_mu() called before any successful calculate(), or no "
                    "member reports a Fermi level. A constant-potential "
                    "integrator cannot run against this calculator."
                )
            return self.mean_mu

        # -- uncertainty bookkeeping -----------------------------------------

        def latest_stats(self) -> CommitteeStats:
            return CommitteeStats(
                sigma_force_ev_per_angstrom=self.sigma_force,
                sigma_energy_ev=self.sigma_energy,
                sigma_mu_ev=self.sigma_mu,
                mean_mu_ev=self.mean_mu,
                n_members=self.n_active_members,
                reduction=self.config.force_reduction,
                n_samples=1,
            )


        def window_stats(self, last_n: Optional[int] = None,
                         aggregate: str = "mean") -> CommitteeStats:
            """Aggregate uncertainty over recent steps.

            The acquisition policy triggers on a WINDOW, not a single frame --
            one noisy step is not evidence, a sustained excursion is.
            `aggregate='max'` is the conservative choice when the question is
            "did this trajectory ever go somewhere the models disagree about".
            """
            history = self._history[-last_n:] if last_n else self._history
            if not history:
                return CommitteeStats(sigma_force_ev_per_angstrom=0.0, n_members=0,
                                      reduction=self.config.force_reduction,
                                      n_samples=0)
            reduce_fn = np.max if aggregate == "max" else np.mean
            mus = [h["mean_mu"] for h in history if not math.isnan(h["mean_mu"])]
            return CommitteeStats(
                sigma_force_ev_per_angstrom=float(
                    reduce_fn([h["sigma_force"] for h in history])),
                sigma_energy_ev=float(reduce_fn([h["sigma_energy"] for h in history])),
                sigma_mu_ev=float(reduce_fn([h["sigma_mu"] for h in history])),
                mean_mu_ev=float(np.mean(mus)) if mus else None,
                n_members=self.n_active_members,
                reduction=self.config.force_reduction,
                n_samples=len(history),
            )

        def history(self) -> List[Dict[str, float]]:
            return list(self._history)

        def reset_history(self) -> None:
            self._history.clear()

        def high_sigma_mu_frames(self, threshold_ev: float) -> List[int]:
            """Indices of recorded steps whose sigma_mu exceeded `threshold_ev`.

            These are the frames the acquisition policy hands to
            cp_dft.jdftx_setup.frames_from_trajectory for grand-canonical
            audit. Frame indices are positions in this calculator's history,
            so the MD driver must record trajectory frames on the same stride.
            """
            return [i for i, h in enumerate(self._history)
                    if h["sigma_mu"] > threshold_ev]

    return CommitteeCalculator()


def _member_mu(member: Any, atoms: Any) -> Optional[float]:
    """Extract a Fermi level from one member, whichever way it exposes one.

    Tried in order:
      * `member.get_mu()`         -- CP-MACE FermiMACE convention
      * `member.results['mu']`    -- ASE results-dict convention
      * `member.get_fermi_level()`-- generic ASE-ish naming
    Returns None if the member has no notion of mu; the caller decides whether
    that is fatal.
    """
    for attribute in ("get_mu", "get_fermi_level"):
        fn = getattr(member, attribute, None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:                       # noqa: BLE001
                continue
    results = getattr(member, "results", None)
    if isinstance(results, dict):
        for key in ("mu", "fermi_level", "potential"):
            if key in results:
                try:
                    return float(results[key])
                except (TypeError, ValueError):
                    continue
    return None


def load_fermi_mace_committee(model_paths: Sequence[str], device: str = "xpu",
                              config: Optional[CommitteeConfig] = None,
                              cp_mace_repo: Optional[str] = None):
    """Committee of trained FermiMACE models.

    Uses CP-MACE's own MACECalculator (an ASE Calculator subclass). If
    `cp_mace_repo` is given, that checkout is prepended to sys.path so the
    CP-MACE fork is used rather than any upstream `mace` that happens to be
    installed -- they share a module name, and picking up stock MACE here
    would silently produce a committee with no Fermi level at all.
    """
    if len(model_paths) < 2:
        raise ValueError(
            f"Committee needs >= 2 models, got {len(model_paths)}. Train "
            "members with different seeds (CP-MACE: vary --seed)."
        )
    if cp_mace_repo:
        import sys
        if cp_mace_repo not in sys.path:
            sys.path.insert(0, cp_mace_repo)
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "Could not import mace.calculators.MACECalculator. Install the "
            "CP-MACE fork (clone github.com/yuanyue-liu-group/CP-MACE, replace "
            "./mace, `pip install ./mace`) and/or pass cp_mace_repo=<path>.\n"
            "NOTE: that repository ships no LICENSE file. Cloning and running "
            "it locally for research is fine; vendoring or redistributing it "
            "is not."
        ) from exc

    members = [MACECalculator(model_paths=[p], device=device) for p in model_paths]
    names = [str(p) for p in model_paths]
    return make_committee_calculator(members, config=config, member_names=names)


def load_esen_committee(model_names: Sequence[str], device: str = "xpu",
                        config: Optional[CommitteeConfig] = None):
    """Committee of eSEN-OC25 members (Route A, before a mu-head exists).

    These have no Fermi level, so `require_mu` is forced off and sigma_mu is
    identically zero. Useful as the force-uncertainty-only ABLATION BASELINE
    (B2 in the research plan) -- which is exactly the comparison that tests
    whether sigma_mu carries information beyond sigma_F. Not usable as a
    constant-potential surrogate.
    """
    from mlip.esen_oc25 import ESENOC25Config, load_esen_oc25_calculator

    if len(model_names) < 2:
        raise ValueError(f"Committee needs >= 2 models, got {len(model_names)}.")
    members = [load_esen_oc25_calculator(ESENOC25Config(model_name=n, device=device))
               for n in model_names]
    cfg = config or CommitteeConfig()
    cfg.require_mu = False
    return make_committee_calculator(members, config=cfg,
                                     member_names=list(model_names))
