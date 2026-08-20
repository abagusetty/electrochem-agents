"""
Record types for a constant-potential electrocatalysis campaign.

Design rules:
  * Plain dataclasses. No I/O, no numpy in the serialised form -- arrays are
    stored as nested lists so a record survives JSON round-trip unchanged.
  * Units are in the field name. `energy_ev`, `mu_ev`, `sigma_uc_cm2`.
    Ambiguous units are how sign-convention bugs survive to publication.
  * Every record carries provenance: what code, which model, which commit.
    A label whose origin you cannot reconstruct is not a label.
  * `converged` is never inferred from a job's exit status. A JDFTx run can
    exit 0 having not converged. Only the parser sets it.
"""

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

# Physical constants (CODATA).
HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903
ELEMENTARY_CHARGE_C = 1.602176634e-19


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNCONVERGED = "unconverged"   # ran to completion, did not converge
    MISSING = "missing"           # expected output absent


@dataclass
class StatePoint:
    """One point in the (facet x potential/charge x cation x coverage) space.

    This is the unit the acquisition policy chooses over, the unit a run
    directory is named after, and the join key between MD results and CP-DFT
    labels. Keep it hashable and stable -- `state_point_id` depends on it.
    """

    facet: str                                  # "100", "310", ...
    cation: Optional[str] = None                # "Li" | "Na" | "K" | "Cs" | None
    n_cation: int = 0
    # Exactly one of these two is the control variable; the other is measured.
    surface_charge_uc_cm2: Optional[float] = None   # constant-charge control
    target_mu_ev: Optional[float] = None            # constant-potential control
    potential_v_she: Optional[float] = None         # calibrated, if known
    co_coverage_ml: Optional[float] = None          # monolayers
    nx: int = 8
    ny: int = 8
    n_layers: int = 5
    solvent: str = "water"
    solvent_depth_angstrom: float = 8.0
    seed: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def ensemble(self) -> str:
        """'constant_potential' if target-mu drives the run, else
        'constant_charge'. Refuses to guess when both or neither is set."""
        has_mu = self.target_mu_ev is not None
        has_q = self.surface_charge_uc_cm2 is not None
        if has_mu and not has_q:
            return "constant_potential"
        if has_q and not has_mu:
            return "constant_charge"
        if has_mu and has_q:
            return "constant_potential"   # mu is the control; q is diagnostic
        raise ValueError(
            f"StatePoint {self.facet!r} sets neither surface_charge_uc_cm2 nor "
            "target_mu_ev -- the ensemble is undefined. Set exactly one."
        )

    @property
    def cell_label(self) -> str:
        return f"{self.nx}x{self.ny}"

    @property
    def condition_label(self) -> str:
        """Matches agents.reasoning.REFERENCE_VALUES keys ('neutral', '-23')."""
        if self.surface_charge_uc_cm2 is None:
            return "neutral" if self.target_mu_ev is None else f"mu{self.target_mu_ev:+.3f}"
        if abs(self.surface_charge_uc_cm2) < 1e-6:
            return "neutral"
        return f"{self.surface_charge_uc_cm2:.0f}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StatePoint":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def state_point_id(sp: StatePoint) -> str:
    """Deterministic, filesystem-safe, human-readable id.

    Readable prefix so a human can scan `ls runs/`, hash suffix so distinct
    state points can never collide into one directory.
    """
    control = (f"mu{sp.target_mu_ev:+.3f}" if sp.target_mu_ev is not None
               else f"q{sp.surface_charge_uc_cm2:+.1f}")
    cation = f"{sp.cation}{sp.n_cation}" if sp.cation else "nocat"
    cov = f"cov{sp.co_coverage_ml:.2f}" if sp.co_coverage_ml is not None else "cov0"
    prefix = f"cu{sp.facet}_{sp.cell_label}_{control}_{cation}_{cov}_s{sp.seed}"
    prefix = prefix.replace(".", "p").replace("+", "p").replace("-", "m")
    digest = hashlib.sha1(
        repr(sorted(sp.to_dict().items())).encode()).hexdigest()[:8]
    return f"{prefix}_{digest}"


@dataclass
class CommitteeStats:
    """Committee spread over an MLIP ensemble.

    sigma_mu_ev is the reason this project exists. It is the standard
    deviation across committee members of the PREDICTED FERMI LEVEL -- i.e.
    disagreement about the electronic boundary condition, which a
    constant-charge committee cannot produce because there is no mu to
    disagree about. It is the acquisition signal (see acquisition.sigma_mu).
    """

    sigma_force_ev_per_angstrom: float          # max or mean over atoms; see reduction
    sigma_energy_ev: float = 0.0
    sigma_mu_ev: float = 0.0
    mean_mu_ev: Optional[float] = None
    n_members: int = 0
    reduction: str = "max"                      # "max" | "mean" over atoms
    n_samples: int = 1                          # frames aggregated

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommitteeStats":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class CPDFTLabel:
    """One grand-canonical DFT single point. The reusable scientific asset.

    `n_electrons` is what JDFTx settled on at fixed `target_mu_ev`; together
    with mu it is exactly what FermiMACE training consumes
    (`electron=` / `potential=` in extended-XYZ).
    """

    state_point_id: str
    energy_ev: float
    mu_ev: float
    forces_ev_per_angstrom: List[List[float]]
    positions_angstrom: List[List[float]]
    species: List[str]
    cell_angstrom: List[List[float]]
    n_electrons: Optional[float] = None
    # WHICH energy `energy_ev` actually is. For a target-mu run JDFTx reports the
    # GRAND POTENTIAL G = F - mu*N, not the Helmholtz free energy F. In a shipped
    # pymatgen fixture the two differ by ~1683 eV. Mixing G-valued and F-valued
    # labels in one training set, or comparing G against a canonical MLIP energy,
    # is silent and catastrophic -- hence this field is mandatory in spirit and
    # `is_usable()` refuses labels where it is unknown.
    etype: Optional[str] = None          # "G" | "F" | "Etot"
    efermi_ev: Optional[float] = None
    is_gc: bool = True
    gc_converged: bool = False           # is_gc AND converged AND mu on target
    converged: bool = False
    target_mu_ev: Optional[float] = None
    fluid_model: Optional[str] = None
    status: RunStatus = RunStatus.DONE

    # Provenance -- non-optional in spirit even where typed Optional.
    source: str = "jdftx"
    code_version: Optional[str] = None
    pseudo_set: Optional[str] = None
    run_dir: Optional[str] = None
    walltime_s: Optional[float] = None
    n_nodes: Optional[int] = None
    frame_index: Optional[int] = None           # if drawn from a trajectory
    parent_trajectory: Optional[str] = None

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def mu_error_ev(self) -> Optional[float]:
        """How far the achieved mu sits from the requested one. A large value
        means the grand-canonical solve did not actually hit the target
        potential, and the label should not be trusted."""
        if self.target_mu_ev is None:
            return None
        return self.mu_ev - self.target_mu_ev

    def is_usable(self, mu_tol_ev: float = 0.05,
                  require_etype: bool = True) -> bool:
        """Gate before a label enters a training set.

        Three independent conditions, all necessary:
          * the SCF converged;
          * the achieved mu is within tolerance of the requested one -- a
            grand-canonical run that missed its target potential is not a
            label at that potential;
          * the energy type is known, so G-valued and F-valued labels are
            never silently mixed.
        """
        if not self.converged or self.status != RunStatus.DONE:
            return False
        if require_etype and self.etype is None:
            return False
        err = self.mu_error_ev
        return err is None or abs(err) <= mu_tol_ev

    @property
    def helmholtz_f_ev(self) -> Optional[float]:
        """F = G + mu*N, recovered from a grand-canonical label.

        Use this, never `energy_ev`, when comparing a target-mu label against a
        canonical (fixed-electron-count) energy such as an OC25 or OMol25
        reference. Returns None when the conversion is not well defined.
        """
        if self.etype == "F":
            return self.energy_ev
        if self.etype != "G" or self.n_electrons is None:
            return None
        return self.energy_ev + self.mu_ev * self.n_electrons

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CPDFTLabel":
        d = dict(d)
        if "status" in d and not isinstance(d["status"], RunStatus):
            d["status"] = RunStatus(d["status"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class MDResult:
    """One biased-MD campaign at one state point.

    `barrier_ev` / `reaction_energy_ev` line up with
    agents.reasoning.SimulationRecord so a finished MDResult converts to an
    exploration record without reinterpretation.
    """

    state_point_id: str
    barrier_ev: Optional[float] = None
    reaction_energy_ev: Optional[float] = None
    barrier_ev_std: Optional[float] = None
    reaction_energy_ev_std: Optional[float] = None
    converged: bool = False
    status: RunStatus = RunStatus.DONE

    sampled_ns: float = 0.0
    n_steps: int = 0
    timestep_fs: float = 0.5
    temperature_k: float = 300.0
    ensemble: str = "constant_charge"
    achieved_mu_mean_ev: Optional[float] = None
    achieved_mu_std_ev: Optional[float] = None   # tracking error vs targetmu
    committee: Optional[CommitteeStats] = None
    # Fraction of sampled frames passing analysis.interface_validity (IF-valid).
    # None = not checked. A trajectory that leaves physical validity is not a
    # cheaper trajectory, it is a wrong one -- see the hard veto in
    # acquisition.sigma_mu.
    if_valid_fraction: Optional[float] = None
    first_invalid_frame: Optional[int] = None

    mlip: str = "esen-oc25"
    mlip_members: List[str] = field(default_factory=list)
    engine: str = "ase+plumed"
    opes_barrier_ev: Optional[float] = None
    opes_pace: Optional[int] = None
    cv_name: str = "c_c_distance"

    trajectory_path: Optional[str] = None
    colvar_path: Optional[str] = None
    run_dir: Optional[str] = None
    walltime_s: Optional[float] = None
    n_nodes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        if self.committee is not None:
            d["committee"] = self.committee.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MDResult":
        d = dict(d)
        if "status" in d and not isinstance(d["status"], RunStatus):
            d["status"] = RunStatus(d["status"])
        if isinstance(d.get("committee"), dict):
            d["committee"] = CommitteeStats.from_dict(d["committee"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_simulation_record(self, state_point: StatePoint):
        """Convert to agents.reasoning.SimulationRecord for the exploration
        agents. Imported lazily so data/ stays independent of agents/."""
        from agents.reasoning import SimulationRecord
        if self.barrier_ev is None or self.reaction_energy_ev is None:
            raise ValueError(
                f"MDResult {self.state_point_id} has no barrier/reaction energy; "
                "it cannot become a SimulationRecord. Check harvest status "
                f"(got {self.status.value})."
            )
        return SimulationRecord(
            facet=state_point.facet,
            surface_charge_density=(state_point.surface_charge_uc_cm2
                                    if state_point.surface_charge_uc_cm2 is not None
                                    else 0.0),
            cation=state_point.cation,
            barrier_ev=self.barrier_ev,
            reaction_energy_ev=self.reaction_energy_ev,
            barrier_ev_std=self.barrier_ev_std,
            reaction_energy_ev_std=self.reaction_energy_ev_std,
            converged=self.converged,
            potential_v=state_point.potential_v_she,
        )


def hartree_to_ev(x: float) -> float:
    return x * HARTREE_EV


def ev_to_hartree(x: float) -> float:
    return x / HARTREE_EV
