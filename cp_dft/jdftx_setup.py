"""
Build JDFTx inputs for grand-canonical (constant-potential) calculations on
explicit-solvent Cu interfaces.

Protocol follows the conventions used by the BEAST DB grand-canonical
electrocatalysis database (arXiv:2405.20239), whose author list includes
JDFTx's lead author -- matching an established protocol buys comparability
that a bespoke one does not.

Key tags:
  target-mu <mu_hartree>       grand-canonical: fix electron chemical
                               potential, let the electron count float.
                               HARTREE, not eV.
  fluid LinearPCM              continuum solvent...
  pcm-variant CANDLE           ...in the CANDLE parameterisation, recommended
                               for charged/polar solutes, handles cation/anion
                               charge asymmetry, keeps the cell neutral under
                               grand-canonical conditions.
  fluid-cation / fluid-anion   electrolyte ions at a molar concentration.
  coulomb-interaction Slab 001 truncate Coulomb along z so periodic slab
                               images do not interact -- mandatory for a
                               charged slab, where the spurious interaction
                               would otherwise diverge with charge.

>>> NOTHING HERE HAS BEEN RUN. <<< Tag spellings and defaults are written
from the JDFTx documentation and should be checked against `jdftx -t` for
your build before a production sweep. Two in particular:
  * `target-mu` sign convention -- see cp_dft.calibration.
  * whether your build wants `pcm-variant CANDLE` with `fluid LinearPCM`
    or exposes CANDLE as a distinct `fluid` value.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from cp_dft.calibration import HARTREE_EV, PotentialCalibration, UNCALIBRATED
from data.schema import StatePoint, state_point_id


@dataclass
class JDFTxProtocol:
    """Convergence and physics settings held FIXED across a campaign.

    Every label in a training set must share these. Mixing cutoffs or
    functionals inside one dataset teaches the model the difference between
    protocols rather than the physics.
    """

    # Electronic structure
    elec_ex_corr: str = "gga-PBE"
    elec_cutoff_eh: float = 20.0            # wavefunction cutoff, Hartree
    elec_cutoff_rho_eh: float = 100.0       # charge-density cutoff, Hartree
    elec_smearing: str = "Fermi"
    elec_smearing_width_eh: float = 0.01    # ~0.27 eV; metals need smearing
    spin_polarized: bool = False
    kpoint_folding: tuple = (1, 1, 1)       # large cells -> Gamma only

    # Pseudopotentials
    pseudo_set: str = "GBRV-pbe"
    pseudo_pattern: str = "GBRV/$ID_pbe.uspp"

    # Fluid / electrolyte
    fluid: str = "LinearPCM"
    pcm_variant: str = "CANDLE"
    fluid_solvent: str = "H2O"
    fluid_cation: Optional[str] = "Na+"
    fluid_anion: Optional[str] = "F-"
    fluid_concentration_mol_l: float = 1.0

    # Slab electrostatics
    coulomb_interaction: str = "Slab 001"
    coulomb_truncation_embed: bool = True

    # Minimisation
    elec_n_iterations: int = 200
    fluid_n_iterations: int = 100
    elec_energy_diff_threshold: float = 1e-7
    symmetries: str = "none"                # MD snapshots have no symmetry
    # target-mu's optional second argument. `no` (default) does direct
    # variational grand-canonical minimisation; `yes` falls back to secant
    # fixed-charge iterations, which is slower but more robust when the direct
    # solve fails to converge.
    target_mu_outer_loop: bool = False

    # Output
    dump_frequency: str = "End"
    dump_vars: List[str] = field(
        default_factory=lambda: ["State", "Forces", "Ecomponents", "ElecDensity"])

    def fingerprint(self) -> str:
        """Stable hash of the protocol. Store it with every label; if it
        changes mid-campaign, the dataset has been silently split in two."""
        import hashlib
        return hashlib.sha1(
            repr(sorted(asdict(self).items())).encode()).hexdigest()[:12]


@dataclass
class JDFTxCalcSpec:
    """One JDFTx calculation: a structure, a protocol, and a potential."""

    state_point: StatePoint
    protocol: JDFTxProtocol = field(default_factory=JDFTxProtocol)
    target_mu_hartree: Optional[float] = None   # None -> neutral / constant charge
    structure_path: Optional[str] = None        # CIF/POSCAR/xyz on disk
    frame_index: Optional[int] = None
    parent_trajectory: Optional[str] = None
    initial_state_dir: Optional[str] = None     # warm start from a prior run
    extra_tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def calc_id(self) -> str:
        base = state_point_id(self.state_point)
        return base if self.frame_index is None else f"{base}_f{self.frame_index:05d}"

    @property
    def target_mu_ev(self) -> Optional[float]:
        return (None if self.target_mu_hartree is None
                else self.target_mu_hartree * HARTREE_EV)


def build_tag_dict(spec: JDFTxCalcSpec) -> Dict[str, Any]:
    """Assemble the JDFTx tag dictionary. Pure -- no I/O, unit-testable."""
    p = spec.protocol
    tags: Dict[str, Any] = {
        "elec-ex-corr": p.elec_ex_corr,
        "elec-cutoff": f"{p.elec_cutoff_eh:g} {p.elec_cutoff_rho_eh:g}",
        "elec-smearing": f"{p.elec_smearing} {p.elec_smearing_width_eh:g}",
        "elec-n-bands": "",                       # let JDFTx choose
        "kpoint-folding": " ".join(str(k) for k in p.kpoint_folding),
        "symmetries": p.symmetries,
        "ion-species": p.pseudo_pattern,
        "coulomb-interaction": p.coulomb_interaction,
        "electronic-minimize": (
            f"nIterations {p.elec_n_iterations} "
            f"energyDiffThreshold {p.elec_energy_diff_threshold:g}"),
    }
    tags = {k: v for k, v in tags.items() if v != ""}

    if p.coulomb_truncation_embed:
        # Centre of mass; JDFTx wants the embedding centre in lattice coords.
        tags["coulomb-truncation-embed"] = "0.5 0.5 0.5"
    if p.spin_polarized:
        tags["spintype"] = "z-spin"

    # Fluid
    if p.fluid and p.fluid.lower() != "none":
        tags["fluid"] = p.fluid
        if p.pcm_variant:
            tags["pcm-variant"] = p.pcm_variant
        tags["fluid-solvent"] = p.fluid_solvent
        tags["fluid-minimize"] = f"nIterations {p.fluid_n_iterations}"
        if p.fluid_cation:
            tags["fluid-cation"] = f"{p.fluid_cation} {p.fluid_concentration_mol_l:g}"
        if p.fluid_anion:
            tags["fluid-anion"] = f"{p.fluid_anion} {p.fluid_concentration_mol_l:g}"

    # Grand canonical.
    #
    # JDFTx HARD-GATES target-mu in the parser (parser.cpp safeProcess -> die()):
    #   REQUIRES  elec-smearing, fluid-cation, fluid-anion
    #             (fluid-cation/anion transitively require fluid-solvent)
    #   FORBIDS   elec-initial-charge, fix-electron-density, fix-electron-potential
    # These are fatal errors, not warnings, and defaults do NOT satisfy them
    # (processDefaults does not mark a command as encountered). There is no
    # supported vacuum or implicit-free grand-canonical path.
    #
    # Consequence to disclose in any manuscript: every JDFTx GC-DFT label
    # carries an IMPLICIT-ELECTROLYTE approximation, so it is not directly
    # comparable to an explicit-ion reference such as OC25 without a stated
    # cross-walk.
    if spec.target_mu_hartree is not None:
        value = f"{spec.target_mu_hartree:.8f}"
        if p.target_mu_outer_loop:
            value += " yes"      # secant fixed-charge fallback
        tags["target-mu"] = value

    if spec.initial_state_dir:
        tags["initial-state"] = str(Path(spec.initial_state_dir) / "jdft.$VAR")

    tags["dump-name"] = "jdft.$VAR"
    tags["dump"] = f"{p.dump_frequency} {' '.join(p.dump_vars)}"

    tags.update(spec.extra_tags)

    # Validate AFTER extra_tags merge: a forbidden tag injected there must be
    # caught, and a required tag supplied there must count as satisfied.
    if "target-mu" in tags:
        validate_gc_tags(tags)
    return tags


GC_REQUIRED_TAGS = ("elec-smearing", "fluid", "fluid-solvent",
                    "fluid-cation", "fluid-anion")
GC_FORBIDDEN_TAGS = ("elec-initial-charge", "fix-electron-density",
                     "fix-electron-potential")


def validate_gc_tags(tags: Dict[str, Any]) -> None:
    """Fail here, not 200 queued jobs later.

    JDFTx enforces these dependencies itself, but only once the job is running
    on a compute node -- by which point a sweep has already consumed its
    allocation producing "Command target-mu is missing dependencies {...}" in
    every log. Checking at input-construction time turns a wasted allocation
    into an immediate error.
    """
    missing = [t for t in GC_REQUIRED_TAGS if t not in tags]
    present_forbidden = [t for t in GC_FORBIDDEN_TAGS if t in tags]
    problems = []
    if missing:
        problems.append(
            "target-mu REQUIRES these tags, which are absent: "
            + ", ".join(missing)
            + ". Grand-canonical DFT in JDFTx is hard-gated on an electrolyte; "
              "set JDFTxProtocol.fluid / fluid_solvent / fluid_cation / "
              "fluid_anion and a non-zero elec_smearing_width_eh.")
    if present_forbidden:
        problems.append(
            "target-mu FORBIDS these tags, which are present: "
            + ", ".join(present_forbidden)
            + ". Remove them from JDFTxCalcSpec.extra_tags.")
    if problems:
        raise ValueError(
            "Invalid grand-canonical JDFTx input:\n  " + "\n  ".join(problems))


def render_input(spec: JDFTxCalcSpec, structure) -> str:
    """Render a complete JDFTx input file as text, via pymatgen.

    pymatgen.io.jdftx is the primary path (it is what atomate2 uses); this
    returns text rather than writing, so callers can diff inputs before a
    sweep -- worth doing once, since a wrong tag replicated across 500 runs
    is an expensive mistake.
    """
    try:
        from pymatgen.io.jdftx.inputs import JDFTXInfile
    except ImportError as exc:
        raise ImportError(
            "pymatgen>=2025.4 is required (pymatgen.io.jdftx). "
            "Install/upgrade pymatgen."
        ) from exc

    infile = JDFTXInfile.from_structure(structure)
    for tag, value in build_tag_dict(spec).items():
        infile[tag] = value
    return str(infile)


def write_input(spec: JDFTxCalcSpec, structure, run_dir,
                filename: str = "in") -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(render_input(spec, structure))
    return path


# ---------------------------------------------------------------------------
# Sweep construction
# ---------------------------------------------------------------------------

def build_potential_sweep(base_state_point: StatePoint,
                          u_she_values: List[float],
                          calibration: PotentialCalibration = UNCALIBRATED,
                          protocol: Optional[JDFTxProtocol] = None
                          ) -> List[JDFTxCalcSpec]:
    """One JDFTxCalcSpec per requested electrode potential.

    Warns (does not fail) on an uncalibrated calibration: sweeps at relative
    potentials are a legitimate early-stage activity, publishing absolute ones
    from them is not.
    """
    import warnings

    if not calibration.calibrated:
        warnings.warn(
            "build_potential_sweep called with an UNCALIBRATED potential "
            "calibration. target-mu values will be internally consistent but "
            "their absolute potentials are not meaningful. Calibrate against "
            "a reference Cu slab before reporting.",
            RuntimeWarning, stacklevel=2,
        )
    protocol = protocol or JDFTxProtocol()
    specs = []
    for u in u_she_values:
        mu_hartree = calibration.u_she_to_target_mu_hartree(u)
        sp = StatePoint(**{**base_state_point.to_dict(),
                           "target_mu_ev": mu_hartree * HARTREE_EV,
                           "potential_v_she": u if calibration.calibrated else None})
        specs.append(JDFTxCalcSpec(state_point=sp, protocol=protocol,
                                   target_mu_hartree=mu_hartree))
    return specs


def build_pzc_reference(facet: str, nx: int = 4, ny: int = 4, n_layers: int = 5,
                        protocol: Optional[JDFTxProtocol] = None) -> JDFTxCalcSpec:
    """A neutral, solvent-free-slab reference calculation for calibration.

    Deliberately small (4x4 by default) and neutral: its only job is to yield
    a trustworthy mu at the point of zero charge for
    cp_dft.calibration.calibrate_from_reference_slab. Running this, and
    checking it, is the first real calculation of the campaign.
    """
    sp = StatePoint(facet=facet, nx=nx, ny=ny, n_layers=n_layers,
                    surface_charge_uc_cm2=0.0, solvent_depth_angstrom=0.0,
                    tags={"role": "pzc_reference"})
    return JDFTxCalcSpec(state_point=sp, protocol=protocol or JDFTxProtocol(),
                         target_mu_hartree=None)


def frames_from_trajectory(trajectory_path, indices: List[int],
                           base_state_point: StatePoint,
                           target_mu_hartree: Optional[float],
                           protocol: Optional[JDFTxProtocol] = None
                           ) -> List[JDFTxCalcSpec]:
    """Specs for auditing selected MD frames with grand-canonical DFT.

    This is the acquisition policy's output path: `indices` are the frames the
    sigma_mu scheduler flagged. Frames are not read here -- only referenced --
    so spec construction stays cheap and side-effect free.
    """
    protocol = protocol or JDFTxProtocol()
    return [
        JDFTxCalcSpec(
            state_point=base_state_point, protocol=protocol,
            target_mu_hartree=target_mu_hartree,
            frame_index=int(i), parent_trajectory=str(trajectory_path),
        )
        for i in indices
    ]
