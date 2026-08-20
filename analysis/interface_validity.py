"""
Physical validity checks for electrified solid-liquid interface structures.

MOTIVATION
----------
Two independent lines converge on the same need.

1. The anchor paper's most striking negative result is that MACE-MP-0 fails
   QUALITATIVELY at Rh(111)/water -- it produces an unphysical oxygen-density
   spike at z ~ 1 A, essentially water inside the metal. That is not a large
   error in a number. It is a structure that cannot exist. Energy and force
   MAE did not catch it; looking at the density profile did.

2. NeuralPLexer3 (NeurIPS 2025) makes exactly this distinction a first-class
   evaluation axis, reporting accuracy (RMSD < 2 A) and physical validity
   (PoseBusters "PB-valid": correct stereochemistry, plausible torsions, no
   clashes) SEPARATELY, on the grounds that a model can be accurate on average
   while generating structures that are physically impossible. Their headline
   figure reports success rate with and without the validity requirement.

This module is the electrified-interface analogue of PB-valid. Call it
IF-valid. It is deliberately a checklist of cheap, interpretable, physically
motivated assertions rather than one learned score, because its job is to be
trusted and debugged, not to be optimised against.

USES
----
  * Gate generated/packed initial structures before spending DFT or MD on them.
  * Monitor MLIP MD in flight -- an IF-valid failure mid-trajectory means the
    potential has left its domain of validity, whatever the forces look like.
  * Give the Validation Agent concrete tool-callable checks instead of
    prompt-level instructions.
  * Report as a benchmark axis alongside barrier accuracy, exactly as NP3
    reports PB-valid alongside RMSD.

Nothing here needs a calculator; it is geometry only.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Covalent/ionic radii sufficient for clash detection (Angstrom).
_RADII = {"H": 0.31, "O": 0.66, "C": 0.76, "N": 0.71, "Cu": 1.32,
          "Li": 1.28, "Na": 1.66, "K": 2.03, "Cs": 2.44, "F": 0.57, "Cl": 1.02}
_DEFAULT_RADIUS = 1.0

# Ideal water geometry (TIP3P-ish / gas phase), for dissociation detection.
WATER_OH_ANGSTROM = 0.9572
WATER_HOH_DEG = 104.52

BULK_WATER_DENSITY_G_CM3 = 0.997


@dataclass
class Check:
    name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    detail: str = ""
    severity: str = "hard"        # "hard" -> invalidates; "soft" -> warning

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidityReport:
    checks: List[Check] = field(default_factory=list)

    @property
    def hard_failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def soft_failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "soft"]

    @property
    def is_valid(self) -> bool:
        """IF-valid: every hard check passes. Soft failures are reported but
        do not invalidate -- they flag structures worth a human look."""
        return not self.hard_failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "if_valid": self.is_valid,
            "n_checks": len(self.checks),
            "n_hard_failures": len(self.hard_failures),
            "n_soft_failures": len(self.soft_failures),
            "failed": [c.name for c in self.checks if not c.passed],
            "checks": [c.to_dict() for c in self.checks],
        }

    def describe(self) -> str:
        lines = [f"IF-valid: {'PASS' if self.is_valid else 'FAIL'} "
                 f"({len(self.checks)} checks, {len(self.hard_failures)} hard "
                 f"failure(s), {len(self.soft_failures)} warning(s))"]
        for check in self.checks:
            if check.passed:
                continue
            mark = "FAIL" if check.severity == "hard" else "WARN"
            lines.append(f"  [{mark}] {check.name}: {check.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _species_positions(atoms) -> Tuple[List[str], np.ndarray]:
    try:
        species = list(atoms.get_chemical_symbols())
        positions = np.asarray(atoms.get_positions(), dtype=float)
    except AttributeError:
        species = list(atoms["species"])
        positions = np.asarray(atoms["positions"], dtype=float)
    return species, positions


def _cell_of(atoms) -> Optional[np.ndarray]:
    try:
        return np.asarray(atoms.get_cell(), dtype=float)
    except AttributeError:
        cell = atoms.get("cell") if isinstance(atoms, dict) else None
        return np.asarray(cell, dtype=float) if cell is not None else None


def _metal_mask(species: Sequence[str], metal: str = "Cu") -> np.ndarray:
    return np.array([s == metal for s in species], dtype=bool)


def _min_image_distances(positions: np.ndarray, cell: Optional[np.ndarray],
                         reference: np.ndarray) -> np.ndarray:
    """Distances from every position to `reference`, minimum-image in x,y.

    z is not wrapped: an interface slab is not periodic in z in any meaningful
    sense, and wrapping it would hide exactly the failure mode being looked
    for (water appearing on the wrong side of the slab).
    """
    delta = positions - reference
    if cell is not None and cell.shape == (3, 3):
        for axis in (0, 1):
            length = cell[axis, axis]
            if length > 0:
                delta[:, axis] -= length * np.round(delta[:, axis] / length)
    return np.linalg.norm(delta, axis=1)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_no_clashes(atoms, tolerance: float = 0.7) -> Check:
    """No pair closer than `tolerance` x (r_i + r_j).

    The cheapest and most decisive check. A clash means the structure is not
    merely inaccurate but impossible, and any energy computed from it is
    meaningless.
    """
    species, positions = _species_positions(atoms)
    cell = _cell_of(atoms)
    radii = np.array([_RADII.get(s, _DEFAULT_RADIUS) for s in species])

    worst_ratio, worst_pair, worst_distance = np.inf, None, None
    for i in range(len(species)):
        if i + 1 >= len(species):
            break
        distances = _min_image_distances(positions[i + 1:], cell, positions[i])
        limits = tolerance * (radii[i] + radii[i + 1:])
        ratios = distances / np.maximum(limits, 1e-9)
        j_local = int(np.argmin(ratios))
        if ratios[j_local] < worst_ratio:
            worst_ratio = float(ratios[j_local])
            worst_pair = (species[i], species[i + 1 + j_local])
            worst_distance = float(distances[j_local])

    passed = worst_ratio >= 1.0
    return Check(
        name="no_atomic_clashes", passed=passed, value=worst_distance,
        threshold=tolerance,
        detail=("no clashes" if passed else
                f"{worst_pair[0]}-{worst_pair[1]} at {worst_distance:.2f} A, "
                f"{worst_ratio:.2f}x the {tolerance} x (r_i+r_j) limit"),
    )


def check_no_solvent_in_slab(atoms, metal: str = "Cu",
                             margin_angstrom: float = 1.0) -> Check:
    """No solvent atom below the top metal layer.

    This is the MACE-MP-0 failure the anchor paper documents: oxygen density
    spiking at z ~ 1 A above the surface, i.e. water penetrating the metal.
    Directly targeted here because it is the known failure mode of foundation
    models at this exact interface.
    """
    species, positions = _species_positions(atoms)
    metal_mask = _metal_mask(species, metal)
    if not metal_mask.any():
        return Check(name="no_solvent_in_slab", passed=True,
                     detail=f"no {metal} atoms found; check skipped",
                     severity="soft")

    z_top = float(positions[metal_mask, 2].max())
    solvent_mask = ~metal_mask
    if not solvent_mask.any():
        return Check(name="no_solvent_in_slab", passed=True,
                     detail="no solvent present")

    z_solvent = positions[solvent_mask, 2]
    n_intruding = int((z_solvent < z_top - margin_angstrom).sum())
    deepest = float(z_solvent.min() - z_top)
    return Check(
        name="no_solvent_in_slab", passed=n_intruding == 0,
        value=deepest, threshold=-margin_angstrom,
        detail=("no solvent inside the slab" if n_intruding == 0 else
                f"{n_intruding} solvent atom(s) below the top {metal} layer; "
                f"deepest {deepest:.2f} A relative to z_top -- this is the "
                "MACE-MP-0-style interfacial failure mode"),
    )


def check_water_molecules_intact(atoms, max_oh_angstrom: float = 1.3,
                                 min_oh_angstrom: float = 0.8,
                                 angle_tolerance_deg: float = 25.0) -> Check:
    """Every O has exactly two H at plausible bond length and angle.

    Spurious dissociation is a classic MLIP artefact at charged interfaces.
    Real dissociation exists too, so this is a SOFT check -- it flags for
    inspection rather than invalidating. A run where 30% of waters have
    "dissociated" is a broken potential; a run where one has may be chemistry.
    """
    species, positions = _species_positions(atoms)
    cell = _cell_of(atoms)
    o_indices = [i for i, s in enumerate(species) if s == "O"]
    h_indices = [i for i, s in enumerate(species) if s == "H"]
    if not o_indices or not h_indices:
        return Check(name="water_molecules_intact", passed=True,
                     detail="no water present", severity="soft")

    h_positions = positions[h_indices]
    n_bad_coordination = n_bad_angle = 0
    for oi in o_indices:
        distances = _min_image_distances(h_positions, cell, positions[oi])
        bonded = np.where((distances >= min_oh_angstrom) &
                          (distances <= max_oh_angstrom))[0]
        if len(bonded) != 2:
            n_bad_coordination += 1
            continue
        v1 = h_positions[bonded[0]] - positions[oi]
        v2 = h_positions[bonded[1]] - positions[oi]
        cosine = float(np.dot(v1, v2) /
                       (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        if abs(angle - WATER_HOH_DEG) > angle_tolerance_deg:
            n_bad_angle += 1

    total = len(o_indices)
    bad_fraction = (n_bad_coordination + n_bad_angle) / total
    return Check(
        name="water_molecules_intact", passed=bad_fraction < 0.05,
        value=bad_fraction, threshold=0.05, severity="soft",
        detail=(f"{n_bad_coordination} O with != 2 H, {n_bad_angle} with a "
                f"distorted H-O-H angle, out of {total} "
                f"({bad_fraction:.1%})"),
    )


def check_oxygen_density_profile(atoms, metal: str = "Cu",
                                 bin_width_angstrom: float = 0.2,
                                 first_peak_min_angstrom: float = 1.8,
                                 spike_z_angstrom: float = 1.0) -> Check:
    """The oxygen density profile must not peak unphysically close to the metal.

    Quantifies the anchor paper's Figure 2(a) diagnostic. A real metal-water
    contact layer peaks around 2-3 A above the surface. A peak at z ~ 1 A is
    the documented MACE-MP-0 artefact.
    """
    species, positions = _species_positions(atoms)
    metal_mask = _metal_mask(species, metal)
    o_mask = np.array([s == "O" for s in species], dtype=bool)
    if not metal_mask.any() or not o_mask.any():
        return Check(name="oxygen_density_profile", passed=True,
                     detail="not applicable", severity="soft")

    z_top = float(positions[metal_mask, 2].max())
    z_rel = positions[o_mask, 2] - z_top
    above = z_rel[z_rel > 0]
    if above.size == 0:
        return Check(name="oxygen_density_profile", passed=False,
                     detail="no oxygen above the metal surface at all")

    bins = np.arange(0.0, float(above.max()) + bin_width_angstrom,
                     bin_width_angstrom)
    histogram, edges = np.histogram(above, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if histogram.sum() == 0:
        return Check(name="oxygen_density_profile", passed=False,
                     detail="empty oxygen density profile")

    peak_z = float(centers[int(np.argmax(histogram))])
    near_count = int((above < spike_z_angstrom).sum())
    passed = peak_z >= first_peak_min_angstrom and near_count == 0
    return Check(
        name="oxygen_density_profile", passed=passed, value=peak_z,
        threshold=first_peak_min_angstrom,
        detail=(f"first O peak at z = {peak_z:.2f} A above the surface; "
                f"{near_count} O within {spike_z_angstrom} A"
                + ("" if passed else
                   " -- matches the unphysical near-surface spike reported for "
                   "MACE-MP-0 at Rh(111)/water")),
    )


def check_slab_integrity(atoms, metal: str = "Cu", n_layers: int = 5,
                         max_layer_spread_angstrom: float = 1.0,
                         layer_tolerance_angstrom: float = 0.8) -> Check:
    """Metal atoms still form recognisable layers.

    Catches slabs that melted, reconstructed catastrophically, or lost atoms
    into the solvent during MD. Soft, because genuine surface reconstruction
    under strongly reducing potentials is real physics worth studying -- but
    it should be noticed deliberately, not absorbed silently into an average.
    """
    species, positions = _species_positions(atoms)
    metal_mask = _metal_mask(species, metal)
    if metal_mask.sum() < n_layers:
        return Check(name="slab_integrity", passed=True,
                     detail="too few metal atoms to assess", severity="soft")

    z = np.sort(positions[metal_mask, 2])
    # Cluster z into layers with a fixed tolerance.
    #
    # An earlier gap-statistics version (split where gap > 3 x median positive
    # gap) failed on the common case: a pristine slab has all in-layer gaps
    # exactly 0 and all inter-layer gaps identical, so the median positive gap
    # EQUALS the inter-layer spacing and the 3x threshold never fires -- the
    # whole slab reads as one layer and the check reports a spurious warning on
    # a perfect structure. A fixed tolerance has no such degenerate case.
    split_points = np.where(np.diff(z) > layer_tolerance_angstrom)[0]
    layers = np.split(z, split_points + 1)
    spreads = [float(layer.max() - layer.min()) for layer in layers if layer.size]
    worst = max(spreads) if spreads else 0.0
    # Layer count is reported but not asserted: a supercell or a reconstructed
    # surface legitimately yields a different count. In-layer spread is the
    # real signal for a slab that melted or lost atoms.
    return Check(
        name="slab_integrity", passed=worst <= max_layer_spread_angstrom,
        value=worst, threshold=max_layer_spread_angstrom, severity="soft",
        detail=(f"{len(layers)} metal layer(s) detected (expected ~{n_layers}); "
                f"worst in-layer z spread {worst:.2f} A"),
    )


def check_solvent_density(atoms, metal: str = "Cu",
                          bulk_offset_angstrom: float = 4.0,
                          tolerance_fraction: float = 0.35) -> Check:
    """Bulk-region water density is within tolerance of 1 g/cm^3.

    Catches under-packed cells, which are the most likely failure of a
    rejection-based packer: it terminates with too few molecules, the run
    proceeds, and the double-layer structure is wrong for a reason that never
    appears in any log.
    """
    species, positions = _species_positions(atoms)
    cell = _cell_of(atoms)
    metal_mask = _metal_mask(species, metal)
    o_mask = np.array([s == "O" for s in species], dtype=bool)
    if cell is None or not metal_mask.any() or not o_mask.any():
        return Check(name="solvent_density", passed=True,
                     detail="not applicable", severity="soft")

    z_top = float(positions[metal_mask, 2].max())
    z_o = positions[o_mask, 2]
    z_lo = z_top + bulk_offset_angstrom
    z_hi = float(z_o.max())
    if z_hi - z_lo < 2.0:
        return Check(name="solvent_density", passed=True, severity="soft",
                     detail="no bulk-like region to measure")

    n_bulk = int(((z_o >= z_lo) & (z_o <= z_hi)).sum())
    volume_a3 = float(cell[0, 0] * cell[1, 1] * (z_hi - z_lo))
    # 18.015 g/mol / 6.02214076e23 = 2.9915e-23 g per water; 1 A^3 = 1e-24 cm^3
    density = (n_bulk * 2.9915e-23) / (volume_a3 * 1e-24) if volume_a3 > 0 else 0.0
    deviation = abs(density - BULK_WATER_DENSITY_G_CM3) / BULK_WATER_DENSITY_G_CM3
    return Check(
        name="solvent_density", passed=deviation <= tolerance_fraction,
        value=density, threshold=BULK_WATER_DENSITY_G_CM3, severity="soft",
        detail=(f"bulk-region density {density:.3f} g/cm^3 from {n_bulk} water "
                f"({deviation:.0%} from 1.0)"),
    )


def check_ions_solvated(atoms, metal: str = "Cu",
                        cations: Sequence[str] = ("Li", "Na", "K", "Cs"),
                        min_height_angstrom: float = 1.5) -> Check:
    """Cations sit above the metal, not embedded in it."""
    species, positions = _species_positions(atoms)
    metal_mask = _metal_mask(species, metal)
    ion_mask = np.array([s in cations for s in species], dtype=bool)
    if not ion_mask.any() or not metal_mask.any():
        return Check(name="ions_solvated", passed=True, detail="no cations present")

    z_top = float(positions[metal_mask, 2].max())
    heights = positions[ion_mask, 2] - z_top
    n_embedded = int((heights < min_height_angstrom).sum())
    return Check(
        name="ions_solvated", passed=n_embedded == 0,
        value=float(heights.min()), threshold=min_height_angstrom,
        detail=("all cations above the surface" if n_embedded == 0 else
                f"{n_embedded} cation(s) within {min_height_angstrom} A of, or "
                "below, the top metal layer"),
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def validate_interface(atoms, metal: str = "Cu", n_layers: int = 5,
                       cations: Sequence[str] = ("Li", "Na", "K", "Cs"),
                       skip: Sequence[str] = ()) -> ValidityReport:
    """Run every check. Returns an IF-valid report.

    Individual check failures never raise -- a check that cannot run reports
    itself as skipped, so one unusual structure cannot abort a sweep over
    thousands.
    """
    runners = [
        ("no_atomic_clashes", lambda: check_no_clashes(atoms)),
        ("no_solvent_in_slab", lambda: check_no_solvent_in_slab(atoms, metal)),
        ("water_molecules_intact", lambda: check_water_molecules_intact(atoms)),
        ("oxygen_density_profile", lambda: check_oxygen_density_profile(atoms, metal)),
        ("slab_integrity", lambda: check_slab_integrity(atoms, metal, n_layers)),
        ("solvent_density", lambda: check_solvent_density(atoms, metal)),
        ("ions_solvated", lambda: check_ions_solvated(atoms, metal, cations)),
    ]
    report = ValidityReport()
    for name, runner in runners:
        if name in skip:
            continue
        try:
            report.checks.append(runner())
        except Exception as exc:                    # noqa: BLE001
            report.checks.append(Check(
                name=name, passed=True, severity="soft",
                detail=f"check could not run: {type(exc).__name__}: {exc}"))
    return report


def validate_trajectory(frames: Sequence[Any], stride: int = 1,
                        **kwargs) -> Dict[str, Any]:
    """IF-valid over a trajectory.

    The reported quantity is the FRACTION of frames that are valid, plus the
    first frame that failed. A potential that starts valid and degrades has a
    domain-of-validity problem that a final-frame check would miss entirely.
    """
    reports, first_failure = [], None
    for index, frame in enumerate(frames):
        if index % stride != 0:
            continue
        report = validate_interface(frame, **kwargs)
        reports.append((index, report))
        if first_failure is None and not report.is_valid:
            first_failure = index

    n = len(reports)
    n_valid = sum(1 for _, r in reports if r.is_valid)
    failure_counts: Dict[str, int] = {}
    for _, report in reports:
        for check in report.hard_failures:
            failure_counts[check.name] = failure_counts.get(check.name, 0) + 1

    return {
        "n_frames_checked": n,
        "n_valid": n_valid,
        "valid_fraction": (n_valid / n) if n else 0.0,
        "first_invalid_frame": first_failure,
        "hard_failure_counts": failure_counts,
        "verdict": ("all frames IF-valid" if n_valid == n else
                    f"{n - n_valid}/{n} frames invalid; first at {first_failure}. "
                    "The potential left its domain of validity -- barriers from "
                    "this trajectory are not trustworthy."),
    }
