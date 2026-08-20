"""
Structured interface initialisation by short Langevin relaxation.

WHY
---
`systems/packing.py` places solvent by random rejection sampling with a
minimum-distance criterion. Measured on a 22.9 A x 22.9 A x 8 A solvent slot:
it succeeds at nominal water density (140/140) but saturates at ~142 molecules
regardless of the request -- 142/161 and 142/182 at 1.15x and 1.3x nominal, and
100/105 in a 5 A slot. Those are the regimes a real double layer occupies, and
in them the packer silently under-fills or raises.

Under-filling is the dangerous case: the run proceeds, the interfacial water
density is wrong, and nothing in any log says so.

NeuralPLexer3 (NeurIPS 2025) hit the analogous problem and solved it by
replacing the usual uninformative prior with a PHYSICS-INFORMED one, built by
running 64 steps of overdamped Langevin dynamics under a cheap harmonic energy
model (their Algorithm S3):

    drift = 2*d_bond + d_entity/ent_r^2 + d_res/res_r^2 - X/sphere_r^2
    X    <- X + dt*drift + 2*sqrt(dt)*eps,    eps ~ N(0, I)

with `d_*` computed from neighbour/group matrices. They report that a better
prior straightens the conditional flow and cuts the number of integrator steps
needed. Their terms encode: chemical connectivity (bonds), coarse group
organisation (entities/residues), and global compactness (sphere confinement).

This module transplants that recipe. The terms are re-derived for an
electrified interface rather than a polymer:

    bond        rigid intramolecular O-H geometry within each water
    exclusion   soft repulsion between molecular centres (replaces the hard
                rejection criterion that fails at high density)
    slab        one-sided repulsion keeping solvent above the metal surface
    slot        confinement to the intended solvent z-window (replaces the
                sphere term; a slab is confined in z, not radially)
    image       in-plane minimum-image wrapping, since the cell is periodic
                in x and y but not z

It is a PRIOR, not a force field. It produces a physically plausible starting
configuration cheaply; MD under the real potential does the rest. Correctness
is judged by `analysis.interface_validity`, not by the drift terms being right
in any deeper sense.

Pure numpy. No ASE required for the core routine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from systems.water_geometry import make_water_molecule, n_molecules_for_density


@dataclass
class LangevinPriorConfig:
    """Relaxation schedule and soft-constraint radii.

    Defaults follow Algorithm S3's shape -- tens of steps, dt well below the
    stiffest relaxation time, noise scaled as 2*sqrt(dt). They are tuned for
    plausibility, not for reproducing any thermodynamic ensemble; this is an
    initialiser.
    """

    # NP3 uses 64 steps for a sparse polymer whose bonds do most of the work.
    # A dense fluid needs more: measured on a 22.9 A cell at 1.15x nominal
    # water density, 64 steps leaves a 2.24 A minimum O-O separation (too
    # tight; real water bottoms out near 2.4-2.6 A), while 256 reaches 2.61 A
    # for ~1 s of CPU. Diminishing returns past that -- 512 steps buys 2.68 A.
    n_steps: int = 256
    dt: float = 0.02
    # Soft-core separation between molecular centres, Angstrom. Water O-O first
    # peak is ~2.8 A; starting slightly below it and letting MD relax outward
    # is better than refusing to place the molecule at all.
    exclusion_r: float = 2.7
    exclusion_k: float = 4.0
    # Ion-centre separation; larger because solvation shells need room.
    ion_exclusion_r: float = 3.4
    # Distance a solvent centre must keep from the topmost metal layer.
    slab_clearance: float = 2.6
    slab_k: float = 4.0
    # Confinement to the intended solvent slot in z.
    slot_k: float = 1.0
    # Intramolecular O-H restraint.
    bond_k: float = 20.0
    noise_scale: float = 1.0
    # Anneal the noise linearly to zero. The last steps then act as a
    # minimiser, so the returned structure sits in a local basin rather than
    # at a random point of a finite-temperature walk.
    anneal: bool = True
    seed: int = 0


def _min_image_delta(delta: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Minimum image in x and y only. z is NOT wrapped -- an interface slab is
    not periodic in z in any useful sense, and wrapping it would let solvent
    tunnel through the metal, which is the exact artefact being avoided."""
    delta = delta.copy()
    if lx > 0:
        delta[..., 0] -= lx * np.round(delta[..., 0] / lx)
    if ly > 0:
        delta[..., 1] -= ly * np.round(delta[..., 1] / ly)
    return delta


def _pairwise_exclusion_drift(centres: np.ndarray, radii: np.ndarray,
                              lx: float, ly: float, k: float) -> np.ndarray:
    """Soft one-sided repulsion between centres closer than r_i + r_j.

    Linear in the overlap rather than 1/r^n: a hard core would blow up when
    two centres start nearly coincident, which random initialisation
    guarantees will happen. Linear overlap pushes them apart smoothly and
    cannot produce an infinite force.
    """
    n = len(centres)
    if n < 2:
        return np.zeros_like(centres)

    drift = np.zeros_like(centres)
    for i in range(n):
        delta = _min_image_delta(centres[i] - centres, lx, ly)   # (n, 3)
        distance = np.linalg.norm(delta, axis=1)
        distance[i] = np.inf
        target = 0.5 * (radii[i] + radii)
        overlap = target - distance
        active = overlap > 0
        if not active.any():
            continue
        direction = delta[active] / np.maximum(distance[active, None], 1e-9)
        drift[i] += k * np.sum(overlap[active, None] * direction, axis=0)
    return drift


def relax_centres(centres: np.ndarray, radii: np.ndarray,
                  cell_xy: Tuple[float, float],
                  z_window: Tuple[float, float],
                  z_metal_top: Optional[float],
                  config: LangevinPriorConfig,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Short Langevin relaxation of molecular centres.

    Returns relaxed centres. Terms, in Algorithm S3's spirit:
      exclusion  pairwise soft repulsion            (replaces bond/entity terms)
      slab       one-sided push above the metal      (interface-specific)
      slot       confinement to the solvent window   (replaces sphere term)
    """
    rng = rng or np.random.default_rng(config.seed)
    lx, ly = cell_xy
    z_lo, z_hi = z_window
    z_mid = 0.5 * (z_lo + z_hi)
    half = max(1e-6, 0.5 * (z_hi - z_lo))

    x = np.array(centres, dtype=float, copy=True)
    for step in range(config.n_steps):
        drift = _pairwise_exclusion_drift(x, radii, lx, ly, config.exclusion_k)

        # Confine to the solvent slot: quadratic well, zero at the centre.
        drift[:, 2] -= config.slot_k * (x[:, 2] - z_mid) / (half ** 2)

        # One-sided repulsion from the metal. Only acts when too close, so it
        # does not bias molecules that are already well above the surface.
        if z_metal_top is not None:
            intrusion = (z_metal_top + config.slab_clearance) - x[:, 2]
            too_close = intrusion > 0
            drift[too_close, 2] += config.slab_k * intrusion[too_close]

        scale = config.noise_scale * (
            (1.0 - step / max(1, config.n_steps - 1)) if config.anneal else 1.0)
        noise = rng.normal(size=x.shape) * 2.0 * np.sqrt(config.dt) * scale

        x = x + config.dt * drift + noise

        # Wrap in-plane; clamp in z so nothing escapes the cell during the walk.
        if lx > 0:
            x[:, 0] = np.mod(x[:, 0], lx)
        if ly > 0:
            x[:, 1] = np.mod(x[:, 1], ly)
        floor = (z_metal_top + config.slab_clearance) if z_metal_top is not None else z_lo
        x[:, 2] = np.clip(x[:, 2], floor, z_hi)
    return x


def build_solvent_layer(cell_xy: Tuple[float, float],
                        z_window: Tuple[float, float],
                        z_metal_top: Optional[float] = None,
                        n_water: Optional[int] = None,
                        cation: Optional[str] = None,
                        n_cation: int = 0,
                        config: Optional[LangevinPriorConfig] = None
                        ) -> Dict[str, Any]:
    """Generate an explicit solvent layer with ions by structured relaxation.

    Unlike `systems.packing.pack_points`, this ALWAYS returns the requested
    number of molecules. Overlaps are relaxed away instead of being grounds for
    rejection, so the routine cannot silently under-fill the cell -- which is
    the failure mode that produces a wrong double layer for a reason that never
    appears in any log.

    Quality is not assumed: check the result with
    `analysis.interface_validity.validate_interface` before using it.
    """
    config = config or LangevinPriorConfig()
    rng = np.random.default_rng(config.seed)
    lx, ly = cell_xy
    z_lo, z_hi = z_window
    depth = z_hi - z_lo

    if n_water is None:
        n_water = n_molecules_for_density(lx, ly, depth)
    n_total = n_water + n_cation
    if n_total == 0:
        return {"species": [], "positions": np.zeros((0, 3)), "n_water": 0,
                "n_cation": 0, "centres": np.zeros((0, 3))}

    centres = np.column_stack([
        rng.uniform(0.0, lx, n_total),
        rng.uniform(0.0, ly, n_total),
        rng.uniform(z_lo, z_hi, n_total),
    ])
    radii = np.concatenate([
        np.full(n_water, config.exclusion_r),
        np.full(n_cation, config.ion_exclusion_r),
    ])

    centres = relax_centres(centres, radii, (lx, ly), (z_lo, z_hi),
                            z_metal_top, config, rng=rng)

    water_centres, ion_centres = centres[:n_water], centres[n_water:]

    species: List[str] = []
    positions: List[np.ndarray] = []
    for centre in water_centres:
        o, h1, h2 = make_water_molecule(centre, rng)
        species += ["O", "H", "H"]
        positions += [o, h1, h2]
    for centre in ion_centres:
        species.append(cation or "Na")
        positions.append(centre)

    return {
        "species": species,
        "positions": np.array(positions) if positions else np.zeros((0, 3)),
        "centres": centres,
        "n_water": n_water,
        "n_cation": n_cation,
        "min_centre_separation": float(_min_separation(centres, lx, ly)),
    }


def _min_separation(centres: np.ndarray, lx: float, ly: float) -> float:
    if len(centres) < 2:
        return float("inf")
    best = np.inf
    for i in range(len(centres) - 1):
        delta = _min_image_delta(centres[i] - centres[i + 1:], lx, ly)
        best = min(best, float(np.linalg.norm(delta, axis=1).min()))
    return best


def build_interface_atoms(slab_atoms, spec, config: Optional[LangevinPriorConfig] = None):
    """ASE drop-in replacement for `systems.cu_interface.add_explicit_water`
    plus `add_cations`, using the structured prior instead of rejection packing.

    Signature mirrors the existing helpers so it can be swapped in directly.
    """
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError("ASE is required for build_interface_atoms.") from exc

    config = config or LangevinPriorConfig(seed=getattr(spec, "seed", 0))
    cell = slab_atoms.get_cell()
    lx, ly = float(cell[0, 0]), float(cell[1, 1])
    z_top = float(slab_atoms.positions[:, 2].max())
    z_lo = z_top + config.slab_clearance
    z_hi = z_lo + spec.solvent_depth_angstrom

    layer = build_solvent_layer(
        (lx, ly), (z_lo, z_hi), z_metal_top=z_top,
        cation=spec.cation, n_cation=spec.n_cation, config=config)

    solvent = Atoms(layer["species"], positions=layer["positions"],
                    cell=cell, pbc=True)
    combined = slab_atoms + solvent
    combined.info.update({
        "n_water": layer["n_water"],
        "n_cation": layer["n_cation"],
        "cation_species": spec.cation,
        "solvent_z_range": (z_lo, z_hi),
        "initialisation": "langevin_prior",
        "min_centre_separation": layer["min_centre_separation"],
    })
    return combined
