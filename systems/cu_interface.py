"""
Structure generation for Cu(100)/Cu(310) explicit-solvent interfaces,
following the OC25 dataset construction recipe (Sahoo et al.,
arXiv:2509.17862, Methods 5.1.1-5.1.2). Requires ASE for slab construction.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from systems.packing import pack_points
from systems.water_geometry import make_water_molecule, n_molecules_for_density

CU_LATTICE_CONSTANT_ANGSTROM = 3.615

CATION_RADIUS_ANGSTROM = {"Li": 0.90, "Na": 1.16, "K": 1.52, "Cs": 1.81}


@dataclass
class InterfaceSpec:
    facet: str
    nx: int
    ny: int
    n_layers: int = 5
    vacuum_angstrom: float = 2.0
    solvent: str = "water"
    solvent_depth_angstrom: float = 8.0
    cation: Optional[str] = None
    n_cation: int = 0
    water_min_dist_angstrom: float = 2.8
    cation_min_dist_angstrom: float = 3.0
    seed: int = 0


def build_cu_slab(spec: InterfaceSpec):
    try:
        from ase.build import bulk, fcc100, surface
    except ImportError as exc:
        raise ImportError(
            "ASE is required to build Cu slabs. Install it via `pip install ase`."
        ) from exc

    if spec.facet == "100":
        slab = fcc100("Cu", size=(spec.nx, spec.ny, spec.n_layers),
                       a=CU_LATTICE_CONSTANT_ANGSTROM, vacuum=spec.vacuum_angstrom)
    else:
        cu_bulk = bulk("Cu", "fcc", a=CU_LATTICE_CONSTANT_ANGSTROM)
        indices = tuple(int(c) for c in spec.facet)
        slab = surface(cu_bulk, indices, spec.n_layers, vacuum=spec.vacuum_angstrom)
        slab = slab.repeat((spec.nx, spec.ny, 1))

    slab.center(axis=2)
    return slab


def add_explicit_water(slab_atoms, spec: InterfaceSpec, rng=None):
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError("ASE is required for add_explicit_water.") from exc

    rng = rng or np.random.default_rng(spec.seed)
    cell = slab_atoms.get_cell()
    lx, ly = cell[0, 0], cell[1, 1]

    z_top = slab_atoms.positions[:, 2].max()
    z_lo = z_top + 2.0
    z_hi = z_lo + spec.solvent_depth_angstrom

    n_water = n_molecules_for_density(lx, ly, spec.solvent_depth_angstrom)
    o_centers, ok, attempts = pack_points(
        (lx, ly), (z_lo, z_hi), n_water,
        min_dist=spec.water_min_dist_angstrom, rng=rng,
    )
    if not ok:
        raise RuntimeError(
            f"Water packing did not reach target density "
            f"({len(o_centers)}/{n_water} placed in {attempts} attempts). "
            "Reduce water_min_dist_angstrom, increase solvent_depth_angstrom, "
            "or switch to a Packmol-based packer for high-density systems."
        )

    water_species, water_positions = [], []
    for o_center in o_centers:
        o, h1, h2 = make_water_molecule(o_center, rng)
        water_species += ["O", "H", "H"]
        water_positions += [o, h1, h2]

    water_atoms = Atoms(water_species, positions=np.array(water_positions), cell=cell, pbc=True)
    combined = slab_atoms + water_atoms
    combined.info["n_water"] = n_water
    combined.info["solvent_z_range"] = (float(z_lo), float(z_hi))
    return combined


def add_cations(interface_atoms, spec: InterfaceSpec, rng=None):
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError("ASE is required for add_cations.") from exc

    if spec.n_cation == 0 or spec.cation is None:
        return interface_atoms

    rng = rng or np.random.default_rng(spec.seed + 1)
    cell = interface_atoms.get_cell()
    lx, ly = cell[0, 0], cell[1, 1]
    z_lo, z_hi = interface_atoms.info["solvent_z_range"]

    existing = interface_atoms.positions
    cation_positions, ok, attempts = pack_points(
        (lx, ly), (z_lo, z_hi), spec.n_cation,
        min_dist=spec.cation_min_dist_angstrom, existing_points=existing, rng=rng,
    )
    if not ok:
        raise RuntimeError(
            f"Cation packing failed ({len(cation_positions)}/{spec.n_cation} placed "
            f"in {attempts} attempts). Increase solvent_depth_angstrom or reduce "
            "cation_min_dist_angstrom."
        )

    cation_atoms = Atoms([spec.cation] * spec.n_cation, positions=cation_positions,
                          cell=cell, pbc=True)
    combined = interface_atoms + cation_atoms
    combined.info.update(interface_atoms.info)
    combined.info["n_cation"] = spec.n_cation
    combined.info["cation_species"] = spec.cation
    return combined


def estimate_surface_charge_density(interface_atoms, n_faces=1):
    elementary_charge_c = 1.602176634e-19
    n_cation = interface_atoms.info.get("n_cation", 0)
    cell = interface_atoms.get_cell()
    area_angstrom2 = cell[0, 0] * cell[1, 1]
    area_cm2 = area_angstrom2 * 1e-16
    charge_c = n_cation * elementary_charge_c
    sigma_c_cm2 = charge_c / (area_cm2 * n_faces)
    return sigma_c_cm2 * 1e6


def build_cu_water_cation_interface(spec: InterfaceSpec, rng=None):
    rng = rng or np.random.default_rng(spec.seed)
    slab = build_cu_slab(spec)
    interface = add_explicit_water(slab, spec, rng=rng)
    interface = add_cations(interface, spec, rng=rng)
    sigma = estimate_surface_charge_density(interface)
    return interface, sigma
