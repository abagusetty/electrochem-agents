"""
Structure generation for Cu(100)/Cu(310) explicit-solvent interfaces.

Builds on pymatgen for slab generation and ASE for solvent/ion packing,
following the OC25 dataset construction recipe (Sahoo et al., arXiv:2509.17862,
Methods 5.1.1-5.1.2).

TODO:
- Pull Cu bulk structure (pymatgen.ext.matproj or local structure file).
- Generate (100)/(310) slabs via pymatgen.core.surface.SlabGenerator.
- Pack explicit water + alkali cations at target surface charge density
  using ASE + Packmol, matching OC25 solvent depth (5-10 A) conventions.
- Export to LAMMPS data format and/or extended XYZ (for CP-MACE).
"""

from dataclasses import dataclass


@dataclass
class InterfaceSpec:
    facet: str  # e.g. "100" or "310"
    n_water: int
    cation: str | None = None
    n_cation: int = 0
    solvent_depth_angstrom: float = 8.0


def build_cu_interface(spec: InterfaceSpec):
    """Construct a Cu/water/cation interface structure.

    Returns an ASE Atoms object (or pymatgen Structure) ready for
    MLIP relaxation and MD. Not yet implemented.
    """
    raise NotImplementedError("Phase 1: implement slab + solvent packing")
