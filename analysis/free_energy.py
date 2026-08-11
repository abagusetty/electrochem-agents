"""
Free-energy and structural analysis utilities.

TODO:
- Extract dG_barrier, dG_reaction from reweighted OPES FES.
- Compute water orientation distributions in the inner Helmholtz layer.
- Plot dG vs surface charge (constant-charge) and dG vs potential (constant-potential)
  on the same axes for direct comparison, replicating Fig. 3(c)/Fig. 5 style analysis
  from Sahoo et al. (arXiv:2509.17862).
"""


def extract_barrier_and_reaction_energy(fes_curve):
    raise NotImplementedError("Phase 1: implement FES peak/well extraction")


def water_orientation_distribution(trajectory, z_range_angstrom=(0.0, 4.5)):
    raise NotImplementedError("Phase 1: implement orientation histogram")
