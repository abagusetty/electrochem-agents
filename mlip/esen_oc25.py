"""
Wrapper around fairchem OC25-trained eSEN models for constant-charge MD.

TODO:
- Load eSEN-S-cons or eSEN-M-d checkpoint via fairchem model registry.
- Expose a calculator-style interface (ASE Calculator or LAMMPS ML-IAP hook)
  for forces/energies on Cu/water/cation configurations.
"""


def load_esen_oc25_calculator(checkpoint: str = "esen-sm-conserving-all-oc25"):
    """Return an ASE-compatible calculator backed by an OC25 eSEN checkpoint."""
    raise NotImplementedError("Phase 1: wire up fairchem checkpoint loading")
