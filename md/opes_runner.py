"""
PLUMED OPES enhanced-sampling driver for the CO-CO dimerization CV,
following the protocol in Sahoo et al. (arXiv:2509.17862, Methods 5.4).

TODO:
- Write PLUMED input (OPES_METAD, adaptive Gaussian kernels, upper wall at 6 A).
- Support convergence diagnostics: block uncertainty, time-resolved dG(t).
- Expose barrier/reaction-energy extraction from reweighted FES.
"""

from dataclasses import dataclass


@dataclass
class OPESConfig:
    cv_atom_indices: tuple[int, int]
    barrier_ev: float = 5.0
    pace: int = 500
    upper_wall_angstrom: float = 6.0
    n_steps: int = 15_000_000  # ~7.5 ns at 0.5 fs timestep


def write_plumed_input(config: OPESConfig, out_path: str):
    raise NotImplementedError("Phase 1: implement PLUMED OPES input writer")


def check_convergence(fes_blocks, tol_ev: float = 0.02) -> bool:
    """Return True if block-resolved free energies have converged within tol_ev."""
    raise NotImplementedError("Phase 4: implement block-uncertainty convergence check")
