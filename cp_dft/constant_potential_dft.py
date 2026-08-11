"""
Constant-potential / grand-canonical DFT wrapper.

Intended backends (pick one available in your environment):
- GPAW (inner-potential / electrochemistry extensions)
- JDFTx (native grand-canonical electron reservoir)
- CP2K (fully converged constant-potential / FCP algorithm)

TODO:
- Implement `prepare_input(structure, target_potential)`.
- Implement `run()` (local or HPC-submitted).
- Implement `parse_results()` -> energies, forces, workfunction/Fermi level.
"""

from dataclasses import dataclass


@dataclass
class CPDFTJob:
    structure_path: str
    target_potential_v: float
    backend: str = "gpaw"


def prepare_input(job: CPDFTJob, out_dir: str):
    raise NotImplementedError("Phase 2: implement backend-specific input writer")


def run(job: CPDFTJob):
    raise NotImplementedError("Phase 2: implement job submission/execution")


def parse_results(out_dir: str):
    """Return dict with energy, forces, workfunction, potential."""
    raise NotImplementedError("Phase 2: implement backend-specific output parser")
