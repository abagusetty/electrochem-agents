"""
LAMMPS MD runner with ML-IAP-backed MLIP forces (eSEN-OC25 or CP-MACE).

TODO:
- Generate LAMMPS input scripts from an InterfaceSpec + calculator choice.
- Support GPU execution on GH200/A100 (ML-IAP-Kokkos).
- Return trajectory handle for downstream OPES/analysis.
"""


def run_md(data_file: str, calculator: str, steps: int, **kwargs):
    raise NotImplementedError("Phase 1: implement LAMMPS input generation + launch")
