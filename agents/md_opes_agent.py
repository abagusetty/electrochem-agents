"""
MDOpesAgent: drives LAMMPS + OPES runs and applies convergence-based
hyperparameter control / early stopping.
"""


class MDOpesAgent:
    def run_and_monitor(self, config: dict):
        raise NotImplementedError("Phase 4: call md.lammps_runner + md.opes_runner with convergence loop")
