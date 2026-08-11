"""MLIPAgent: selects/trains eSEN-OC25 or CP-MACE depending on workflow spec."""


class MLIPAgent:
    def get_calculator(self, kind: str):
        raise NotImplementedError("Phase 4: dispatch to mlip.esen_oc25 or mlip.cp_mace_wrapper")
