"""
Minimal planner agent: reads a workflow YAML spec and dispatches to
SystemAgent, CPDFTAgent, MLIPAgent, MDOpesAgent, and AnalysisAgent in order.

This module is intentionally thin: it is glue/orchestration code, not the
scientific core of the project (see docs/ROADMAP.md).

TODO:
- Load workflow YAML (see workflows/cu100_dimerization.yaml).
- Instantiate and call sub-agents in the correct order.
- Handle HPC submission (GH200 node vs A100 cluster) via a Scheduler stub.
"""

import yaml


class PlannerAgent:
    def __init__(self, workflow_path: str):
        with open(workflow_path) as f:
            self.spec = yaml.safe_load(f)

    def run(self):
        raise NotImplementedError("Phase 4: implement agent dispatch sequence")
