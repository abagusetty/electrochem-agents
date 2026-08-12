"""
CLI entry point for running an electrochemical simulation workflow through
the agentic harness. Reads a workflow YAML (see
workflows/cu100_dimerization.yaml, workflows/cu310_cpmace.yaml) and starts
the AutoGen/AG2 multi-agent conversation via agents.manager.

Usage:
    python -m workflows.run_workflow workflows/cu100_dimerization.yaml
"""

import sys

import yaml

from agents.manager import ElectrochemWorkflowManager, WorkflowRequest


def load_request(yaml_path: str) -> WorkflowRequest:
    with open(yaml_path) as fh:
        spec = yaml.safe_load(fh)
    return WorkflowRequest(
        task_description=spec.get("task", "electrochemical simulation"),
        facet=str(spec.get("facet", "100")),
        cations=spec.get("cations", []),
        cation_counts=spec.get("cation_counts", []),
        cell=spec.get("cell", "8x8"),
        mlip=spec.get("mlip", "esen-oc25"),
        audit_with_cp_dft=spec.get("audit_with_cp_dft", False),
        trajectory_length_ns=spec.get("trajectory_length_ns", 7.0),
    )


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    request = load_request(sys.argv[1])
    manager = ElectrochemWorkflowManager()
    manager.run(request)


if __name__ == "__main__":
    main()
