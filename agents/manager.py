"""
Top-level orchestrator for the electrochemical simulation agentic
workflow, mirroring LAMMPS-Agents' manager/GroupChat pattern
(github.com/ANL-NST/LAMMPS-Agents: a "manager" ConversableAgent with
explicit workflow-order rules in its system message, coordinating
specialist agents through AutoGen's GroupChat/GroupChatManager).

This module is intentionally thin glue: it builds the agent group and
starts the conversation. The actual reasoning is in agents/system_messages.py
(embedded domain rules) and agents/reasoning.py (deterministic reference
comparisons); the actual physics is in systems/, mlip/, md/, cp_dft/,
analysis/. Consistent with keeping the agentic harness a small part of
this project.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from agents.agent_factory import ElectrochemAgentFactory, LLMConfig


@dataclass
class WorkflowRequest:
    """High-level task specification, e.g. loaded from a workflow YAML
    (see workflows/cu100_dimerization.yaml, workflows/cu310_cpmace.yaml).
    """
    task_description: str
    facet: str
    cations: list = field(default_factory=list)
    cation_counts: list = field(default_factory=list)
    cell: str = "8x8"
    mlip: str = "esen-oc25"
    audit_with_cp_dft: bool = False
    trajectory_length_ns: float = 7.0


class ElectrochemWorkflowManager:
    """Builds the manager + specialist agents and runs a GroupChat for a
    given WorkflowRequest.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.factory = ElectrochemAgentFactory(llm_config=llm_config)

    def build_group_chat(self, max_round: int = 30):
        try:
            from autogen import GroupChat, GroupChatManager
        except ImportError as exc:
            raise ImportError("AG2/AutoGen is required (`pip install ag2`).") from exc

        manager_agent = self.factory.build_manager_agent()
        system_builder = self.factory.build_system_builder_agent()
        mlip_agent = self.factory.build_mlip_agent()
        sampling_agent = self.factory.build_enhanced_sampling_agent()
        analyst_agent = self.factory.build_results_analyst_agent()
        validation_agent = self.factory.build_validation_agent()
        user_proxy = self.factory.build_user_proxy()

        group_chat = GroupChat(
            agents=[
                manager_agent, system_builder, mlip_agent, sampling_agent,
                analyst_agent, validation_agent, user_proxy,
            ],
            messages=[],
            max_round=max_round,
        )
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.factory.llm_config.to_autogen_config(),
        )
        return group_chat_manager, user_proxy

    def run(self, request: WorkflowRequest, max_round: int = 30):
        """Kick off the multi-agent workflow for `request`. Returns the
        AutoGen chat result object (contains the full conversation
        history, which includes every tool call and every agent's
        reasoning about convergence/validity along the way).
        """
        group_chat_manager, user_proxy = self.build_group_chat(max_round=max_round)

        prompt = (
            f"Task: {request.task_description}\n"
            f"Facet: Cu({request.facet})\n"
            f"Cell: {request.cell}\n"
            f"Cations: {request.cations} (counts: {request.cation_counts})\n"
            f"MLIP: {request.mlip}\n"
            f"CP-DFT audit requested: {request.audit_with_cp_dft}\n"
            f"Target trajectory length: {request.trajectory_length_ns} ns\n"
            "Follow the workflow order and validation rules in your system "
            "messages. Report final barrier/reaction energy only after the "
            "Results Analyst Agent confirms convergence."
        )
        return user_proxy.initiate_chat(group_chat_manager, message=prompt)
