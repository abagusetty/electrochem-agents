"""
Top-level orchestrator for the electrochemical simulation agentic
workflow, combining LAMMPS-Agents' manager/GroupChat pattern with the
abductive/inductive exploration-agent split from Fei et al.
(arXiv:2604.11957). LLM access is via ALCF Inference Endpoints.

This module is intentionally thin glue: it builds the agent group,
tracks the accumulated dataset, and decides when to swap the
PatternFindingAgent for the BOAssistedPatternFindingAgent. The actual
reasoning is in agents/system_messages.py and agents/reasoning.py; the
actual physics is in systems/, mlip/, md/, cp_dft/, analysis/.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from agents.agent_factory import ElectrochemAgentFactory
from agents.llm_backend import ALCFLLMConfig
from agents.reasoning import SimulationRecord


@dataclass
class WorkflowRequest:
    task_description: str
    facet: str
    cations: list = field(default_factory=list)
    cation_counts: list = field(default_factory=list)
    cell: str = "8x8"
    mlip: str = "esen-oc25"
    audit_with_cp_dft: bool = False
    trajectory_length_ns: float = 7.0


class ElectrochemWorkflowManager:
    """Builds the manager + specialist + exploration agents and runs a
    GroupChat for a given WorkflowRequest.

    `bo_transition_threshold` (default 30) sets how many converged
    SimulationRecords must accumulate in `self.accumulated_records`
    before the PatternFindingAgent is swapped for the
    BOAssistedPatternFindingAgent -- mirroring arXiv:2604.11957's
    heuristic transition (289 of 352 samples in their campaign), scaled
    down here since each of our records is a full OPES campaign, not a
    single synthesis run.
    """

    def __init__(self, reasoning_llm_config: Optional[ALCFLLMConfig] = None,
                 tool_llm_config: Optional[ALCFLLMConfig] = None,
                 bo_transition_threshold: int = 30):
        self.factory = ElectrochemAgentFactory(
            reasoning_llm_config=reasoning_llm_config, tool_llm_config=tool_llm_config,
        )
        self.bo_transition_threshold = bo_transition_threshold
        self.accumulated_records: List[SimulationRecord] = []

    def should_use_bo_assisted_agent(self) -> bool:
        return len(self.accumulated_records) >= self.bo_transition_threshold

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
        abnormality_agent = self.factory.build_abnormality_detection_agent()

        if self.should_use_bo_assisted_agent():
            pattern_agent = self.factory.build_bo_assisted_pattern_finding_agent()
        else:
            pattern_agent = self.factory.build_pattern_finding_agent()

        user_proxy = self.factory.build_user_proxy()

        group_chat = GroupChat(
            agents=[
                manager_agent, system_builder, mlip_agent, sampling_agent,
                analyst_agent, validation_agent, abnormality_agent,
                pattern_agent, user_proxy,
            ],
            messages=[], max_round=max_round,
        )
        group_chat_manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.factory.reasoning_llm_config.to_autogen_config(),
        )
        return group_chat_manager, user_proxy

    def record_result(self, record: SimulationRecord) -> None:
        if record.converged:
            self.accumulated_records.append(record)

    def run(self, request: WorkflowRequest, max_round: int = 30):
        group_chat_manager, user_proxy = self.build_group_chat(max_round=max_round)

        prompt = (
            f"Task: {request.task_description}\n"
            f"Facet: Cu({request.facet})\n"
            f"Cell: {request.cell}\n"
            f"Cations: {request.cations} (counts: {request.cation_counts})\n"
            f"MLIP: {request.mlip}\n"
            f"CP-DFT audit requested: {request.audit_with_cp_dft}\n"
            f"Target trajectory length: {request.trajectory_length_ns} ns\n"
            f"Accumulated converged records so far: {len(self.accumulated_records)} "
            f"(BO-assisted pattern-finding active: {self.should_use_bo_assisted_agent()})\n"
            "Follow the workflow order and validation rules in your system "
            "messages. Report final barrier/reaction energy only after the "
            "Results Analyst Agent confirms convergence. After convergence, "
            "run both the Abnormality-Detection Agent and the (BO-assisted) "
            "Pattern-Finding Agent, and report which agent proposed each "
            "next state point and why."
        )
        return user_proxy.initiate_chat(group_chat_manager, message=prompt)
