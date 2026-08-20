"""
Agent factory for electrochemical interface simulations, combining two
architectural references:
  - github.com/ANL-NST/LAMMPS-Agents: Manager + specialist agents wired
    to Python tool functions via AutoGen/AG2 function-calling.
  - Fei et al., "Agentic LLM Reasoning in a Self-Driving Laboratory for
    Air-Sensitive Lithium Halide Spinel Conductors" (arXiv:2604.11957):
    splitting exploration into abductive (AbnormalityDetectionAgent) and
    inductive (PatternFindingAgent / BOAssistedPatternFindingAgent) modes
    rather than one monolithic decision-maker.

LLM access is via ALCF Inference Endpoints (agents.llm_backend.ALCFLLMConfig).
Model assignment follows ALCF's documented tool-calling (T) / reasoning
(R) capability flags per agent role:
  - Manager, Results Analyst, AbnormalityDetectionAgent, PatternFindingAgent,
    BOAssistedPatternFindingAgent: need both T and R (all of these weigh
    evidence, not just dispatch calls) -> default Qwen/Qwen3-235B-A22B.
  - System Builder, MLIP Agent, Enhanced-Sampling Agent, Validation Agent:
    mostly deterministic tool dispatch -> lighter T-only model
    (meta-llama/Llama-3.3-70B-Instruct by default).

The reasoning content lives in agents/system_messages.py and
agents/reasoning.py, not here -- this module is deliberately thin glue.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agents.llm_backend import ALCFLLMConfig
from agents.vllm_backend import VLLMLLMConfig, VLLMServerConfig
from agents.system_messages import (
    MANAGER_SYSTEM_MESSAGE,
    SYSTEM_BUILDER_SYSTEM_MESSAGE,
    MLIP_AGENT_SYSTEM_MESSAGE,
    ENHANCED_SAMPLING_AGENT_SYSTEM_MESSAGE,
    RESULTS_ANALYST_SYSTEM_MESSAGE,
    VALIDATION_AGENT_SYSTEM_MESSAGE,
    ABNORMALITY_DETECTION_AGENT_SYSTEM_MESSAGE,
    PATTERN_FINDING_AGENT_SYSTEM_MESSAGE,
    BO_ASSISTED_PATTERN_FINDING_AGENT_SYSTEM_MESSAGE,
)

LLMConfig = ALCFLLMConfig  # backward-compatible alias

# ---------------------------------------------------------------------------
# BACKEND SELECTION
# ---------------------------------------------------------------------------
# Default is a LOCAL vLLM-XPU server on the Aurora allocation, not the remote
# ALCF inference endpoint. Reasons, in order (see agents.vllm_backend):
#   * compute nodes have no outbound network -- a remote call hangs, silently,
#     inside the allocation;
#   * `inference_auth_token.py authenticate` is an interactive browser Globus
#     flow no batch job can perform, and tokens expire mid-campaign;
#   * a shared endpoint's models go hot/cold behind other users.
# The `frameworks` module ships vllm 0.26.1+xpu and vllm-xpu-kernels, so this
# costs nothing to install -- only GPU tiles, which must be budgeted.
#
# Set ELECTROCHEM_LLM_BACKEND=alcf to use the remote endpoint instead, e.g.
# when driving from a login node.
LLM_BACKEND = os.environ.get("ELECTROCHEM_LLM_BACKEND", "vllm").lower()

# Remote ALCF endpoint model names.
REASONING_MODEL_DEFAULT = "Qwen/Qwen3-235B-A22B"
TOOL_ONLY_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"

# Local vLLM: a filesystem path, because compute nodes cannot download.
# Stage the weights under $ELECTROCHEM_MODEL_DIR first.
VLLM_REASONING_MODEL_DEFAULT = os.environ.get(
    "ELECTROCHEM_VLLM_MODEL", "Qwen/Qwen3-32B")
VLLM_TOOL_MODEL_DEFAULT = os.environ.get(
    "ELECTROCHEM_VLLM_TOOL_MODEL", VLLM_REASONING_MODEL_DEFAULT)


def default_llm_config(model: str, backend: Optional[str] = None):
    """Build the right config object for the selected backend.

    One server usually serves BOTH roles: the reasoning/tool-only split exists
    because the remote endpoint offers different models with different
    capabilities. Locally there is one server, so both roles point at it unless
    a second is explicitly started -- serving two large models to save on one
    role's token cost is a bad trade when tiles are the scarce resource.
    """
    backend = (backend or LLM_BACKEND).lower()
    if backend == "alcf":
        return ALCFLLMConfig(model=model)
    if backend == "vllm":
        return VLLMLLMConfig(model=model)
    raise ValueError(
        f"Unknown ELECTROCHEM_LLM_BACKEND={backend!r}; expected 'vllm' (local "
        "vLLM-XPU server, default) or 'alcf' (remote inference endpoint).")


class ElectrochemAgentFactory:
    """Builds the specialist + exploration + manager agents for the
    electrochemical simulation workflow.
    """

    def __init__(self, reasoning_llm_config=None, tool_llm_config=None,
                 backend: Optional[str] = None):
        backend = (backend or LLM_BACKEND).lower()
        self.backend = backend
        reasoning_default = (VLLM_REASONING_MODEL_DEFAULT if backend == "vllm"
                             else REASONING_MODEL_DEFAULT)
        tool_default = (VLLM_TOOL_MODEL_DEFAULT if backend == "vllm"
                        else TOOL_ONLY_MODEL_DEFAULT)
        self.reasoning_llm_config = (
            reasoning_llm_config or default_llm_config(reasoning_default, backend))
        self.tool_llm_config = (
            tool_llm_config or default_llm_config(tool_default, backend))
        self.llm_config = self.reasoning_llm_config

    @classmethod
    def from_vllm_server(cls, server_config: VLLMServerConfig,
                         temperature: float = 0.1) -> "ElectrochemAgentFactory":
        """Point every agent at one already-running local server."""
        config = VLLMLLMConfig.from_server(server_config, temperature=temperature)
        return cls(reasoning_llm_config=config, tool_llm_config=config,
                   backend="vllm")

    def _base_kwargs(self, system_message: str, reasoning: bool) -> dict:
        config = self.reasoning_llm_config if reasoning else self.tool_llm_config
        return {"system_message": system_message, "llm_config": config.to_autogen_config()}

    def _new_agent(self, name: str, system_message: str, reasoning: bool):
        try:
            from autogen import ConversableAgent
        except ImportError as exc:
            raise ImportError(
                "AG2/AutoGen is required. Install via `pip install ag2`."
            ) from exc
        return ConversableAgent(name=name, **self._base_kwargs(system_message, reasoning))

    def build_system_builder_agent(self):
        agent = self._new_agent("system_builder_agent", SYSTEM_BUILDER_SYSTEM_MESSAGE, reasoning=False)
        self._register_system_builder_tools(agent)
        return agent

    def build_mlip_agent(self):
        agent = self._new_agent("mlip_agent", MLIP_AGENT_SYSTEM_MESSAGE, reasoning=False)
        self._register_mlip_tools(agent)
        return agent

    def build_enhanced_sampling_agent(self):
        agent = self._new_agent("enhanced_sampling_agent", ENHANCED_SAMPLING_AGENT_SYSTEM_MESSAGE, reasoning=False)
        self._register_enhanced_sampling_tools(agent)
        return agent

    def build_results_analyst_agent(self):
        agent = self._new_agent("results_analyst_agent", RESULTS_ANALYST_SYSTEM_MESSAGE, reasoning=True)
        self._register_results_analyst_tools(agent)
        return agent

    def build_validation_agent(self):
        return self._new_agent("validation_agent", VALIDATION_AGENT_SYSTEM_MESSAGE, reasoning=False)

    def build_manager_agent(self, human_input_mode: str = "NEVER"):
        try:
            from autogen import ConversableAgent
        except ImportError as exc:
            raise ImportError("AG2/AutoGen is required (`pip install ag2`).") from exc
        return ConversableAgent(
            name="manager_agent", human_input_mode=human_input_mode,
            **self._base_kwargs(MANAGER_SYSTEM_MESSAGE, reasoning=True),
        )

    def build_abnormality_detection_agent(self):
        agent = self._new_agent("abnormality_detection_agent",
                                 ABNORMALITY_DETECTION_AGENT_SYSTEM_MESSAGE, reasoning=True)
        self._register_abductive_tools(agent)
        return agent

    def build_pattern_finding_agent(self):
        agent = self._new_agent("pattern_finding_agent",
                                 PATTERN_FINDING_AGENT_SYSTEM_MESSAGE, reasoning=True)
        self._register_inductive_tools(agent)
        return agent

    def build_bo_assisted_pattern_finding_agent(self):
        agent = self._new_agent("bo_assisted_pattern_finding_agent",
                                 BO_ASSISTED_PATTERN_FINDING_AGENT_SYSTEM_MESSAGE, reasoning=True)
        self._register_inductive_tools(agent)
        self._register_bo_tools(agent)
        return agent

    def build_user_proxy(self, code_execution_work_dir: str = "agent_runs"):
        try:
            from autogen import UserProxyAgent
            from autogen.coding import LocalCommandLineCodeExecutor
        except ImportError as exc:
            raise ImportError("AG2/AutoGen is required (`pip install ag2`).") from exc
        executor = LocalCommandLineCodeExecutor(work_dir=code_execution_work_dir)
        return UserProxyAgent(name="user_proxy", human_input_mode="NEVER",
                               code_execution_config={"executor": executor})

    def _register_system_builder_tools(self, agent) -> None:
        from systems.cu_interface import build_cu_water_cation_interface, estimate_surface_charge_density
        self._register(agent, build_cu_water_cation_interface, "build_cu_water_cation_interface")
        self._register(agent, estimate_surface_charge_density, "estimate_surface_charge_density")

    def _register_mlip_tools(self, agent) -> None:
        from mlip.esen_oc25 import load_esen_oc25_calculator
        self._register(agent, load_esen_oc25_calculator, "load_esen_oc25_calculator")

    def _register_enhanced_sampling_tools(self, agent) -> None:
        from md.opes_runner import write_plumed_input
        from md.ase_opes_runner import run_md_with_opes
        self._register(agent, write_plumed_input, "write_plumed_input")
        self._register(agent, run_md_with_opes, "run_md_with_opes")

    def _register_results_analyst_tools(self, agent) -> None:
        from analysis.free_energy import extract_barrier_and_reaction_energy, block_free_energy_convergence, is_converged
        from agents.reasoning import build_comparison_report
        self._register(agent, extract_barrier_and_reaction_energy, "extract_barrier_and_reaction_energy")
        self._register(agent, block_free_energy_convergence, "block_free_energy_convergence")
        self._register(agent, is_converged, "is_converged")
        self._register(agent, build_comparison_report, "build_comparison_report")

    def _register_abductive_tools(self, agent) -> None:
        from agents.reasoning import find_local_abnormalities
        self._register(agent, find_local_abnormalities, "find_local_abnormalities")

    def _register_inductive_tools(self, agent) -> None:
        from agents.reasoning import distill_patterns
        self._register(agent, distill_patterns, "distill_patterns")

    def _register_bo_tools(self, agent) -> None:
        from agents.reasoning import propose_bo_candidates
        self._register(agent, propose_bo_candidates, "propose_bo_candidates")

    @staticmethod
    def _register(agent, fn: Callable, name: str) -> None:
        if hasattr(agent, "register_for_llm"):
            agent.register_for_llm(name=name, description=fn.__doc__ or name)(fn)
        if hasattr(agent, "register_for_execution"):
            agent.register_for_execution(name=name)(fn)
