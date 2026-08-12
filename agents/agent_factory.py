"""
Agent factory for electrochemical interface simulations, mirroring the
architecture of github.com/ANL-NST/LAMMPS-Agents: a set of specialist
ConversableAgents (System Builder, MLIP, Enhanced-Sampling, Results
Analyst, Validation) coordinated by a Manager agent, each wired to Python
tool functions via AutoGen/AG2 function-calling.

Uses the AG2 fork of AutoGen (`pip install ag2`, importable as
`autogen`), which is what LAMMPS-Agents itself uses. LLM access is via
ALCF Inference Endpoints (agents.llm_backend.ALCFLLMConfig) rather than a
commercial API, since that is the compute resource actually available for
this project (docs.alcf.anl.gov/services/inference-endpoints/).

Model assignment follows ALCF's documented tool-calling (T) / reasoning
(R) capability flags per agent role:
  - Manager, Results Analyst: need both T (to call tools) and R (to
    reason about convergence/literature comparisons) -> default to
    Qwen/Qwen3-235B-A22B or Qwen/QwQ-32B.
  - System Builder, MLIP Agent, Enhanced-Sampling Agent, Validation Agent:
    mostly deterministic tool dispatch -> a lighter T-only model
    (meta-llama/Llama-3.3-70B-Instruct by default) is sufficient and
    cheaper on shared ALCF resources.

The reasoning content lives in agents/system_messages.py, not here -- this
module is deliberately thin glue, consistent with keeping the agentic
harness itself a small part of the project (the scientific core is the
constant-potential physics in systems/, mlip/, md/, cp_dft/, analysis/).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agents.llm_backend import ALCFLLMConfig
from agents.system_messages import (
    MANAGER_SYSTEM_MESSAGE,
    SYSTEM_BUILDER_SYSTEM_MESSAGE,
    MLIP_AGENT_SYSTEM_MESSAGE,
    ENHANCED_SAMPLING_AGENT_SYSTEM_MESSAGE,
    RESULTS_ANALYST_SYSTEM_MESSAGE,
    VALIDATION_AGENT_SYSTEM_MESSAGE,
)

# Backward-compatible alias: earlier revisions of this module exposed a
# generic `LLMConfig` name; keep it importable as an alias for
# ALCFLLMConfig so existing call sites (agents/manager.py) don't break.
LLMConfig = ALCFLLMConfig

REASONING_MODEL_DEFAULT = "Qwen/Qwen3-235B-A22B"
TOOL_ONLY_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"


class ElectrochemAgentFactory:
    """Builds the specialist + manager agents for the electrochemical
    simulation workflow, wiring each specialist agent to its
    corresponding tool functions via AutoGen's function-calling.

    Accepts two LLM configs: `reasoning_llm_config` for agents that need
    to weigh evidence against literature thresholds (Manager, Results
    Analyst), and `tool_llm_config` for agents that mostly dispatch
    deterministic tool calls (System Builder, MLIP, Enhanced-Sampling,
    Validation). Both default to ALCF-backed configs if not supplied.
    """

    def __init__(self, reasoning_llm_config: Optional[ALCFLLMConfig] = None,
                 tool_llm_config: Optional[ALCFLLMConfig] = None):
        self.reasoning_llm_config = reasoning_llm_config or ALCFLLMConfig(
            model=REASONING_MODEL_DEFAULT,
        )
        self.tool_llm_config = tool_llm_config or ALCFLLMConfig(
            model=TOOL_ONLY_MODEL_DEFAULT,
        )
        # Kept for backward compatibility with code that reads
        # `factory.llm_config` directly (e.g. agents/manager.py's
        # GroupChatManager construction uses the reasoning config, since
        # the group-level manager needs to weigh handoffs across agents).
        self.llm_config = self.reasoning_llm_config

    def _base_kwargs(self, system_message: str, reasoning: bool) -> dict:
        config = self.reasoning_llm_config if reasoning else self.tool_llm_config
        return {
            "system_message": system_message,
            "llm_config": config.to_autogen_config(),
        }

    def build_system_builder_agent(self):
        try:
            from autogen import ConversableAgent
        except ImportError as exc:
            raise ImportError(
                "AG2/AutoGen is required. Install via `pip install ag2` "
                "(the fork LAMMPS-Agents itself depends on, importable "
                "as `autogen`)."
            ) from exc

        agent = ConversableAgent(
            name="system_builder_agent",
            **self._base_kwargs(SYSTEM_BUILDER_SYSTEM_MESSAGE, reasoning=False),
        )
        self._register_system_builder_tools(agent)
        return agent

    def build_mlip_agent(self):
        from autogen import ConversableAgent

        agent = ConversableAgent(
            name="mlip_agent",
            **self._base_kwargs(MLIP_AGENT_SYSTEM_MESSAGE, reasoning=False),
        )
        self._register_mlip_tools(agent)
        return agent

    def build_enhanced_sampling_agent(self):
        from autogen import ConversableAgent

        agent = ConversableAgent(
            name="enhanced_sampling_agent",
            **self._base_kwargs(ENHANCED_SAMPLING_AGENT_SYSTEM_MESSAGE, reasoning=False),
        )
        self._register_enhanced_sampling_tools(agent)
        return agent

    def build_results_analyst_agent(self):
        from autogen import ConversableAgent

        agent = ConversableAgent(
            name="results_analyst_agent",
            **self._base_kwargs(RESULTS_ANALYST_SYSTEM_MESSAGE, reasoning=True),
        )
        self._register_results_analyst_tools(agent)
        return agent

    def build_validation_agent(self):
        from autogen import ConversableAgent

        agent = ConversableAgent(
            name="validation_agent",
            **self._base_kwargs(VALIDATION_AGENT_SYSTEM_MESSAGE, reasoning=False),
        )
        return agent

    def build_manager_agent(self, human_input_mode: str = "NEVER"):
        from autogen import ConversableAgent

        return ConversableAgent(
            name="manager_agent",
            human_input_mode=human_input_mode,
            **self._base_kwargs(MANAGER_SYSTEM_MESSAGE, reasoning=True),
        )

    def build_user_proxy(self, code_execution_work_dir: str = "agent_runs"):
        try:
            from autogen import UserProxyAgent
            from autogen.coding import LocalCommandLineCodeExecutor
        except ImportError as exc:
            raise ImportError("AG2/AutoGen is required (`pip install ag2`).") from exc

        executor = LocalCommandLineCodeExecutor(work_dir=code_execution_work_dir)
        return UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config={"executor": executor},
        )

    # -- tool registration -------------------------------------------------
    # Each specialist agent is wired to the real (non-LLM) implementation
    # functions already implemented in systems/, mlip/, md/, analysis/.
    # AutoGen's function-calling means the LLM decides WHEN to call these
    # given its system message's reasoning, but the functions themselves
    # are the same deterministic code used outside the agentic harness.

    def _register_system_builder_tools(self, agent) -> None:
        from systems.cu_interface import (
            InterfaceSpec,
            build_cu_water_cation_interface,
            estimate_surface_charge_density,
        )
        self._register(agent, build_cu_water_cation_interface, "build_cu_water_cation_interface")
        self._register(agent, estimate_surface_charge_density, "estimate_surface_charge_density")

    def _register_mlip_tools(self, agent) -> None:
        from mlip.esen_oc25 import ESENOC25Config, load_esen_oc25_calculator
        self._register(agent, load_esen_oc25_calculator, "load_esen_oc25_calculator")

    def _register_enhanced_sampling_tools(self, agent) -> None:
        from md.opes_runner import OPESConfig, write_plumed_input
        from md.ase_opes_runner import MDRunConfig, run_md_with_opes
        self._register(agent, write_plumed_input, "write_plumed_input")
        self._register(agent, run_md_with_opes, "run_md_with_opes")

    def _register_results_analyst_tools(self, agent) -> None:
        from analysis.free_energy import (
            extract_barrier_and_reaction_energy,
            block_free_energy_convergence,
            is_converged,
            water_orientation_distribution,
        )
        from agents.reasoning import build_comparison_report
        self._register(agent, extract_barrier_and_reaction_energy, "extract_barrier_and_reaction_energy")
        self._register(agent, block_free_energy_convergence, "block_free_energy_convergence")
        self._register(agent, is_converged, "is_converged")
        self._register(agent, water_orientation_distribution, "water_orientation_distribution")
        self._register(agent, build_comparison_report, "build_comparison_report")

    @staticmethod
    def _register(agent, fn: Callable, name: str) -> None:
        """Register a plain Python function as a callable tool on an
        AutoGen ConversableAgent. Uses `register_for_llm`/
        `register_for_execution` if available (current AG2 API);
        falls back to attribute assignment for older/newer variants.
        """
        if hasattr(agent, "register_for_llm"):
            agent.register_for_llm(name=name, description=fn.__doc__ or name)(fn)
        if hasattr(agent, "register_for_execution"):
            agent.register_for_execution(name=name)(fn)
