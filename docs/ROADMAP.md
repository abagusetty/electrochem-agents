# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

- systems/water_geometry.py, systems/packing.py, systems/io_utils.py,
  systems/cu_interface.py -- interface construction (OC25 recipe).
- mlip/esen_oc25.py -- fairchem-core >=2.x unified API. OC25 checkpoints
  are gated under Meta's FAIR Chemistry License (huggingface.co/facebook/OC25).
- mlip/cp_mace_wrapper.py -- CP-MACE dataset writer / mace_run_train wrapper.
- md/opes_runner.py, md/ase_opes_runner.py -- PLUMED OPES matching Methods 5.4.
- analysis/free_energy.py -- reweighting, convergence checks, orientation analysis.
- 23/23 unit tests passing on pure numpy (no ase/pymatgen/fairchem/JDFTx required).

## Phase 2 -- CP-DFT audit module -- ON HOLD (paused 2026-08-12)

cp_dft/jdftx_interface.py implements a pymatgen.io.jdftx-first design.
Verified against pymatgen-core source. Output-side attribute names on
JDFTXOutfile were NOT yet confirmed before this work was paused.

## Phase 3 -- CP-MACE for Cu interfaces

- Use mlip/cp_mace_wrapper.py to build CP-MACE-format datasets from OC25 +
  CP-DFT labels; train FermiMACE; validate against CP-DFT/eSEN-OC25.

## Phase 4 -- Agentic orchestration -- IMPLEMENTED, UPDATED 2026-08-12 (ALCF backend)

Built on AG2/AutoGen, matching github.com/ANL-NST/LAMMPS-Agents'
architecture: a Manager agent coordinating specialist agents (System
Builder, MLIP, Enhanced-Sampling, Results Analyst, Validation) via
GroupChat, each wired to real tool functions in systems/, mlip/, md/,
analysis/.

LLM backend: **ALCF Inference Endpoints**
(docs.alcf.anl.gov/services/inference-endpoints/), verified against
Argonne's own reference implementation
(github.com/argonne-lcf/ATPESC_MachineLearning/tree/master/13_agentic_workflows_for_science
and .../11_Agentic_tools_part1, and .../14_agentic_tools_part2/ATPESC-Agents-Tutorial.ipynb).

- agents/llm_backend.py -- ALCFLLMConfig + get_alcf_access_token(), using
  the EXACT token-resolution precedence confirmed from ATPESC's own
  alcf_llm.py source:
    1. `ALCF_ACCESS_TOKEN` env var (manual override, e.g. after
       `source scripts/get_alcf_token.sh`).
    2. `inference_auth_token.get_access_token()` (auto-refreshing cached
       Globus token; interactive login only on first use).
  `ALCF_BASE_URL` env var is also overridable, matching ATPESC's
  .env.example convention. `stream: False` is hardcoded (Globus backend
  does not support streaming, confirmed). Verified with 5 manual checks
  (env-var precedence, base_url override precedence, default fallback,
  explicit-arg precedence over env, full to_autogen_config() shape) --
  all passed without needing network access or a real token.
  `check_model_availability()` queries ALCF's `/jobs` endpoint for
  Live/Starting/Queued/Offline status before committing to a model.
- Model assignment (agents/agent_factory.py) follows ALCF's documented
  Tool-Calling (T) / Reasoning (R) capability flags per role:
    - Manager, Results Analyst (need both T and R): default
      `Qwen/Qwen3-235B-A22B` (also `Qwen/QwQ-32B` available).
    - System Builder, MLIP, Enhanced-Sampling, Validation Agents (mostly
      deterministic tool dispatch): default `meta-llama/Llama-3.3-70B-Instruct`.
  Note: ATPESC's own tutorial notebook uses `openai/gpt-oss-120b` as a
  general example, but that model is Reasoning-only (no confirmed
  Tool-Calling) on this endpoint as of this snapshot -- do not use it for
  AG2 function-calling agents; it would be fine for a pure-text-reasoning
  role with no tool registration.
- agents/system_messages.py -- domain-specific reasoning rules grounded
  in arXiv:2509.17862 reference values.
- agents/reasoning.py -- deterministic (non-LLM) comparison of results
  against anchor-paper reference values; verified with 5 manual cases.
- agents/agent_factory.py -- builds each ConversableAgent with the
  appropriate ALCF model tier and registers its tool functions.
- agents/manager.py -- GroupChat/GroupChatManager orchestration; entry
  point is workflows/run_workflow.py.

Remaining Phase 4 work:
- End-to-end run against the live ALCF endpoint with a real Globus token
  (all logic verified in isolation; no live network call has been made).
- Confirm current model availability/capability flags via
  `check_model_availability()` before a production run.
- Wire the Validation Agent's three checks to concrete function calls.
