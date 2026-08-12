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
Verified against pymatgen-core source: JDFTXInfile is a `dict` subclass
with a real `from_structure()` classmethod; `target-mu` is a confirmed
real tag; `is_gc` is set via `key_exists("target-mu", text)`. Output-side
attribute names on JDFTXOutfile were NOT yet confirmed before this work
was paused -- verify before relying on `run_jdftx_single_point()`.

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
(docs.alcf.anl.gov/services/inference-endpoints/), not a commercial API.

- agents/llm_backend.py -- ALCFLLMConfig: builds an AG2 `llm_config`
  pointed at ALCF's OpenAI-compatible vLLM endpoint
  (https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1),
  authenticated via a Globus access token from the `inference_auth_token.py`
  helper (NOT a static API key -- tokens are valid 48h with auto-refresh,
  weekly re-auth required). `stream: False` is hardcoded per ALCF's
  documented limitation (the Globus backend does not support streaming).
  `check_model_availability()` queries ALCF's `/jobs` endpoint so a
  workflow can check Live/Starting/Queued/Offline status before
  committing to a model (cold starts can take 10-15 minutes).
- Model assignment (agents/agent_factory.py) follows ALCF's documented
  Tool-Calling (T) / Reasoning (R) capability flags per role:
    - Manager, Results Analyst (need both T and R -- reasoning about
      convergence/literature comparisons, not just dispatching calls):
      default `Qwen/Qwen3-235B-A22B` (also `Qwen/QwQ-32B` available).
    - System Builder, MLIP, Enhanced-Sampling, Validation Agents (mostly
      deterministic tool dispatch): default `meta-llama/Llama-3.3-70B-Instruct`.
  This model list is a snapshot from the ALCF docs (2026-08-12) and may
  change; call `check_model_availability()` before a production run.
- agents/system_messages.py -- domain-specific reasoning rules grounded
  in arXiv:2509.17862 reference values.
- agents/reasoning.py -- deterministic (non-LLM) comparison of results
  against anchor-paper reference values; verified with 5 manual cases.
- agents/agent_factory.py -- builds each ConversableAgent with the
  appropriate ALCF model tier and registers its tool functions.
- agents/manager.py -- GroupChat/GroupChatManager orchestration; entry
  point is workflows/run_workflow.py.

Remaining Phase 4 work:
- End-to-end run against the live ALCF endpoint (agents and reasoning
  logic are implemented and unit-verified in isolation; llm_backend.py's
  token-fetch/config logic verified without network calls, but no live
  ALCF request has been made yet).
- Confirm current model availability/capability flags via
  `check_model_availability()` before a production run, since ALCF's
  model roster and hot/cold status changes over time.
- Wire the Validation Agent's three checks to concrete function calls.
