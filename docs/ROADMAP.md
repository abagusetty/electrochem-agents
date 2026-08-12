# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

- systems/, mlip/, md/ -- interface construction, eSEN-OC25 (fairchem-core
  >=2.x), PLUMED OPES matching arXiv:2509.17862 Methods 5.1-5.4.
- analysis/free_energy.py -- reweighting, convergence checks.
- 23/23 unit tests passing on pure numpy.

## Phase 2 -- CP-DFT audit module -- ON HOLD (paused 2026-08-12)

cp_dft/jdftx_interface.py: pymatgen.io.jdftx-first design, verified
against pymatgen-core source; output-side attribute names NOT yet
confirmed before pausing.

## Phase 3 -- CP-MACE for Cu interfaces

- mlip/cp_mace_wrapper.py: CP-MACE dataset writer / mace_run_train wrapper.

## Phase 4 -- Agentic orchestration -- IMPLEMENTED, UPDATED 2026-08-12 (self-learning exploration)

Combines two architectural references:

1. **github.com/ANL-NST/LAMMPS-Agents**: Manager + specialist agents
   (System Builder, MLIP, Enhanced-Sampling, Results Analyst, Validation)
   via AG2/AutoGen GroupChat, each wired to real tool functions.
2. **Fei, Rendy, Yang et al., "Agentic LLM Reasoning in a Self-Driving
   Laboratory for Air-Sensitive Lithium Halide Spinel Conductors"
   (arXiv:2604.11957)**: splitting EXPLORATION into two complementary,
   behaviorally distinct reasoning modes rather than one monolithic
   decision-maker. Their code repo (github.com/CederGroupHub/alab_gpss_public)
   is mostly lab-automation infrastructure (backend/system/ui/daemon) plus
   post-analysis scripts (claim extraction, causal-effect extraction,
   Shannon-surprise novelty metric); the agent prompts themselves are not
   public, so the implementation here is built from the paper's detailed
   textual description of each agent's role, not copied code.

### New exploration agents (agents/system_messages.py, agents/reasoning.py)

- **AbnormalityDetectionAgent (abductive)**: calls
  `reasoning.find_local_abnormalities` to flag records that deviate from
  their LOCAL chemical neighbors (same facet, nearby charge/potential) --
  not just fixed literature values, since exploration moves beyond what
  arXiv:2509.17862 studied. Generates a specific hypothesis for each
  flagged deviation and proposes ONE targeted follow-up (longer sampling,
  a CP-DFT audit, or an intermediate state point), tagged with a
  `strategy` field for traceability (per the anchor paper's finding that
  untraceable monolithic reasoning obscures whether success reflects real
  insight). Verified: correctly flagged a deliberately inserted outlier
  record in a 7-record synthetic test, plus correctly identified two
  additional records whose local neighbor mean was skewed by that
  outlier -- a defensible, if conservative, behavior for a first pass.
- **PatternFindingAgent (inductive)**: calls `reasoning.distill_patterns`
  to get a deterministic statistical backbone (per-facet charge-slope,
  facet ranking by mean reaction energy, cation-effect magnitude), then
  proposes new state points that extrapolate distilled trends into
  unexplored (facet, charge, cation) space. Verified: on synthetic data
  matching the anchor paper's actual Cu(100) vs Cu(310) trend, correctly
  ranked Cu(310) as more favorable than Cu(100) -- reproducing the real
  finding, not a coincidence of the synthetic setup.
- **BOAssistedPatternFindingAgent**: activated once
  `ElectrochemWorkflowManager.accumulated_records` exceeds
  `bo_transition_threshold` (default 30; the anchor paper's campaign
  transitioned after 289 of 352 samples -- ours is scaled down since each
  record here is a full OPES campaign, not a single synthesis run).
  Calls `reasoning.propose_bo_candidates`, a dependency-free novelty +
  extrapolated-favorability scoring heuristic (swap in a real
  `sklearn.gaussian_process.GaussianProcessRegressor` for calibrated
  uncertainty if available) that ranks not-yet-run (facet, charge, cation)
  candidates. Verified: correctly prioritized a brand-new facet and a
  favorable-facet extrapolation over a low-novelty interpolation point.

### Manager updates (agents/manager.py)

`ElectrochemWorkflowManager` now tracks `accumulated_records` (a list of
`agents.reasoning.SimulationRecord`) and swaps the inductive agent
variant automatically via `should_use_bo_assisted_agent()`, verified with
a manual threshold-crossing test (including correct exclusion of a
non-converged record from the count).

### LLM backend: ALCF Inference Endpoints

Per docs.alcf.anl.gov/services/inference-endpoints/ and Argonne's own
reference implementation (github.com/argonne-lcf/ATPESC_MachineLearning,
13_agentic_workflows_for_science and 11_Agentic_tools_part1):
- Token resolution: `ALCF_ACCESS_TOKEN` env var first, then
  `inference_auth_token.get_access_token()` (auto-refreshing Globus token).
- `stream: False` hardcoded (Globus backend does not support streaming).
- Model tiers: reasoning+tool-calling agents (Manager, Results Analyst,
  and all three exploration agents) default to `Qwen/Qwen3-235B-A22B`;
  tool-only agents default to `meta-llama/Llama-3.3-70B-Instruct`.

Remaining Phase 4 work:
- End-to-end run against the live ALCF endpoint with real accumulated
  data (all logic verified in isolation with synthetic records; no live
  network call or real simulation campaign has been run yet).
- Consider swapping `propose_bo_candidates`' distance-based heuristic for
  a real Gaussian Process once enough real data exists to fit one
  meaningfully.
- Add the "Shannon surprise" style novelty metric from
  CederGroupHub/alab_gpss_public's post_analysis/ as a possible
  complementary exploration-value signal (not yet implemented).
