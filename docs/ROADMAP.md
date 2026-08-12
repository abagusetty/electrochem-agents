# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

Status: core geometry, I/O, PLUMED input generation, and free-energy
analysis are implemented and unit-tested (23/23 passing on pure numpy, no
ase/pymatgen/fairchem/JDFTx required for the tested subset).

- systems/water_geometry.py, systems/packing.py, systems/io_utils.py,
  systems/cu_interface.py -- interface construction (OC25 recipe).
- mlip/esen_oc25.py -- fairchem-core >=2.x unified API
  (pretrained_mlip.get_predict_unit + FAIRChemCalculator). OC25 checkpoints
  are gated under Meta's FAIR Chemistry License (huggingface.co/facebook/OC25).
- mlip/cp_mace_wrapper.py -- CP-MACE dataset writer / mace_run_train wrapper.
- md/opes_runner.py, md/ase_opes_runner.py -- PLUMED OPES matching Methods 5.4.
- md/lammps_runner.py -- LAMMPS ML-IAP path (secondary).
- analysis/free_energy.py -- reweighting, convergence checks, orientation analysis.

## Phase 2 -- CP-DFT audit module -- ON HOLD (paused 2026-08-12)

cp_dft/jdftx_interface.py implements a pymatgen.io.jdftx-first design
(JDFTXInfile/JDFTXOutfile as primary path, official ASE JDFTx calculator
kept only for PLUMED-coupled MD). Verified against pymatgen-core source
(github.com/materialsproject/pymatgen-core): JDFTXInfile is a `dict`
subclass with a real `from_structure()` classmethod; `target-mu` is a
confirmed real tag (nested under the `fluid` category); `is_gc` in
JDFTXOutfileSlice is set via `key_exists("target-mu", text)`. Output-side
attribute names (`.e`, `.forces`, `.mu`, `.converged` on JDFTXOutfile) were
NOT yet confirmed against source before this work was paused -- verify
before relying on `run_jdftx_single_point()`'s return dict.

## Phase 3 -- CP-MACE for Cu interfaces

- Use mlip/cp_mace_wrapper.py to build CP-MACE-format datasets from OC25 +
  CP-DFT labels; train FermiMACE; validate against CP-DFT/eSEN-OC25.

## Phase 4 -- Agentic orchestration -- IMPLEMENTED (2026-08-12)

Built on the AG2/AutoGen framework (`import autogen`), matching the
architecture of github.com/ANL-NST/LAMMPS-Agents: a Manager agent
coordinating specialist agents (System Builder, MLIP, Enhanced-Sampling,
Results Analyst, Validation) via GroupChat, each wired to the real
deterministic tool functions in systems/, mlip/, md/, analysis/ through
function-calling.

- agents/system_messages.py -- domain-specific reasoning rules embedded in
  each agent's prompt, grounded in concrete values from arXiv:2509.17862
  (convergence timescales, reference barriers/reaction energies per
  facet/condition, expected magnitude of cation vs. charge effects,
  when an MLIP should be audited against CP-DFT before being trusted).
- agents/reasoning.py -- deterministic (non-LLM) comparison of a
  completed run's results against the anchor paper's reference values,
  producing structured `flags` that ground the Results Analyst Agent's
  judgment in concrete numbers rather than asking it to recall literature
  from memory. Verified with 5 manual test cases (converged-and-matching,
  large deviation, non-convergence, novel/out-of-reference facet, large
  cation effect) -- all produced correct, distinct flags.
- agents/agent_factory.py -- builds each ConversableAgent and registers
  its tool functions (e.g. Enhanced-Sampling Agent gets
  md.ase_opes_runner.run_md_with_opes; Results Analyst Agent gets
  analysis.free_energy.is_converged, block_free_energy_convergence, etc.).
- agents/manager.py -- GroupChat/GroupChatManager orchestration; entry
  point is workflows/run_workflow.py, which loads a workflow YAML and
  starts the multi-agent conversation.

Note: this uses the AG2 fork of AutoGen (`pip install ag2`, importable as
`autogen`), the same package LAMMPS-Agents itself depends on -- NOT
Microsoft's newer autogen-agentchat (>=0.4), which has an incompatible
async API. Do not install both in the same environment.

Remaining Phase 4 work:
- End-to-end run against a live LLM + real GPU-backed tool calls (agents
  and reasoning logic are implemented and unit-verified in isolation, but
  not yet run as a live multi-agent conversation).
- Wire the Validation Agent's three checks (system validity, MLIP
  readiness, CP-DFT prerequisites) to concrete function calls rather than
  leaving them as prompt-level instructions.
