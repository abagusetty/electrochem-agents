# Implementation Status

Module-by-module state of the code. Scientific rationale lives in
[`PROPOSAL.md`](PROPOSAL.md); phasing, gates, and metrics in
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

**Last verified:** 2026-08-20.

**JDFTx runs GPU-native on Aurora PVC via a SYCL port** (project capability,
confirmed 2026-08-20). Upstream JDFTx has no SYCL/oneAPI/HIP backend, so
`ELECTROCHEM_JDFTX_GPU_BIN` must point at the project's build. Consequence for
the code: `BundledLauncher` is the **default** execution mode, because MD and
the grand-canonical audits it triggers now run in one allocation — see
`RESEARCH_PLAN.md` §4.5.

**Execution stack added 2026-08-20** — `hpc/`, `data/`, `acquisition/`,
`cp_dft/{calibration,jdftx_setup,jdftx_driver}`, `mlip/{committee,xpu}`,
`md/cp_md_driver`, `analysis/interface_validity`, `workflows/{campaign,run_campaign}`.
All 22 new modules import cleanly and their pure-logic paths are exercised; none has
been run against JDFTx, a trained MLIP, a scheduler, or an LLM endpoint.

> **The one thing to know before reading further.** Nothing in this repository has
> ever executed against a real simulation stack. Every module is verified by
> pure-numpy unit tests in a sandbox with no ASE, pymatgen, fairchem, PLUMED, JDFTx,
> or CP-MACE installed, and no live LLM endpoint. Zero MD steps have been run, zero
> DFT calculations performed, zero LLM calls made. "Implemented" below means *written
> and logic-tested*, never *executed*.

---

## Status summary

| Area | Module(s) | State |
|---|---|---|
| **HPC / Aurora** | `hpc/{paths,aurora,pbs,launcher}.py` | **new** — node topology, SYCL env, PBS Pro scripting/polling, local\|pbs\|bundle launchers |
| **Campaign data** | `data/{schema,store,harvest,xyz}.py` | **new** — StatePoint/CPDFTLabel/MDResult, append-only JSONL, output parsing, CP-MACE XYZ export |
| **Acquisition** | `acquisition/{policies,sigma_mu,registry}.py` | **new** — B0/B1/B2 baselines, A1 σ_µ policy + calibration, frozen pre-registration |
| **CP-DFT setup/drive** | `cp_dft/{calibration,jdftx_setup,jdftx_driver}.py` | **new** — potential calibration, CANDLE/target-mu inputs, Aurora submission |
| **Committee** | `mlip/committee.py` | **new** — σ_F/σ_µ, implements `get_mu()` |
| **CP-MD** | `md/cp_md_driver.py` | **new** — `run(1)` loop, thresholds, agent hooks |
| **IF-valid** | `analysis/interface_validity.py` | **new** — physical-validity checklist |
| **Campaign** | `workflows/{campaign,run_campaign}.py` | **new** — end-to-end loop + CLI |
| Interface construction | `systems/` | implemented, unit-tested, never run |
| MLIP (constant charge) | `mlip/esen_oc25.py` | implemented; blocked on gated checkpoints (G1) |
| Enhanced sampling | `md/ase_opes_runner.py`, `md/opes_runner.py` | implemented, never run |
| Analysis | `analysis/free_energy.py` | implemented, unit-tested |
| CP-DFT | `cp_dft/jdftx_interface.py` | implemented; **output-attribute gate now closed** |
| CP-MACE driver | `mlip/cp_mace_simulation.py` | **verified compatible with upstream**; one real gap (below) |
| CP-MACE training | `mlip/cp_mace_wrapper.py` | dataset writer + CLI wrapper; nothing trained |
| Agents | `agents/` | implemented; synthetic-data verification only |
| LAMMPS path | `md/lammps_runner.py` | scaffold; not on the critical path |

---

## Resolved this session

Two long-standing `[?]` items closed by reading upstream source directly.

### JDFTx output parsing — CLOSED, no code change needed

`pymatgen.io.jdftx`'s `JDFTXOutfile` public attributes confirmed as `.e`, `.forces`,
`.mu`, `.converged`, `.structure`, `.is_gc`. `cp_dft/jdftx_interface.py`'s
`run_jdftx_single_point()` assumed exactly these names and was correct.
`JDFTXInfile` classmethods confirmed: `from_file`, `from_str`, `from_structure`,
`from_jdftxstructure`, `from_dict`; instance methods `write_file`, `validate_tags`,
`to_pmg_structure`.

Source: https://pymatgen.org/pymatgen.io.jdftx.html

### CP-MACE `NoseHoover` API — CLOSED, module confirmed compatible

The full constructor signature was retrieved:

```python
class NoseHoover(MolecularDynamics):
    def __init__(self, atoms, timestep, constraints, increm, temperature, ttime,
                 Mne, eta_length, targetmu,
                 f0=None, trajectory=None, logfile=None, loginterval=1, **kwargs)
```

`CPMACEIntegratorConfig.to_cp_mace_dict()` emits exactly the eight required keys
(`timestep`, `temperature`, `ttime`, `constraints`, `increm`, `Mne`, `eta_length`,
`targetmu`) and `run_cp_mace_md()` passes them all as keyword arguments, so
positional ordering is irrelevant. **The module is compatible as written.**

Electron-number control law, verbatim from `step()`:

```python
self.Vne[1] = self.Vne[0] + (self.targetmu - self.atoms.get_calculator().get_mu()) \
              * self.delT / 2 / self.Mne
```

**Interface contract this establishes:** any calculator driven by this integrator
**must implement `get_mu()`**. This is the single API requirement a Route-A eSEN
µ-head has to satisfy (`RESEARCH_PLAN.md` §4.2).

Other confirmed semantics: `increm` increments each constraint's target distance per
step (`c[-1] += self.increm`); `eta_length` sizes the Nose–Hoover chain arrays
(`Vlogs`, `Xlogs`, `Glogs`, `Qmass`); unit conversions are `timestep *= units.fs` and
`temperature *= units.kB` for `NoseHoover`/`NoseHooverChain`.

---

## RESOLVED: `run_cp_mace_md()` bypassed upstream's per-step checks

**Fixed by `md/cp_md_driver.py`.** `ConstantPotentialMD.run()` now batches
`integrator.run(1)`, evaluates `force_threshold` and `fermi_threshold` between
batches, records σ_µ per sampled step to `sigma.jsonl`, writes achieved µ to
`mu.log`, and exposes a hook interface (`StepContext`) through which the
sampling-control agent can stop the run, retune OPES, or flag a frame for
grand-canonical audit. `mlip/cp_mace_simulation.py` is unchanged and remains the
low-level integrator loader; the driver is the thing to call.

Detail of what was wrong, retained because the reasoning still applies:

### Original finding

`run_cp_mace_md()` calls `dyn.run(config.steps)` once. Upstream's `simulate.py`
instead calls `integrator.run(1)` inside a Python loop, and evaluates
`force_threshold` and `fermi_threshold` between steps.

`CPMACERunConfig` carries both threshold fields, but with a single `run(steps)` call
**they are never checked** — the run cannot abort on a force blow-up or a Fermi-level
excursion. Two reasons to switch to the loop form:

1. Correctness — the thresholds are safety rails for grand-canonical MD, where the
   electron degree of freedom can run away.
2. Design — the `run(1)` loop is the intervention point the sampling-control agent
   needs to adjust OPES parameters mid-run (`RESEARCH_PLAN.md` §4.4c). Without it,
   there is nowhere for that agent to act.

**Action:** rewrite `run_cp_mace_md()` as a `run(1)` loop with threshold evaluation
and an agent-callable hook. Small change; do it before Phase 2.

---

## Licensing — new hard gate

**CP-MACE ships no LICENSE file.** GitHub API reports `license: null`; no LICENSE in
the repository tree. Upstream MACE is MIT, but this fork grants nothing, so the
default is all-rights-reserved.

| Action | Status |
|---|---|
| Read the code; run it locally for research | fine |
| Live-import `NoseHoover` from a user-supplied local checkout | fine — and this is already what `load_cp_mace_integrator_class()` does |
| Vendor their code into this repo | **not permitted** |
| Redistribute their two 15.68 MB pretrained `.model` files | **not permitted** |
| Release a derivative model or code | **needs written permission** |

The existing live-import design was chosen to avoid reconstructing physics from
partial source. It turns out to be the legally correct architecture as well. **Keep
it. Do not vendor.** Request an explicit license grant from the authors
(`RESEARCH_PLAN.md` G8).

Upstream is also unmaintained — last push 2025-09-16. Pin a commit hash.

---

## Phase 1 — constant-charge MVP

`systems/`, `mlip/esen_oc25.py`, `md/ase_opes_runner.py`, `analysis/free_energy.py`
implement interface construction, eSEN-OC25 inference, PLUMED OPES matching the anchor
paper's Methods §5.1–5.4, and free-energy reweighting with convergence checks.
23/23 pure-numpy unit tests pass.

**fairchem v2 API** (v1's `OCPCalculator` retained only behind `use_legacy_v1=True`):

```python
predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-oc25", device="cuda")
calc = FAIRChemCalculator(predictor)
```

**Checkpoint access is gated.** The OC25 *dataset* is CC-BY-4.0; the *checkpoints* are
not — they sit behind a gated Hugging Face repo under Meta's FAIR Chemistry License
requiring legal name, DOB, organization, and AUP acceptance. Not yet requested; this
blocks Phase 1 entirely (`RESEARCH_PLAN.md` G1).

`ase>=3.23.0` pinned for `ase.calculators.plumed.Plumed`, which also needs `py-plumed`.

---

## Phase 2 — CP-DFT (JDFTx)

`cp_dft/jdftx_interface.py`, pymatgen-first:

- **`run_jdftx_single_point()`** — primary path. Builds a `JDFTXInfile`, runs JDFTx by
  subprocess, parses with `JDFTXOutfile`. No ASE dependency. This is what label
  generation needs.
- **`load_ase_jdftx_calculator()`** — secondary, retained solely for PLUMED coupling,
  which needs a `Calculator` object callable every timestep. pymatgen has no MD driver.
  Explicitly marked not-for-single-points.

Constant-potential control goes through JDFTx's native `target-mu` in both paths — a
property of JDFTx, not of the wrapper. The official JDFTx ASE calculator
(`from JDFTx import JDFTx`) is a thin pass-through with no constant-potential argument
and no structured output parsing.

Tag-assembly logic (`_base_tags`, `build_jdftx_commands`) unit-tested, 4/4, with no
pymatgen/JDFTx/ASE required.

**Still open:** no JDFTx build exists in this workflow. Executable, pseudopotentials,
and CANDLE setup are all ahead (`RESEARCH_PLAN.md` G3) — longest lead time of any gate.
`target-mu` sign convention must be calibrated against a known Cu work function before
any absolute potential is trusted.

---

## Phase 3 — potential-aware MLIP

`mlip/cp_mace_wrapper.py` writes CP-MACE-format datasets and wraps `mace_run_train`.
Nothing has been trained; no CP-DFT labels exist yet (blocked on Phase 2).

Verified upstream training invocation:

```
--model="FermiMACE"          # node augmentation;  FermiMACE_2 = global state variant
--loss="fermi_weighted"
--error_table="Fermi_PerAtomRMSE"
--energy_weight=1.0  --forces_weight=100.0  --potential_weight=10.0
--r_max=5.0  --batch_size=10  --max_num_epochs=300
```

Dataset format: extended-XYZ with `electron=<N>` and `potential=<µ>` on the comment
line (upstream `init.xyz` example: `electron=661.7`, `potential=-3.407347`).

Two upstream simulation modes, both ASE-based:

- **slow growth** — `NoseHoover` only, no PLUMED. CV support is limited to *distance
  between two atoms*, which happens to be exactly the CO-dimerization coordinate.
- **metadynamics** — `Plumed(calc=…, input=…, timestep=1.0*units.fs, atoms=…, kT=…)`
  wraps the *calculator*; `NoseHoover` is still the integrator. PLUMED and the
  integrator are complementary layers, not alternatives.

**Not yet implemented, and now the highest-value addition:** upstream's metadynamics
driver builds an `AverageForceCalculator` over two models that reports force standard
deviation **and chemical-potential variation**. That σ_µ signal is the basis of the
project's headline contribution (`PROPOSAL.md` §3) and needs a committee wrapper on
this side. Nothing in `mlip/` currently constructs a committee.

---

## Phase 4 — agentic layer

Two architectural references: **LAMMPS-Agents** (ANL-NST) for the Manager + specialist
GroupChat pattern, and **A-Lab GPSS** (arXiv:2604.11957, `CederGroupHub/alab_gpss_public`)
for splitting exploration into behaviorally distinct reasoning modes. The GPSS repo is
mostly lab-automation infrastructure plus post-analysis scripts; the agent prompts are
not public, so the implementation here derives from the paper's textual descriptions,
not copied code.

### Exploration agents (`agents/system_messages.py`, `agents/reasoning.py`)

- **`AbnormalityDetectionAgent`** (abductive) — `reasoning.find_local_abnormalities`
  flags records deviating from their *local* chemical neighbors (same facet, nearby
  charge), not just a fixed literature table, since exploration leaves the anchor
  paper's coverage. Emits a specific hypothesis plus exactly one targeted follow-up,
  tagged with a `strategy` field. *Verified:* flagged a deliberately inserted outlier
  in a 7-record synthetic set, plus two records whose local neighbor mean the outlier
  skewed — conservative but defensible.
- **`PatternFindingAgent`** (inductive) — `reasoning.distill_patterns` produces a
  deterministic statistical backbone (per-facet charge slope, facet ranking by mean
  reaction energy, cation-effect magnitude); the agent extrapolates into unexplored
  (facet, charge, cation) space. *Verified:* on synthetic data shaped like the real
  Cu(100)/Cu(310) trend, correctly ranked Cu(310) as more favorable — reproducing the
  anchor paper's actual finding.
- **`BOAssistedPatternFindingAgent`** — activates once `accumulated_records` exceeds
  `bo_transition_threshold` (default 30; GPSS transitioned after 289 of 352 samples,
  scaled down here because one record is a full OPES campaign, not a single synthesis).
  `reasoning.propose_bo_candidates` is a dependency-free novelty + extrapolated-
  favorability heuristic. *Verified:* prioritized a new facet and a favorable-facet
  extrapolation over a low-novelty interpolation point.

`ElectrochemWorkflowManager` tracks `accumulated_records` and swaps the inductive
variant via `should_use_bo_assisted_agent()`; threshold crossing manually tested,
including correct exclusion of non-converged records.

### LLM backend — ALCF Inference Endpoints

Per `docs.alcf.anl.gov/services/inference-endpoints/` and
`argonne-lcf/ATPESC_MachineLearning`:

- Token resolution: `ALCF_ACCESS_TOKEN` env var first, then
  `inference_auth_token.get_access_token()` (auto-refreshing Globus token).
  `ALCF_BASE_URL` overridable.
- `stream: False` hardcoded — the Globus backend does not support streaming.
- Model tiers: reasoning + tool-calling agents (Manager, Results Analyst, all three
  exploration agents) default to `Qwen/Qwen3-235B-A22B`; tool-only agents to
  `meta-llama/Llama-3.3-70B-Instruct`. ATPESC's own notebook uses
  `openai/gpt-oss-120b`, which was *not* adopted — the docs' capability table lists it
  reasoning-only with no confirmed tool-calling, and these agents must invoke real
  tool functions.

Token-resolution precedence verified with 5 offline checks. No network call made.

### Remaining Phase 4 work

- **The σ_µ audit scheduler does not exist yet.** It is the project's headline
  contribution (`PROPOSAL.md` §3) and currently has no module. Highest priority in
  this phase.
- The Validation Agent's checks are prompt-level instructions, not wired to callable
  functions. They should encode the *intricacies of computational electrochemistry*
  guardrails — potential referencing, finite-size effects, implicit-vs-explicit
  solvation, ensemble choice.
- No live LLM call, ever. Requires one interactive
  `python inference_auth_token.py authenticate` Globus login, which nothing in the
  repo can perform for the user.
- Model roster is a 2026-08-12 snapshot; `check_model_availability()` exists but has
  never run.
- No multi-agent conversation has ever run. `GroupChat`/`GroupChatManager`
  construction is verified only by import-shape checks.
- Swap `propose_bo_candidates`' distance heuristic for a real
  `sklearn.gaussian_process.GaussianProcessRegressor` once real data exists to
  calibrate against.
- Consider the Shannon-surprise novelty metric from GPSS's `post_analysis/` as a
  complementary exploration-value signal.

---

## Immediate code actions

Done this session:

1. ~~Committee wrapper (≥2 models, σ_F and σ_µ)~~ → `mlip/committee.py`. Implements
   `get_mu()`, so it satisfies the CP-MACE integrator contract. `load_esen_committee`
   forces `require_mu=False` and exists specifically as the B2 force-only ablation arm.
2. ~~`run(1)` loop with threshold checks and an agent hook~~ → `md/cp_md_driver.py`.
3. ~~σ_µ audit scheduler~~ → `acquisition/sigma_mu.py`, plus the B0/B1/B2 baselines it
   must be measured against and a pre-registration that cannot be edited after results
   exist.
4. Validation Agent guardrails → **partly done**: `analysis/interface_validity.py`
   supplies real, tool-callable geometric checks (IF-valid). The
   electrochemistry-specific guardrails (potential referencing, finite-size effects,
   ensemble choice) are still prompt-level.
5. ~~Pin a CP-MACE commit~~ → `SoftwareStack.cp_mace_commit`; local-checkout
   requirement documented in the README and enforced by `load_cp_mace_integrator_class`.

Still open:

6. Wire `md.cp_md_driver.run_constant_potential_md` into `Campaign.md_runner`. It is
   deliberately left as `None` until trained committee members exist — a stub that
   silently produced MDResults would corrupt the acquisition comparison.
7. Encode the remaining electrochemistry guardrails as callable checks.
8. Replace `propose_bo_candidates`' distance heuristic with a real GP once calibrated
   data exists.
9. Run `python -m workflows.run_campaign doctor` on an Aurora compute node to settle
   gate G4 (fairchem/MACE on XPU). Now the only Intel-GPU unknown, since JDFTx is
   SYCL-ported. No longer blocking — it decides whether the acquisition loop stays
   co-located in one allocation or splits across machines.
10. Strong-scaling run for GC-DFT on PVC (200/400/800-atom cells) to replace the
    guessed `atoms_per_rank` and the x8 GPU multiplier in
    `hpc.aurora.nodes_for_jdftx`. These are the least-grounded numbers in the stack.

## Verification status of the new code

Verified: `BundledLauncher.run_all` throttling — 10 tasks at capacity 4 over an
8-host nodefile ran in 3 waves with disjoint host groups, input order preserved, no
oversubscription (previously the inherited `run_all` launched all 10 at once and
wrapped the host cursor, silently double-booking PVC tiles). Also verified: all 22
modules import with numpy only; PBS script generation; Aurora mpiexec
construction; SYCL env; StatePoint ids and the ensemble guard; potential-calibration
round-trip against the Cu(100) work function; JDFTx tag assembly; the uncalibrated-sweep
warning; all four acquisition policies; `SigmaMuPolicy`'s refusal to act on an
uncalibrated threshold; threshold calibration correctly separating correlated from
uncorrelated σ/error data; pre-registration fingerprinting; and all three
`compare_policies` verdicts (A1 wins / no difference / inconclusive).

Not verified: anything requiring JDFTx, a trained MLIP, PLUMED, ASE, pymatgen, a PBS
scheduler, or an LLM endpoint.
