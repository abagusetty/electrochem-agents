# Research Plan — Agentic Constant-Potential Electrocatalysis

**Version:** 2026-08-20. Supersedes the phase notes in `docs/ROADMAP.md` for planning
purposes. Companion to `docs/PROPOSAL.md` (the *why*); this document is the *what and when*.

**Provenance tags:** `[V]` verified against a live source in §9 · `[C]` verified by
reading source code · `[P]` project capability asserted by the team ·
`[A]` assumption/design choice · `[?]` unverified, blocks manuscript.

---

## 0. The one-paragraph version

The anchor paper (arXiv:2509.17862) runs 7 ns of explicit-solvent enhanced sampling on
>800-atom Cu/water cells at **constant charge**, and explicitly asks for grand-canonical
coupling or direct work-function prediction as future work. `[V-1]` CP-MACE
(JCTC 2025) supplies a working constant-potential MLFF and a grand-canonical
Nose–Hoover integrator, but was demonstrated on Ni–N–C and Au, not Cu, and its
enhanced-sampling driver already carries a two-model committee that reports **variance
in the Fermi level itself**. `[V-2]` `[C]` BEAST DB supplies 20,000+ grand-canonical
JDFTx calculations — but in *implicit* solvent. `[V-3]` The unoccupied cell in that
grid is **explicit solvent × constant potential**, and the reason it is unoccupied is
cost. This plan fills it by making a committee-disagreement signal — specifically
disagreement in µ, which only exists in a constant-potential model — the trigger for
spending grand-canonical JDFTx calls, under agent control.

---

## 1. What changed with the new information

Five facts verified this session materially reshape the plan.

### 1.1 The anchor paper is not the OC25 dataset paper, and Ulissi is an author `[V-1]`

**Title:** *Insights into CO dimerization at electrified Cu interfaces from large-scale
machine learning simulations*. v1 2025-09-22, v2 2026-06-08.

**Authors:** Sushree Jagriti Sahoo, Mikael Maraschin, Joel B. Varley, Daniel S. Levine,
**Zachary Ulissi**, C. Lawrence Zitnick, Wayu Takemura, Joseph A. Gauthier,
Nitish Govindarajan, Muhammed Shuaibi.

This is a **Meta FAIR Chemistry + LLNL + Texas Tech** collaboration. Consequences:

- The "how do we interest Ulissi" question is not a cold-outreach problem. He is an
  author on the paper whose Discussion section this project proposes to answer.
  The pitch is *continuation*, not *introduction* (§7).
- Govindarajan (LLNL) is also the first author of *The intricacies of computational
  electrochemistry* — i.e. a co-author of the anchor paper is simultaneously the person
  writing the field's methodological guardrails. Those guardrails should be encoded as
  the validation agent's automated checks (§4.4), which is both good practice and a
  visible signal of engagement with that group's concerns.
- LLNL involvement means DOE-lab collaboration on this exact system already exists.
  A DOE-lab entrant is joining an established pattern, not proposing a novel one.
- Precise numbers: cells >800 atoms, enhanced sampling up to 7 ns, "the largest
  explicit-solvent CO dimerization study to date."

### 1.2 CP-MACE has a two-model committee that reports Fermi-level variance `[C]`

Read from `simulation/metadynamics/simulate.py`:

```python
calculator1 = MACECalculator(model_paths=['AuORR_run-1130.model'], device='cuda')
calculator2 = MACECalculator(model_paths=['AuORR_run-2008.model'], device='cuda')
average_calculator = AverageForceCalculator([calculator1, calculator2])
```

`AverageForceCalculator` averages forces/energies across models **and computes force
standard deviation and chemical-potential variation**. `[C]`

**This is the most important finding of the session.** A constant-potential MLIP
committee produces an uncertainty signal that a constant-charge committee structurally
cannot: **σ_µ, the disagreement in predicted electrode potential.** That signal is:

- physically meaningful — it is disagreement about the electronic boundary condition,
  not about a force component;
- already implemented upstream, so not something this project must invent;
- exactly the right trigger for an agent deciding when to spend a grand-canonical
  DFT call.

The audit policy in §4.4 is therefore built on σ_µ, not on generic force uncertainty.
This converts the agent layer from "a plausible orchestration idea" into a method with
a specific, defensible, constant-potential-native decision variable.

### 1.3 The `NoseHoover` signature and µ-control law are now known exactly `[C]`

```python
class NoseHoover(MolecularDynamics):
    def __init__(self, atoms, timestep, constraints, increm, temperature, ttime,
                 Mne, eta_length, targetmu,
                 f0=None, trajectory=None, logfile=None, loginterval=1, **kwargs)
```

Electron-number control law, verbatim from `step()`:

```python
self.Vne[1] = self.Vne[0] + (self.targetmu - self.atoms.get_calculator().get_mu()) \
              * self.delT / 2 / self.Mne
```

Consequences that are now facts rather than inferences:

| Item | Resolution |
|---|---|
| Calculator API contract | any calculator handed to this integrator **must implement `get_mu()`**. This is the single interface a Route-A eSEN head must satisfy. `[C]` |
| `increm` semantics | slow-growth: increments each constraint's target distance per step (`c[-1] += self.increm`). `[C]` |
| `eta_length` | Nose–Hoover *chain* length; sizes `Vlogs`/`Xlogs`/`Glogs`/`Qmass`. `[C]` |
| Run pattern | upstream calls `integrator.run(1)` inside a Python loop, not `run(steps)` — this is what makes per-step agent intervention possible. `[C]` |
| Unit conversions | `timestep *= units.fs`; `temperature *= units.kB` for `NoseHoover`/`NoseHooverChain`. `[C]` |

The previously-flagged `[?]` on `run_cp_mace_md()`'s inferred signature is **closed**.
The `run(1)`-in-a-loop pattern is a gift: the sampling-control agent (§4.4c) can adjust
bias parameters between single steps without patching the integrator.

### 1.4 CP-MACE has no LICENSE file `[V-2]`

GitHub API `license: null`; no LICENSE in the tree; `pushed_at` 2025-09-16 (no code
changes in ~11 months); 48 stars, 7 forks; two 15.68 MB pretrained `.model` files under
`simulation/slow_growth/`.

Upstream MACE is MIT, but **this fork ships no license grant**, so the default is
all-rights-reserved. This is a hard legal gate (§6, G8), not a footnote:

- fine → reading it, running it locally for research;
- **not fine** → vendoring their code into a public repo, redistributing their
  `.model` files, or releasing a derivative without written permission.

The existing design decision to *live-import* their `NoseHoover` from a user-supplied
local checkout rather than vendor a copy turns out to be the legally correct
architecture as well as the technically correct one. Keep it. Email the authors for an
explicit license grant — a one-line reply resolves it, and asking also opens the
Liu-group channel.

### 1.5 BEAST DB defines the gap precisely `[V-3]`

*BEAST DB: Grand-Canonical Database of Electrocatalyst Properties* (arXiv:2405.20239) —
Tezak, Clary, …, **Sundararaman** (JDFTx lead author), Musgrave, Vigil-Fowler.
20,000+ surface calculations, grand-canonical DFT, **implicit solvent**, consistent
parameters, both self-consistent fixed-potential and constant-charge, open-source with
query interfaces.

This lets the contribution be stated as a 2×2 with one empty cell:

|  | **Constant charge** | **Constant potential** |
|---|---|---|
| **Implicit solvent** | routine | **BEAST DB** — 20k calcs `[V-3]` |
| **Explicit solvent** | **OC25** — 7.8M calcs `[V-1]` | ← *empty; this project* |

**Important qualification, verified `[V-7]`.** JDFTx hard-gates `target-mu` on
an electrolyte: the parser *requires* `elec-smearing`, `fluid-cation`,
`fluid-anion` and `fluid-solvent`, and *forbids* `elec-initial-charge`,
`fix-electron-density`, `fix-electron-potential`, terminating via `die()` if
violated. There is no vacuum or implicit-free grand-canonical path.

So the cell this project can actually produce is **explicit water + implicit
ions**, not fully explicit. That is still the unoccupied square relative to
BEAST DB (implicit everything) and OC25 (explicit everything, constant
charge), but the claim must be stated as *explicit-solvent, implicit-electrolyte
grand-canonical* — and the approximation disclosed whenever these labels are
compared against OC25's explicit ions. `cp_dft.jdftx_setup.validate_gc_tags`
enforces the requirement at input-construction time so a sweep fails
immediately rather than after burning an allocation.

The empty cell is empty because explicit-solvent grand-canonical DFT is expensive.
That is not a reason to avoid it — it is the reason an *acquisition policy* is the
scientific contribution rather than an engineering detail. Sundararaman's presence on
BEAST DB also means the JDFTx protocol choices (CANDLE, potential referencing) have a
community-standard precedent to follow rather than invent. `[V-4]`

### 1.6 Slow-growth CV limitation is a lucky fit `[V-2]`

CP-MACE's slow-growth driver supports one CV type: **distance between two atoms**.
Normally a serious limitation. Here, the CO-dimerization reaction coordinate *is* a
C–C distance. Slow growth is therefore usable out of the box for the primary reaction,
with OPES/metadynamics reserved for the cases where a one-dimensional distance CV is
insufficient (e.g. cation-coupled or coverage-dependent pathways).

### 1.8 JDFTx runs on Aurora PVC GPUs via a SYCL port `[P]` `[V-7]`

**Established, supported project capability (2026-08-20):** JDFTx has been
ported from CUDA to SYCL and runs on Aurora's Intel Data Center GPU Max
(Ponte Vecchio). The port is **local and internal** to this project, and is
enabled and supported. Grand-canonical DFT label generation is GPU-native on
Aurora. Treated as fact throughout, with no contingency branch.

*(New provenance tag: `[P]` = project capability asserted by the team —
neither a literature-verified fact nor a speculative assumption.)*

**What the literature sweep independently establishes `[V-7]`:** upstream
JDFTx has no SYCL, oneAPI, HIP or ROCm backend. `jdftx/CMakeLists.txt`
L253–306 exposes only CUDA options and links `CUDA::cublas/cufft/cudart`; a
case-insensitive grep for `hip|rocm|sycl|oneapi|dpcpp` over the tree returns
zero hits; the GitHub language API reports 88,485 bytes of CUDA and no SYCL;
the same holds on `2.0.alpha`.

That gap is the point. The port is **not** a build flag anyone else can
flip — it is a capability this project holds and no competitor does.

#### Consequences, in order of importance

**1. Co-location — the architectural win.** This is the consequence that
actually changes the design. The σ_µ acquisition loop is inherently
tight-coupled: run biased MD → committee flags high-σ_µ frames → grand-canonical
audit → recalibrate τ_µ → choose the next state points. Before, MLIP inference
and DFT labelling would have lived on different machines, so every turn of that
loop cost a cross-system data stage plus a fresh queue wait — hours of latency
per iteration, which is what usually forces closed-loop campaigns to degenerate
into batch sweeps.

With JDFTx GPU-native on Aurora, **both halves of the loop run inside a single
allocation**. `hpc.launcher.BundledLauncher` already supports this: it slices
`$PBS_NODEFILE` into disjoint host groups and pins each task to its own group,
so MD trajectories and their DFT audits execute concurrently on the same job.
The loop turns in minutes, not queue cycles. Adopt this as the default
execution mode (§4.5) — a genuinely closed loop is a different scientific
instrument from a batch sweep, and it is worth saying so in the manuscript.

**2. Throughput becomes the enabling resource.** GPU-native GC-DFT is what
makes the explicit-water/implicit-ion label set affordable at all. A PVC rank
drives one tile with ~64 GB HBM, so an 800-atom explicit-solvent cell lands on
roughly one to two nodes rather than nine
(`hpc.aurora.nodes_for_jdftx(..., gpu=True)`). Both that multiplier and
`atoms_per_rank` remain unmeasured for this workload — a strong-scaling run is
now a Phase-0 task, because sizing is the only remaining unknown.

**3. G3 closes; G4 narrows and stops being blocking.** Grand-canonical DFT on
Aurora is no longer a gate. The only remaining Intel-GPU question is whether
fairchem/MACE run on XPU (G4) — and it now has a clean fallback that does not
disturb the critical path: MLIP inference on NVIDIA with JDFTx staying on
Aurora. That fallback costs the co-location benefit above, which is a reason to
settle G4 early, not a reason to delay.

**4. The port is a deliverable, not infrastructure.** A SYCL port of a
production grand-canonical DFT code, with performance data on 10,000+ PVC
nodes, is publishable in its own right (SC / ISC / IXPUG). It is also the
credential that makes the DOE-lab side of a FAIR Chemistry collaboration
concretely valuable rather than nominal: FAIR has models and data at scale; the
scarce complementary asset is GC-DFT throughput, and this is what supplies it.
See `PROPOSAL.md` §6, Paper 0.

**Reporting discipline.** Because the port is internal, a manuscript must say
so and state how a reader can obtain or reproduce it — reviewers who check the
upstream tree will find CUDA only. This is a citation/reproducibility
requirement, not a caveat on the capability.

Consequences, all now reflected in the code:

- **G3 narrows to logistics.** The blocking question stops being "can we run
  grand-canonical DFT on Aurora at all" and becomes pseudopotential staging
  plus confirming the CANDLE tag spelling for this build.
- **G4 narrows to the MLIP stack.** JDFTx leaving the Intel-GPU question means
  the only remaining XPU unknown is fairchem/MACE. That is a much smaller gate,
  and it has a clean fallback (MLIP on NVIDIA) that JDFTx did not.
- **Sizing changes by ~8x.** A GPU rank drives one PVC tile with ~64 GB HBM, so
  it absorbs far more atoms than a CPU rank. An 800-atom explicit-solvent cell
  lands on roughly one to two nodes instead of nine
  (`hpc.aurora.nodes_for_jdftx(..., gpu=True)`). Both the multiplier and
  `atoms_per_rank` are guesses until a strong-scaling run exists.
- **SYCL environment is explicit.** `hpc.aurora.jdftx_sycl_env()` pins
  `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`. Without it a SYCL runtime can
  silently select the OpenCL CPU device and run correctly at CPU speed — a
  failure that shows up as a mysteriously slow job, not an error.
- **Bundling matters more.** Cheaper single points mean an acquisition round
  produces many small independent jobs. `hpc.launcher.BundledLauncher` runs
  them concurrently inside one allocation over disjoint `$PBS_NODEFILE`
  slices, which suits Aurora queue policy better than many small submissions.

### 1.11 MLIP-on-XPU verdict: split `[V-8]`

Source inspection, 2026-08-20, against upstream `main` of both projects. This
settles G4 without needing an allocation, and the answer differs by route.

**CP-MACE (Route B) — runs on XPU.** `mace/tools/torch_tools.py::init_device`
is an if/elif chain over `cuda` / `mps` / `xpu` / cpu-fallback, with an
explicit `elif device_str == "xpu"` returning `torch.device("xpu")`.
*Caveat:* unlike the cuda and mps branches it does **not** assert
availability, so it will hand back an XPU device on a node that has none.
Check `torch.xpu.is_available()` yourself before trusting it — a silent CPU
fall-through would look like a very slow but working run.

**fairchem-core (Route A) — does not, as shipped.**
`src/fairchem/core/units/mlip_unit/predict.py`:

```python
assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"
self.device = get_device_for_local_rank() if device == "cuda" else "cpu"
backend = "gloo" if device == "cpu" else "nccl"
num_gpu_per_worker = 1 if device == "cuda" else 0
```

A runtime assert, not merely the `Literal["cuda","cpu"]` hint on
`get_predict_unit()`. `device="xpu"` stops immediately. Note also that
anything not `"cuda"` silently resolves to CPU, so even bypassing the assert
would run on CPU rather than error.

**The block is shallow, though.** The fairchem tree contains no `.cu`, `.cpp`
or `.cuh` sources — pure Python over torch. The CUDA-specific paths that exist
(Triton kernels for UMA-S GPU execution mode, the tf32 context manager, NCCL)
are already gated behind `device == "cuda"` and would simply be skipped. A
working XPU path needs three things: widen the assert, resolve `self.device`
for xpu, and pick a non-NCCL distributed backend (ccl or gloo).

**Consequences for the plan:**

1. **Route B is promoted from "faster to derisk" to "the Aurora-native path".**
   §4.2 already ordered it first; the reason is now much stronger than
   convenience.
2. **The co-located acquisition loop (§4.5) survives** — it only requires *one*
   MLIP stack on Aurora GPUs, and CP-MACE qualifies. This is what keeps the
   SYCL-port advantage intact.
3. **Route A gets a decision, not a blocker.** Options in order: run fairchem
   on CPU for the reproduction phase (Xeon CPU Max with HBM; far below GPU
   throughput but Phase 1 is a fixed, bounded workload); run it on an NVIDIA
   system; or patch upstream. The patch is small enough to be worth costing —
   and upstreaming an XPU path to fairchem would itself be a visible
   contribution to the FAIR Chemistry ecosystem, which §5 argues is the point
   of the collaboration.
4. **`mlip/esen_oc25.py` now fails early** with an explanation naming these
   options, rather than surfacing fairchem's bare `AssertionError`, which on
   Aurora reads as a config mistake rather than an upstream limitation.

**Dependency compatibility against the module** (`frameworks`, 2026.1.0):
`numpy 2.3.5` ✓ (`>=2.0,<2.5`), `scipy 1.18.0` ✓, `numba 0.65.0` ✓,
`ray 2.56.1` ✓, `huggingface_hub 1.16.1` ✓, `setuptools 79.0.1` ✓ (`<81`).
Absent and needed: `e3nn>=0.5` (a **hard** fairchem dependency, not merely
transitive), `ase>=3.26.0` (note: fairchem needs 3.26, not the 3.23 this
project previously pinned), plus `lmdb orjson submitit hydra-core torchtnt
monty wandb clusterscope ase-db-backends`.

**One hard install hazard, verified:** fairchem pins `torch~=2.13.0`. The
module ships `2.13.0a0+gitcf30153`, and under PEP 440 an alpha sorts *below*
the release — `2.13.0a0 < 2.13.0` — so the pin is genuinely unsatisfied
(checked with `packaging.specifiers`, including `prereleases=True`). pip will
fetch torch 2.13.0 from PyPI, a CUDA/CPU wheel, and shadow the XPU build.
`--no-deps` is mandatory. A looser floor such as `torch>=2.1` *is* satisfied;
the hazard is specific to pins at or above 2.13.0.

### 1.10 Aurora software environment `[P]`

`module load frameworks` provides, by default: the oneAPI SDK, and an
ALCF-built PyTorch with XPU support. Confirmed by the project team 2026-08-20.

Consequences:

- **Do not pip-install torch.** PyPI wheels are CUDA or CPU builds with no XPU
  backend; installing one into the module environment shadows the working
  build with a broken one. `requirements.txt` says so explicitly and omits
  torch.
- **`intel_extension_for_pytorch` is optional.** The module's torch has XPU
  upstreamed. Import it only if a stack demands it.
- **G4 narrows to op coverage, not runtime availability.** This distinction
  matters and conflating the two is how an allocation gets misspent: a working
  torch-XPU build says nothing about whether eSEN and FermiMACE actually
  execute there. The specific risks are cuEquivariance (CUDA-only accelerator
  path), CUDA-only `torch_scatter`/`torch_sparse` wheels, flash-attention, any
  `.cu` extension in the dependency tree, partial `torch.compile` coverage,
  and fp64 paths that are available but slow on PVC.

`mlip.xpu.probe_mlip_on_xpu()` is the test that settles it, exposed as
`run_campaign doctor --probe`. It runs a real **forward and backward** for
each stack on device against a small Cu/water cell. Backward matters more than
forward: a missing XPU autograd kernel produces a model that loads, returns an
energy, and cannot run MD. The probe reports that signature explicitly
(`stage="backward_forces"` with a clean forward).

It also separates *device* failures from *environment* failures. A missing ASE
install reported as "G4 FAIL on XPU" would send the MLIP work to another
machine for no reason, so the probe returns `g4_answered=false` in that case
and says so rather than issuing a verdict.

### 1.7 JDFTx output-parsing gate closed `[V-5]`

`JDFTXOutfile` public attributes confirmed: `.e`, `.forces`, `.mu`, `.converged`,
`.structure`, `.is_gc`. The repo's assumed names were correct. **G2 closes with no code
change.** `JDFTXInfile` classmethods confirmed: `from_file`, `from_str`,
`from_structure`, `from_jdftxstructure`, `from_dict`; instance methods `write_file`,
`validate_tags`, `to_pmg_structure`.

---

### 1.9 What NeuralPLexer3 contributes `[V-6]`

NeuralPLexer3 (Qiao, Ding, Dresselhaus, … Welborn; Iambic Therapeutics,
NeurIPS 2025) is a flow-matching model for biomolecular complex structure
prediction. Most of it does not transfer — MSA modules, PairFormer blocks,
Flash-TriangularAttention, DockQ/pLDDT, flow matching for structure prediction
itself. A close reading of the supplementary material yields six things that
do. Four are now implemented; two of the four fixed real defects in this
codebase.

**(a) Physical validity as an axis separate from accuracy — IMPLEMENTED.**
NP3 reports PoseBusters success rate *with and without* the `PB-valid`
requirement, because a model can be accurate on average while generating
structures that cannot exist. Their ablation (Fig. 2B) lists clash and
chirality penalties in ranking among the changes that actually moved the
headline number.

This maps onto the anchor paper's own most striking negative result:
MACE-MP-0 fails **qualitatively** at Rh(111)/water, producing an unphysical
oxygen-density spike at z ≈ 1 Å. Energy and force MAE did not catch it.

`analysis/interface_validity.py` implements the analogue, **IF-valid**: no
clashes, no solvent inside the slab, first O-density peak at a physical
height, water molecules intact, slab layers intact, bulk density near
1 g/cm³, cations solvated. A checklist, not a learned score, because its job
is to be trusted and debugged rather than optimised against. Contribution C5.

**(b) Physical validity as a HARD VETO in ranking — IMPLEMENTED, fixed a gap.**
NP3 ranks samples as `pLDDT(LG) − 1000 × (is_clash + is_chirality_violation)`
(their S.9). Validity is a veto, not a weighted term, and the coefficient is
large enough that nothing can outbid it.

The acquisition policy had no validity term at all. It would happily have
spent a grand-canonical DFT label on a trajectory whose water had migrated
into the copper — and a broken geometry produces *high* σ_µ, so it would have
ranked such candidates **first**. `SigmaMuPolicy` now carries
`veto_penalty=1000.0` / `min_if_valid_fraction=0.9` and removes vetoed
candidates outright rather than ranking them last. Verified: a candidate with
σ_µ = 0.5 eV (25× threshold) and IF-valid = 0.10 is excluded.

**(c) Confidence must be calibrated against realised error, and stratified —
IMPLEMENTED, fixed a gap.** This is the most valuable item.

NP3's confidence module is trained by cross-entropy against realised
LDDT/distance error (Algorithm S4), not on raw model variance. More useful is
*why* they changed it: "Analysis of preliminary models revealed limited
sensitivity of confidence scores to sampled conformations, despite good
correlation between accuracy and confidence scores over diverse targets." They
attribute it to under-training plus insufficient paired conformational data
for the same topology, and fix it by generating multiple hypotheses per
topology and adding an InfoNCE contrastive loss to discern minor quality
differences.

That failure mode is *exactly* what σ_µ frame selection is exposed to. σ_µ can
look strongly predictive simply because Cu(100) and Cu(310) are different
problems, while carrying no information about which frame of one trajectory
deserves a DFT audit. Between-group signal justifies choosing state points;
only within-group signal justifies choosing frames.

`calibrate_threshold` now takes `groups=` and reports `between_spearman` and
`within_spearman` separately (the latter on group-mean-centred values), and
`select_frames` **refuses** without demonstrated within-group discrimination,
falling back to evenly-spaced frames. Verified on a deliberately constructed
trap: aggregate Spearman +0.955, between-group +1.000, within-group +0.215 →
frame selection correctly declined. Without stratification that calibration
report would have read as a clean success while the audit budget went to noise.

*Open option:* a learned µ-error head trained the NP3 way would replace an
N-model committee with one forward pass. Revisit once labels exist.

**(d) Physics-informed prior instead of noise — IMPLEMENTED.** NP3 replaces the
Gaussian prior with a globular-polymer prior built by 64 steps of overdamped
Langevin dynamics under a cheap harmonic model (Algorithm S3:
`drift = 2·d_bond + d_entity/ent_r² + d_res/res_r² − X/sphere_r²`,
`X ← X + dt·drift + 2√dt·ε`), and reports that a better prior straightens the
flow and cuts integrator steps.

`systems/langevin_prior.py` transplants the recipe with interface-appropriate
terms: soft pairwise exclusion, one-sided slab repulsion, z-slot confinement,
in-plane minimum image, annealed noise. Measured against the existing
rejection packer on a 22.9 Å × 22.9 Å × 8 Å slot: rejection succeeds at
nominal density (140/140) but **saturates at ~142 molecules regardless of the
request** — 142/161 and 142/182 at 1.15× and 1.3× nominal, 100/105 in a 5 Å
slot. It silently under-fills, which produces a wrong double layer for a
reason that appears in no log. The Langevin prior places all requested
molecules and reaches 2.61 Å minimum O–O separation in ~1 s. It needs 256
steps rather than NP3's 64 — a dense fluid is not a sparse polymer; 64 leaves
2.24 Å, and 512 buys only 2.68 Å. Generated interfaces pass IF-valid.

**(e) Benchmark the RESPONSE, not the value — IMPLEMENTED.** NP3 built
ConfBench because existing benchmarks did not measure ligand-*induced*
conformational change, and used a symmetric, bounded, normalised differential
score (S.11.2) so systems with large and small responses stay comparable.

The same hole exists here and matters more: response to potential is the
entire reason a constant-potential model exists. A potential-conditioned MLIP
can hit low force MAE at every training potential while learning no dependence
on the conditioning variable — absolute values near the ensemble mean, and
dG/dU wrong. No absolute-accuracy metric can see this.

`analysis/potential_response.py` adapts their score to (U₁ → U₂) and adds a
slope check. Verified on synthetic models: a flat mean-predictor with
**MAE = 0.048 eV** — which would pass unremarked in most MLIP papers — is
correctly flagged unresponsive (slope +0.002 vs reference +0.160), and a
wrong-sign model is excluded from ranking rather than ranked poorly.
Contribution C6.

**(f) Symmetry correction for indistinguishable entities — DESIGN NOTE.** NP3
aligns and greedily permutes identical entity copies before computing loss
(S.5), so identical chains cannot be penalised for label ordering. Water
molecules are the same problem. This does not affect σ_F or σ_µ (all committee
members see one indexed `Atoms` object), but it does affect any *structural*
comparison across trajectories or against a reference. Needed before any
structural error metric is reported; not needed for the acquisition loop.

**Also noted, not adopted:** rollout every N=10 iterations rather than every
one (AF3 uses N=1) to amortise confidence training; decoder parallelism —
multiple independent coordinate initialisations per training step — whose
analogue is multiple independent MD seeds per state point per round.

## 2. Revised contributions

| | Contribution | Novelty basis | Risk |
|---|---|---|---|
| **C1** | First **explicit-solvent, grand-canonical** DFT reference set for Cu(100)/Cu(310) CO dimerization; CC-vs-CP discrepancy mapped vs. cell size, facet, cation | fills the empty 2×2 cell (§1.5) | cost — mitigated by C3 |
| **C2** | Potential-aware MLIP for Cu/water: Route A (µ-head on eSEN-OC25) vs Route B (FermiMACE retrained on Cu), benchmarked head-to-head | CP-MACE is Ni–N–C/Au, not Cu `[V-2]`; anchor paper asks for exactly this `[V-1]` | Route A must implement `get_mu()` `[C]` |
| **C3** | **σ_µ-triggered acquisition**: agents use constant-potential committee disagreement in the *Fermi level* to schedule grand-canonical DFT, plus mid-run OPES hyperparameter control | σ_µ as an acquisition signal is CP-native and, as far as verified, unclaimed | prior art check pending (§9) |
| **C4** | Open artifacts: the CP label set, the µ-heads, and the §3.1 reproduction table as a scored benchmark task | complements OC25 rather than competing | licensing (G8) |
| **C5** | **IF-valid**: a physical-validity benchmark for electrified interfaces, reported alongside accuracy | after NeuralPLexer3's PB-valid `[V-6]`; directly quantifies the anchor paper's MACE-MP-0 failure mode `[V-1]` | low — geometry only |
| **C6** | **Potential-response benchmark**: normalised differential score + dG/dU slope check, exposing models that hit low MAE while ignoring their conditioning variable | after NP3's ConfBench `[V-6]`; the crowded potential-conditioned-MLIP field (§3.4) has no shared response metric | low — post-processing |

**C3 is the headline.** C1 and C2 make it credible. C4 is what makes FAIR Chemistry
want in.

---

## 3. Falsifiable targets

### 3.1 Reproduction gate — Phase 1 is not complete until these are hit `[V-1]`

| Quantity | Facet | Condition | Target |
|---|---|---|---|
| Dimerization barrier | Cu(100) | neutral | ~0.64 eV |
| Dimerization barrier | Cu(310) | neutral | ~0.57 eV |
| Dimerization barrier | Cu(310) | −23 µC/cm² | ~0.49 eV |
| ΔG_rxn | Cu(100) | neutral | ~0.375 eV |
| ΔG_rxn | Cu(310) | neutral | ~0.088 eV |

Qualitative targets that must also survive: weak charge/cation sensitivity with
appreciable stabilization only at the most negative charge densities; Cu(310) more
favorable at modest reducing potentials. `[V-1]`

### 3.2 The primary quantitative result

**ΔΔG(cell size, facet, cation) ≡ ΔG_CP − ΔG_CC.**

The anchor paper's claim is that large cells make this small. `[V-1]` Confirming it is
a useful negative result that validates a large body of constant-charge work;
refuting it is a bigger result. Either outcome is publishable, which is the mark of a
well-posed question. Report it as a curve in 1/A, not a single number.

### 3.3 The agentic metric — pre-register before running

**Cost to reach the §3.1 table within tolerance**, measured in (i) grand-canonical DFT
single points and (ii) ns of biased MD, for:

- **B0** uniform grid over (facet × charge/potential × cation) — the baseline;
- **B1** random subsampling at matched budget — guards against "beats a bad baseline";
- **B2** force-uncertainty-triggered acquisition — the ablation that isolates σ_µ;
- **A1** the full σ_µ-triggered agentic loop.

**A1 must beat B2 for C3 to mean anything.** If A1 ≈ B2, the finding is "generic MLIP
uncertainty is sufficient; the potential-specific signal adds nothing" — report it and
retitle. Pre-register B0's grid resolution before the first A1 run; a baseline chosen
after seeing results is not a baseline.

---

### 3.4 What the deep-research pass changed `[V-7]`

A 105-agent adversarially-verified literature sweep (2026-08-20) returned four
findings that alter the plan. Two are opportunities, two are threats.

**THREAT — the potential-conditioned MLIP niche is crowded.** At least four
independent lines now exist: CP-MACE (JCTC 2025); **PE-MACE / EEP-MLFF
(arXiv:2604.07322, April 2026)**, which linearly embeds a scalar potential into
the initial node features; **TRECI (arXiv:2511.19338)**, which puts seven
discrete Franken random-Fourier-feature readout heads on *frozen* MACE node
features, one per bias from −0.50 to −2.00 V vs SHE; and a further
grand-canonical equivariant potential (10.1021/acs.jctc.5c01381). C2 as
originally framed — "build a potential-aware MLIP" — will not survive review as
novel. It has to be reframed as *a work-function head on the OC25/UMA
foundation-model line specifically*, which is the part nobody has done.

**THREAT — the agentic layer is also crowded.** ChemGraph (arXiv:2506.06363),
**Catalyst-Agent (arXiv:2603.01311, closed-loop autonomous ORR/NRR/CO2RR
screening)**, TritonDFT, LARA, AutoDFT, and — most directly —
**"Multi-Agent Orchestration for High-Throughput Materials Screening on a
Leadership-Class System" (arXiv:2604.07681, April 2026), an Argonne
planner–executor framework running gpt-oss-120b + MCP + Parsl on Aurora.**
"LLM agents drive simulations on a DOE machine" is now demonstrably not a
paper. The surviving distinction is narrow but real: those systems have agents
control *job submission*; this project has agents control *the physics* — µ
setpoints, bias parameters, convergence-triggered stopping. Say it that way or
not at all.

**OPPORTUNITY — the hook is a verbatim quote, not an inference.** From the
anchor paper's *Outlook and future directions*, with Ulissi as coauthor:

> "An important quantity that is currently not explored in this work is the
> interface workfunction… it would be highly desirable to have ML models that
> can predict the interface workfunctions directly. Access to the interface
> workfunction during the simulation can also enable constant potential
> (grand-canonical) simulations to estimate the thermodynamics and kinetics of
> electrochemical reactions. Developing models that can accurately predict both
> the Fermi level and the vacuum potential, and in turn the interfacial
> workfunction, is an exciting direction for future research in the study of
> solid–liquid interfaces."

That is FAIR Chemistry publicly posting the exact vacancy this project fills.
It is the strongest asset in the proposal and belongs in the first paragraph of
any outreach.

**OPPORTUNITY — a new framing nobody owns: cross-engine calibration.** Four
mutually incompatible constant-potential conventions now coexist — JDFTx
`target-mu` with implicit electrolyte, VASP double-reference/FCP (TRECI),
explicit-ion constant charge (OC25), and potential-conditioned MLIPs — with no
cross-walk between them. Quantifying how they disagree on the *same* reaction at
the *same* potential, and publishing the transfer functions plus the energy
bookkeeping needed to mix them in one training set, is unglamorous, citable by
everyone, low preemption risk, and directly enabled by machinery this project
already needs. **Promote this to a co-equal framing** (§8).

The bookkeeping hazard is not hypothetical: in a shipped pymatgen fixture
(`GC_ion.out`) the grand potential G and Helmholtz F differ by **~1683 eV**
(Etot −1120.8545, F −1120.8565, muN −61.866, G −1058.9905 Ha). `JDFTXOutfile.e`
returns **G** for a `target-mu` run, and the distinction lives in `.etype` —
*not* `.eopt_type`, which records the minimiser and which atomate2 itself
conflates. `data.schema.CPDFTLabel` now carries `etype`, exposes
`helmholtz_f_ev`, and `data.xyz.write_cp_mace_xyz` refuses to emit a
mixed-energy-type dataset.

Also corrected: `is_gc` reports only that the `target-mu` *tag was present*,
not that a grand-canonical SCF converged at that µ
(`jdftxoutfileslice.py:386` computes it as `key_exists('target-mu', text)`).
Harvest now sets a separate `gc_converged` from `is_gc AND converged AND
on-target-µ`.

## 4. Architecture, revised

```
L5  Orchestration    Parsl / Globus Compute · GH200 · A100 · (Aurora gated)
L4  Agents           reasoning · exploration · sampling-control · σ_µ audit scheduler
L3  Sampling         slow-growth (C–C distance CV) | PLUMED OPES · NoseHoover(targetmu)
L2  Surrogate        Route A: eSEN-OC25 + µ-head   | Route B: FermiMACE
                     committee of ≥2 → σ_F, σ_µ
L1  Ground truth     JDFTx grand-canonical (target-mu, CANDLE), explicit solvent
```

### 4.1 L1 — JDFTx

- Grand-canonical via native `target-mu`: electron count floats at fixed electron
  chemical potential. `[V-4]`
- **CANDLE** as the solvation model — recommended for strongly charged/polar solutes,
  handles cation/anion charge asymmetry, maintains cell neutrality in GC-DFT, and has
  a calibrated point of zero charge to reference potentials against. `[V-4]`
  Following BEAST DB's parameter conventions gives comparability for free. `[V-3]`
- I/O via `pymatgen.io.jdftx`; attribute names confirmed (§1.7). ASE's JDFTx
  calculator retained only for the PLUMED-coupling case. `[C]`
- **Calibrate the `target-mu` sign convention against a known Cu work function before
  trusting any absolute potential.** Unchanged, non-negotiable. `[A]`

### 4.2 L2 — the two routes

The distinguishing requirement is now concrete rather than aesthetic:

> **Any surrogate used with the CP-MACE integrator must implement `get_mu()`.** `[C]`

**Route A — µ-head on eSEN-OC25.** Add a work-function/µ head; expose it as
`get_mu()`. Pros: what the anchor paper asked for `[V-1]`; base model already validated
on Cu/water `[V-1]`; avoids the MACE-MP-0 interfacial-density failure `[V-1]`; the
resulting artifact is directly interesting to FAIR Chemistry. Cons: gated checkpoints
(G1); new head to train.

**Route B — FermiMACE retrained on Cu.** Use CP-MACE as-is with a Cu/water dataset in
their format. Verified training invocation `[V-2]`:

```
--model="FermiMACE"        # node augmentation;  FermiMACE_2 = global state
--loss="fermi_weighted"
--error_table="Fermi_PerAtomRMSE"
--energy_weight=1.0  --forces_weight=100.0  --potential_weight=10.0
--r_max=5.0  --batch_size=10  --max_num_epochs=300
```

Dataset format: extended-XYZ with `electron=<N>` and `potential=<µ>` on the comment
line (example values from their `init.xyz`: `electron=661.7`, `potential=-3.407347`). `[V-2]`
Pros: potential is native, not bolted on; two architectural variants to compare;
reference implementation exists. Cons: no license (G8); demonstrated on Ni–N–C/Au, not
Cu `[V-2]`; upstream unmaintained since 2025-09 `[V-2]`.

**Decision `[A]`:** build both. Route B **first** — it is the faster path to a working
constant-potential Cu MD and derisks the whole plan, since it needs no new architecture.
Route A second, as the contribution FAIR Chemistry actually wants. The comparison
answers a real question: *does a potential-native model trained on less relevant data
beat an interface-specialized model with a potential head bolted on?* Nobody has
answered that.

**Committee.** Both routes train ≥2 models with different seeds, mirroring upstream's
`AverageForceCalculator`. σ_F and σ_µ are the acquisition signals. `[C]`

**OMol25/UMA.** Baseline and transferability probe only. Anchor-paper numbers show
UMA-OC20 trailing eSEN-OC25 interfacially, and UMA-S-ft(OC25) still trailing on strict
splits. `[V-1]` The interesting question — *does a µ-head transfer across UMA's task
heads?* — is a foundation-model question, which is the FAIR Chemistry hook (§7). It is
not on the accuracy-critical path and must not be allowed to become one.

### 4.3 L3 — sampling

Two modes, both upstream-verified:

- **Slow growth** — `increm` walks the C–C constraint; single distance CV, which fits
  CO dimerization exactly (§1.6). Cheapest route to a first constant-potential Cu
  profile. `[V-2]` `[C]`
- **OPES/metadynamics** — `Plumed(calc=…, input=…, timestep=1.0*units.fs, atoms=…, kT=…)`
  wraps the *calculator*; `NoseHoover` remains the integrator in both modes. PLUMED and
  the integrator are not substitutes — they act at different layers. `[C]`

Upstream's `integrator.run(1)`-in-a-loop pattern is the intervention point for the
sampling-control agent. `[C]`

### 4.4 L4 — agents

**(a) Electrocatalysis reasoning.** Domain-grounded; deterministic tool-call comparison
against §3.1 so the model cannot hallucinate agreement. `[C]`

**(b) Phase-space exploration** — abductive `AbnormalityDetectionAgent`, inductive
`PatternFindingAgent`, `BOAssistedPatternFindingAgent` past 30 records; every proposal
carries a `strategy` tag for traceability. After A-Lab GPSS, with attribution. `[C]`

**(c) Sampling control.** Adjusts OPES `BARRIER`/`PACE` and stopping between
`run(1)` calls, against block-averaged ΔG stability.

**(d) σ_µ audit scheduler — the new centerpiece.** Policy sketch:

```
per window of biased MD:
    σ_F, σ_µ  ←  committee spread over the window
    if σ_µ > τ_µ:                    →  JDFTx grand-canonical single point
                                        (the electronic boundary condition is
                                         in doubt — a force-only trigger misses this)
    elif σ_F > τ_F and near barrier: →  cheaper: extend sampling, or a CC audit
    else:                            →  continue
    τ's recalibrated as labels accumulate
```

The ablation against B2 (force-only trigger, §3.3) is what tests whether σ_µ carries
information beyond σ_F. That ablation *is* the experiment.

**Validation agent.** Encode the *intricacies of computational electrochemistry*
guardrails — potential referencing, finite-size effects, implicit-vs-explicit
solvation, CC-vs-CP ensemble choice — as concrete tool-called checks, not prompt text.
Currently prompt-level only. `[?]`

**LLM backend.** ALCF Inference Endpoints, OpenAI-compatible, Globus auth. Tool-calling
is mandatory, so reasoning-only models are disqualified regardless of tutorial
examples. `[C]` No live call has ever been made. `[?]`

### 4.5 L5 — orchestration: co-located acquisition loop

**Default mode, enabled by the SYCL port `[P]`: one Aurora allocation holds the
whole loop.**

```
  PBS job (N nodes, one allocation)
  └── BundledLauncher slices $PBS_NODEFILE into disjoint host groups
      ├── group A ──> constant-potential MD  (committee: sigma_F, sigma_mu)
      ├── group B ──> constant-potential MD
      ├── group C ──> JDFTx GC-DFT audit of frames flagged this round
      └── group D ──> JDFTx GC-DFT audit
              |
              v  harvest -> recalibrate tau_mu -> select next round
```

Why this matters beyond convenience: the acquisition loop's value is that it
*adapts*, and adaptation rate is bounded by loop latency. Split across two
machines, each turn costs a data stage plus a queue wait — hours — and the
campaign degenerates into a batch sweep with extra steps. Co-located, a turn is
the wall time of one MD window plus one DFT audit. That is the difference
between a closed-loop instrument and a scripted parameter scan, and it is a
claim the manuscript can make concretely.

Sizing within an allocation: `BundledLauncher(nodes_per_task=k)` gives
`capacity = len(hosts) // k` concurrent tasks. Throttle submissions to
`capacity`; the cursor wraps rather than oversubscribing, so the caller is
responsible for not exceeding it.

Fallback if G4 resolves negatively (fairchem/MACE not viable on XPU): MLIP
inference on NVIDIA, JDFTx labelling stays on Aurora, loop latency degrades to
the cross-system case. Functional, slower, and worth avoiding — hence settle G4
in Phase 0.

---

## 5. Phased plan

Durations are working estimates `[A]`, and every phase has an explicit exit test.

### Phase 0 — Gate clearance (~1–2 weeks). Nothing downstream is meaningful first.

| Task | Exit test |
|---|---|
| Request OC25 checkpoint access (gated HF, FAIR Chemistry License, legal name/DOB/org + AUP) | `pretrained_mlip.get_predict_unit(...)` returns |
| Email CP-MACE authors for an explicit license grant | written reply on file (G8) |
| JDFTx SYCL build on Aurora: stage pseudopotentials, confirm `pcm-variant CANDLE` for this build | one Cu slab GC single point converges on PVC; `.mu`, `.etype`, `nElectrons` all parse |
| **Strong-scaling run** for GC-DFT on PVC — replaces the `atoms_per_rank` guess | measured nodes-vs-walltime for 200/400/800-atom cells; `hpc.aurora.nodes_for_jdftx` retuned |
| Co-location rehearsal: `BundledLauncher` running MD + DFT audit concurrently in one allocation | both task types complete in one job, disjoint host groups, no interference |
| One live ALCF LLM call with a tool-calling model | `check_model_availability()` runs; one real tool round-trip |
| fairchem/CP-MACE op coverage on XPU (G4) | `run_campaign doctor --probe` returns `g4_answered=true` with a per-stack verdict. Decides co-located vs split loop; does not block Aurora hours |

`JDFTXOutfile` attribute verification — **closed, no work needed** (§1.7).

### Phase 1 — Reproduction (~3–4 weeks). The credibility floor.

Run eSEN-OC25 + PLUMED OPES on Cu(100)/Cu(310), >800-atom cells, and hit §3.1.
Every module in `systems/`, `mlip/`, `md/`, `analysis/` is currently verified only by
pure-numpy unit tests in a sandbox with no ASE/pymatgen/fairchem/PLUMED — **zero real
MD steps have ever been run.** This phase converts the repo from plausible to real.

*Exit:* §3.1 reproduced within stated error bars. If not, stop and debug — do not proceed.

### Phase 2 — Constant-potential MD, fast path (~2–3 weeks)

Run upstream CP-MACE unmodified on their pretrained Au model to validate the toolchain
end-to-end, then slow-growth on Cu with the C–C distance CV. Confirm `get_mu()`,
`targetmu` tracking, and NH-chain stability before investing in training.

*Exit:* a constant-potential Cu trajectory in which measured µ tracks `targetmu`.

### Phase 3 — CP-DFT label generation (~3–4 weeks, allocation-bound)

*Shortened from 4–6 weeks: GPU-native GC-DFT on Aurora is the throughput
assumption this phase rests on `[P]`. Revisit after the Phase-0 scaling run —
that measurement, not this estimate, sets the real duration.*

Calibrate `target-mu` against Cu work function. Follow BEAST DB conventions. `[V-3]`
Generate explicit-water / implicit-ion grand-canonical labels — **this is C1 and the reusable
artifact**. Structures drawn from Phase 1/2 trajectories; the *selection* is where
Phase 5's policy eventually plugs in. Bootstrap with a uniform grid so B0 exists.

*Exit:* labeled set with converged `.mu`, `.e`, `.forces`; ΔΔG vs 1/A curve for at
least one facet.

### Phase 4 — Both surrogate routes (~4 weeks)

Route B first (FermiMACE on Cu labels, verified flags §4.2), then Route A (µ-head on
eSEN with `get_mu()`). ≥2 seeds each for the committee.

*Exit:* both routes reproduce Phase-3 CP-DFT ΔG within tolerance; σ_µ calibrated
against realized CP-DFT error — an uncalibrated uncertainty is not an acquisition signal.

### Phase 5 — Closed agentic loop (~4–6 weeks). The headline.

Run B0, B1, B2, A1 (§3.3) on the same pre-registered budget.

*Exit:* the cost-to-reproduce comparison, including the A1-vs-B2 ablation, reported
whichever way it comes out.

### Phase 6 — Write-up

- **Paper 1 (physics/method):** C1 + C2. Target: *JACS* / *Nature Catalysis* /
  *ACS Catalysis*.
- **Paper 2 (agentic science):** C3 + C4. Target: *Nature Machine Intelligence* /
  *Digital Discovery* / *JCTC*, or a NeurIPS AI4Science track.
- **Artifacts:** label set, µ-heads, §3.1 as a scored benchmark task.

---

## 6. Gates and risks

| # | Gate | Status | Fallback |
|---|---|---|---|
| G1 | OC25 **checkpoint** access — gated HF repo, Meta FAIR Chemistry License. Dataset is CC-BY-4.0; **checkpoints are not**. `[C]` | not requested | train eSEN-class from the open dataset; slower, viable |
| G2 | `JDFTXOutfile` attribute names | **CLOSED** `[V-5]` | — |
| G3 | JDFTx GPU execution on Aurora | **CLOSED** — SYCL port in hand `[P]`. Residual, non-blocking: pseudopotential staging, `pcm-variant CANDLE` tag check for this build, and a strong-scaling run to replace the `atoms_per_rank` guess | — |
| G4 | MLIP stacks on Intel XPU | **ANSWERED by source inspection, 2026-08-20 `[V-8]`. Split verdict.** CP-MACE: `init_device` has an explicit `xpu` branch → **Aurora-native**. fairchem-core: `predict.py` asserts `device in ["cpu","cuda"]` → **does not run on XPU as shipped**. | Route B on Aurora GPUs (co-located loop intact). Route A: CPU on Aurora, or NVIDIA, or a shallow patch — fairchem has no CUDA sources. |
| G5 | Agent LLM with working tool-calling | **restructured.** Default is now a LOCAL vLLM-XPU server (`vllm 0.26.1+xpu` ships in the `frameworks` module) `[P]`, which removes the outbound-network and interactive-Globus dependencies entirely. Remaining check is the tool-call parser, verified in one round-trip by `VLLMServer.verify_tool_calling()` | remote ALCF endpoint via `ELECTROCHEM_LLM_BACKEND=alcf`, for login-node driving |
| G6 | Long-range electrostatics — anchor gap 3 `[V-1]` | unaddressed | scope-limit to systems without strong electrostatic ordering, and say so in the manuscript |
| G7 | Adaptive-OPES prior art | pending §9 | re-frame on the σ_µ audit policy; cite head-on |
| **G8** | **CP-MACE has no LICENSE — all rights reserved** `[V-2]` | **open** | live-import from a user-supplied checkout (already the design); never vendor or redistribute their code or `.model` files; get written permission before releasing any derivative |
| G9 | CP-MACE unmaintained since 2025-09-16 `[V-2]` | accepted | pin a commit; Route A is the hedge |
| G10 | Scoop risk — the anchor authors are a large, funded group who know this gap | live | the σ_µ acquisition policy and the DOE-lab GC-DFT capability are the defensible parts; move on Phase 0 now, and consider §7 outreach early rather than late |

G10 deserves a plain statement: the people best positioned to close this gap are the
people who wrote the paper naming it. That is an argument for collaborating (§7)
rather than racing, and for moving quickly on the gates.

---

## 7. The FAIR Chemistry / Ulissi case, revised

Ulissi is an author on the anchor paper. `[V-1]` The framing changes accordingly.

**Do not pitch:** "here is an idea about your model."
**Pitch:** "your paper's Discussion names two open items. We have a DOE-lab
grand-canonical capability, and here is a first result on one of them."

Five concrete arguments:

1. **Answers a stated open item, verbatim.** "Couple OC25 models with grand-canonical
   approaches or develop models that directly predict the work function." `[V-1]`
   Route A is literally that: a µ-head on their architecture.
2. **Supplies labels they cannot cheaply make.** Explicit-solvent grand-canonical DFT
   sits outside the pipeline that generated OC25. BEAST DB shows the community has
   grand-canonical *implicit* solvent at 20k scale `[V-3]`; nobody has the explicit
   cell. Complementary capability, not duplicated effort.
3. **Probes UMA where UMA is weak, constructively.** Their own numbers show UMA trailing
   eSEN interfacially. `[V-1]` "Does a potential head transfer across UMA task heads?"
   is a foundation-model question — their actual research question, not ours.
4. **Ships in the format their ecosystem consumes.** Dataset + checkpoints + a scored
   benchmark task (§3.1). Leaderboard-shaped.
5. **The agentic acquisition layer is orthogonal to what FAIR builds.** Their strength
   is models and data at scale. A σ_µ-triggered DFT-in-the-loop policy is a different
   contribution, not a competing one.

**The ask, bounded:** co-authorship on the label-set/benchmark release plus advisory
input on the µ-head architecture. Low cost to them, high visibility, no open-ended
commitment.

**Secondary channels worth using:** Govindarajan and Varley at LLNL (DOE-lab peers,
already collaborating on this exact system); Gauthier at Texas Tech; and separately the
Liu group, whom the G8 license question gives a natural, non-presumptuous reason to
contact.

**Timing:** the credible moment to reach out is after Phase 1 reproduces §3.1 — that
demonstrates capability rather than intent — but G10 argues against waiting past Phase 3.

---

## 8. Immediate next actions

1. Request OC25 checkpoint access (G1). Blocks Phase 1. Do today.
2. Email CP-MACE authors re: license (G8). Blocks any release. Do today.
3. Scope the JDFTx + CANDLE build (G3). Blocks Phase 3, longest lead time.
4. One live ALCF tool-calling round-trip (G5).
5. Resolve fairchem-on-XPU (G4) before any Aurora request.
6. Pre-register the B0 grid (§3.3) before any A1 run.
7. Wire the validation agent's guardrail checks to real functions, not prompt text.

Note that (1)–(5) are mechanical, not intellectual, and are what currently stand between
this project and its first real number.

---

## 9. Source ledger

| Tag | Source |
|---|---|
| `[V-1]` | Sahoo, Maraschin, Varley, Levine, Ulissi, Zitnick, Takemura, Gauthier, Govindarajan, Shuaibi. *Insights into CO dimerization at electrified Cu interfaces from large-scale machine learning simulations.* arXiv:2509.17862, v1 2025-09-22, v2 2026-06-08. DOI 10.48550/arXiv.2509.17862 — https://arxiv.org/abs/2509.17862 |
| `[V-2]` | CP-MACE — https://github.com/yuanyue-liu-group/CP-MACE (README, tree, and GitHub API metadata read 2026-08-20). Paper: Wang, Fang, Huang, Liu. *Constant-Potential Machine Learning Force Field for the Electrochemical Interface.* JCTC 2025, DOI 10.1021/acs.jctc.5c00784 |
| `[V-3]` | Tezak et al. *BEAST DB: Grand-Canonical Database of Electrocatalyst Properties.* arXiv:2405.20239 — https://arxiv.org/abs/2405.20239 |
| `[V-4]` | JDFTx — https://jdftx.org/ ; Sundararaman et al., *Grand canonical electronic density-functional theory*, J. Chem. Phys. 146, 114104 (2017) — https://pubs.aip.org/aip/jcp/article/146/11/114104/195000/ ; BEAST JDFTx practice notes — https://beast-echem.org/workshops/2022/jdftx.pdf |
| `[V-5]` | `pymatgen.io.jdftx` API — https://pymatgen.org/pymatgen.io.jdftx.html |
| `[V-6]` | Qiao, Ding, Dresselhaus, Rosenfeld, Han, Howell, Iyengar, Opalenski, Christensen, Sirumalla, Manby, Miller, Welborn. *NeuralPLexer3: Accurate Biomolecular Complex Structure Prediction with Flow Models.* NeurIPS 2025 (paper PDF read directly 2026-08-20) |
| `[V-7]` | Deep-research pass, 2026-08-20, 105 agents, adversarially verified. Primary sources include: https://jdftx.org/CommandTargetMu.html · jdftx `parser.cpp`, `CMakeLists.txt` L253–306 · https://api.github.com/repos/shankar1729/jdftx/languages · pymatgen-core `jdftxoutfileslice.py`, `joutstructure.py` · atomate2 PR #955 (merged 2026-03-12) · PE-MACE arXiv:2604.07322 · TRECI arXiv:2511.19338 · Catalyst-Agent arXiv:2603.01311 · ChemGraph arXiv:2506.06363 · ANL Aurora orchestration arXiv:2604.07681 · OMol25 arXiv:2505.08762 · UMA arXiv:2506.23971 |
| `[V-8]` | fairchem-core `pyproject.toml` and `src/fairchem/core/units/mlip_unit/predict.py`, `src/fairchem/core/calculate/pretrained_mlip.py` (github.com/facebookresearch/fairchem, main, read 2026-08-20); CP-MACE `mace/tools/torch_tools.py`; OpenEquivariance README + GitHub API (github.com/PASSIONLab/OpenEquivariance) |
| `[C]` | Source read directly: `simulation/slow_growth/integrator.py`, `simulation/metadynamics/simulate.py` (CP-MACE); this repo's `agents/`, `cp_dft/`, `md/`, `mlip/` |

**Pending:** a full deep-research pass covering adaptive-OPES prior art (G7), the
agentic-HPC landscape, and FAIR Chemistry collaboration precedent is in flight; §3.3,
§6/G7 and §7 will be updated from it. Nothing in this document depends on that pass
being favorable.
