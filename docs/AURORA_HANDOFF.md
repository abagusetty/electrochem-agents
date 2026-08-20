# Aurora / Sunspot Handoff

**Written:** 2026-08-20, from a Mac with no simulation stack installed.
**Read this first if you are picking the project up on Sunspot or Aurora.**

Sunspot and Aurora share the architecture (Intel Xeon CPU Max + Data Center GPU
Max / Ponte Vecchio, oneAPI, PBS Pro), so everything here applies to both.
Sunspot is the sane place to shake this out: shorter queues, same failure modes.

---

## 0. What state the project is actually in

**Nothing in this repository has ever executed against a real simulation
stack.** Zero MD steps, zero DFT calculations, zero LLM calls, zero jobs
submitted. Every module is written and logic-tested with numpy only.

That is not a disclaimer, it is the task list. The whole point of this document
is that the first real number is now a few mechanical steps away, and those
steps are ordered below by what unblocks the most.

Verified so far (no allocation needed): all modules import with numpy alone;
PBS script generation; `mpiexec` construction; SYCL env; potential-calibration
round-trip against the Cu(100) work function; JDFTx tag assembly and its
grand-canonical dependency guards; all four acquisition policies; σ_µ threshold
calibration including the stratified within/between split; pre-registration
fingerprinting; `BundledLauncher` throttling; the Langevin solvent prior against
the rejection packer; IF-valid on a generated interface.

---

## 1. Environment

`module load frameworks` supplies oneAPI and an XPU-enabled PyTorch
(`torch 2.13.0a0+gitcf30153`). Do **not** pip-install torch — a PyPI wheel is
CUDA or CPU only and will shadow the working build.

```bash
module load frameworks
python -m venv --system-site-packages elechem-venv
source elechem-venv/bin/activate
pip install --no-deps -r requirements-aurora.txt     # ase pymatgen ag2 (+e3nn)
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

That last line must still print `2.13.0a0+gitcf30153 True`. If the version
changed, something replaced torch — see the `--no-deps` note in
`requirements-aurora.txt`.

**Confirmed on Sunspot 2026-08-20** `[V-9]`: `frameworks/2026.1.0` gives
`torch 2.13.0a0+gitcf30153`, `torch.xpu.is_available() True`, **12 devices**
(6 × Max 1550 × 2 tiles), `numpy 2.3.5`, `triton-xpu 3.7.2`.

The `--no-deps` hazard is **real, not hypothetical** — reproduced live: fairchem
pins `torch~=2.13.0`, and `2.13.0a0` sorts *below* `2.13.0` under PEP 440, so an
unconstrained `pip install fairchem-core` genuinely resolves a PyPI torch. A
constraints file is the cheap guard:

```bash
printf 'torch==2.13.0a0+gitcf30153\nnumpy==2.3.5\n' > /tmp/aurora-constraints.txt
pip install -c /tmp/aurora-constraints.txt <pkg>     # fails loudly instead of swapping torch
```

Absent from the module and therefore unavoidable installs (checked against all
276 module packages): `ase`, `pymatgen`, `e3nn`, `fairchem-core`, plus
`lmdb orjson submitit hydra-core torchtnt monty wandb clusterscope
ase-db-backends pymatgen-core opt_einsum_fx pyre-extensions tensorboard`.
None of these shadow a module-provided package.

Sunspot compute nodes **do** have outbound network through
`http_proxy=http://proxy.alcf.anl.gov:3128`, so `pip` and `huggingface_hub`
work in-job. This contradicts the "no outbound network" note elsewhere in this
document; trust this line, it was measured. `[V-9]`

### Already in the module (do not install)

`numpy 2.3.5` · `scipy` · `scikit-learn` · `matplotlib` · `pandas` · `h5py` ·
`sympy` · `networkx` · `einops` · `mpi4py` · `PyYAML` · `requests` ·
`openai` · `typer` · `pytest` · `ray` · `numba` · `transformers` ·
`huggingface_hub` · **`triton-xpu 3.7.2`** · **`vllm 0.26.1+xpu`** ·
`mcp` · `dpctl` / `dpnp`

Absent and needed: `ase>=3.26.0`, `pymatgen>=2025.4`, `ag2`, `e3nn>=0.5`.

---

## 2. Do this first — 5 minutes, settles the biggest open question

```bash
scp scripts/check_aurora_env.py <you>@sunspot:~/      # no repo dependency
pip install --no-deps e3nn ase
python check_aurora_env.py                            # login node: env only
```

Then inside a job:

```bash
qsub -I -A <project> -q debug -l select=1 -l walltime=00:30:00 -l filesystems=flare:home
module load frameworks && source elechem-venv/bin/activate
python check_aurora_env.py
```

Six checks, cheapest-decisive first. **Check 2 is the one that matters.**

| # | Check | Why |
|---|---|---|
| 1 | torch + XPU, matmul fwd+**bwd**, fp64 | settles the runtime |
| 2 | **e3nn CG tensor product fwd+bwd, 4096 edges** | see below |
| 3 | `index_add` / `scatter_reduce` fwd+bwd | message passing |
| 4 | Triton kernel compiles + runs on XPU | see §6 |
| 5 | fairchem eSEN energy + forces (`--deep`) | Route A |
| 6 | CP-MACE FermiMACE + `get_mu()` (`--deep`) | Route B |

Equivariant MLIPs spend their time in Clebsch-Gordan tensor products, and both
accelerators are closed to Intel: cuEquivariance is NVIDIA-only,
OpenEquivariance is CUDA+HIP-only (checked 2026-08-20 — no SYCL backend, no
open issue asking for one). So eSEN and MACE must run e3nn's native pure-torch
path. If that does forward **and backward** on XPU, everything above it will
run. If not, nothing will — and you know in seconds rather than after
downloading a gated checkpoint. The printed ms/iter is the number to weigh
against "put MLIP inference on NVIDIA instead".

**Every check runs backward, not just forward.** A missing XPU autograd kernel
gives a model that imports, loads, returns an energy, and cannot do MD.

Exit codes: `0` usable · `1` device failure · `2` environment incomplete
(**not** an XPU verdict — nothing reached the device).

---

## 3. The two MLIP routes differ on Aurora. This is settled.

Determined by reading upstream source, 2026-08-20. No allocation needed.

### Route B — CP-MACE: runs on XPU

`mace/tools/torch_tools.py::init_device` has an explicit
`elif device_str == "xpu"` branch. **This is the Aurora-native path**, and it
is why the co-located acquisition loop (§5) survives.

Caveat: unlike its cuda and mps branches it does **not** assert availability —
it returns `torch.device("xpu")` even where no XPU exists. Check
`torch.xpu.is_available()` yourself, or a silent CPU fall-through reads as
"working, just oddly slow".

```bash
git clone https://github.com/yuanyue-liu-group/CP-MACE   # pin a commit
cd CP-MACE && pip install --no-deps ./mace
export ELECTROCHEM_CP_MACE_REPO=$PWD
```

**Licensing:** that repo ships **no LICENSE**, so it is all-rights-reserved.
Clone and run locally: fine. Vendoring its code or redistributing its
`.model` files: not. `mlip/cp_mace_simulation.py` live-imports the `NoseHoover`
integrator from your checkout for exactly this reason — keep it that way.
Upstream is also unmaintained (last push 2025-09-16), so pin a commit.

### Route A — fairchem: **ported to XPU, verified on Sunspot** `[V-9]`

> **Status changed 2026-08-20.** A full CUDA→XPU port now exists at
> **github.com/abagusetty/fairchem, branch `xpu-support`**, and it has been run
> on real hardware rather than reasoned about. Prefer it over the local patch
> script below; the patch script remains valid for a stock install you do not
> want to replace.
>
> The port adds `fairchem/core/common/device_utils.py` — a device abstraction
> built on torch's own generic APIs (`get_device_module`, `torch.accelerator`,
> `torch.amp.autocast`) — and routes every CUDA call site through it, rather
> than widening the assert alone. Behaviour on NVIDIA is unchanged.
>
> Things the 6-edit patch does **not** cover, found while porting `[C]`:
>
> * `layer_norm.py`, `normalizer.py`, `element_references.py` carry
>   `@torch.autocast("cuda", enabled=False)` + `("cpu", ...)` decorator pairs
>   with no `xpu` sibling, so autocast would **not** be disabled on XPU where
>   the author intended it to be.
> * `graph_parallel_a2a.py` selects the native `all_to_all` path with
>   `backend == "nccl"`, silently demoting oneCCL to pairwise send/recv.
> * `distutils.py` logs `CUDA_VISIBLE_DEVICES` on a machine masked by
>   `ZE_AFFINITY_MASK` / `ONEAPI_DEVICE_SELECTOR` — a confident "None" hiding a
>   real misconfiguration.
> * `rotation_cuda_graph.py` graph capture — `torch.xpu` does expose
>   `make_graphed_callables` and `Stream`, so it ports cleanly.
>
> **Collectives use `xccl`, and `xccl` *is* oneCCL** — not a separate library.
> Verified on Sunspot: `libtorch_xpu.so` links `libccl.so.1`/`libccl.so.2` from
> `/opt/aurora/.../oneapi/ccl/latest/lib` and exports `onecclAllReduce`,
> `onecclAllToAll`, `onecclAllGather`. `[V-9]` The *legacy* out-of-tree name is
> `"ccl"` (via `oneccl_bindings_for_pytorch`), which is **absent** from
> `frameworks/2026.1.0` — requesting it fails with "Backend not registered".
> PyTorch's own `Backend.default_device_backend_map` maps `xpu -> xccl`. NCCL is
> unavailable in this build (`is_nccl_available() == False`).
>
> UMA-S's fused Triton path stays **opt-in** off CUDA
> (`FAIRCHEM_ENABLE_TRITON_XPU=1`): those kernels are autotuned for NVIDIA
> occupancy, so compiling on XPU implies neither correctness nor speed.
>
> Verified on a Sunspot compute node (Intel Data Center GPU Max 1550, torch
> `2.13.0a0+gitcf30153`, 12 tiles), on a Cu bulk cell: `[V-9]`
>
> * forward — energy and forces on `xpu:0`, finite;
> * **backward** — conservative forces `-dE/dx` from on-device autograd, and 79
>   parameter-gradient tensors all finite and non-zero. This is the check that
>   separates "loads and returns an energy" from "can do MD";
> * **XPU and CPU energies agree to 1e-3** on identical weights and input;
> * fairchem's own suite: 417 passed / 50 skipped in the touched areas, plus 27
>   new device tests. The 15 graph-parallel failures were **pre-existing** —
>   the tests hard-code `PGConfig(backend="nccl")`, and they fail identically on
>   stock upstream on this machine. The port makes them follow the device type.

### The original blocker, for reference — fairchem does NOT run on XPU as shipped

`src/fairchem/core/units/mlip_unit/predict.py`:

```python
assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"
self.device = get_device_for_local_rank() if device == "cuda" else "cpu"
backend    = "gloo" if device == "cpu" else "nccl"
```

A runtime assert, not merely the `Literal["cuda","cpu"]` hint. Note also that
anything not `"cuda"` silently resolves to CPU, so bypassing the assert alone
would give a slow CPU run that looks like it worked.

The block is shallow though — no `.cu`/`.cpp`/`.cuh` anywhere in the tree, pure
Python over torch:

```bash
python scripts/patch_fairchem_xpu.py --check-requested-device  # verify first
python scripts/patch_fairchem_xpu.py --dry-run                 # read the diff
python scripts/patch_fairchem_xpu.py                           # apply
python scripts/patch_fairchem_xpu.py --revert                  # undo
python scripts/patch_fairchem_xpu.py --diff > fairchem-xpu.patch
```

6 exact edits across 2 files: the assert, device resolution, the XPU branch in
`assign_device_for_local_rank` and `get_device_for_local_rank`, oneCCL backend,
GPU slot reservation. Every target is verified before anything is written — if
upstream moved, it names what it could not find and writes **nothing**. A
half-applied patch is worse than none.

`--ccl-backend xccl` (default) is oneCCL native in torch; `ccl` selects the
external `torch-ccl` bindings. That line is only reached under Ray-distributed
multi-worker use, so single-device inference never touches it.

`--enable-triton-xpu` is opt-in and experimental (§6).

`mlip/esen_oc25.py` reads the installed source, so it starts accepting
`device="xpu"` the moment you patch, and blocks again if a pip upgrade reverts
you. `ELECTROCHEM_FAIRCHEM_XPU=1` overrides.

**If the patch works, upstream it.** An XPU path in fairchem is a visible
contribution to the FAIR Chemistry ecosystem, which `PROPOSAL.md` §5 argues is
the currency of the collaboration.

---

## 4. Gate status

| Gate | State | Action |
|---|---|---|
| **G1** OC25 checkpoints | **OPEN — the HF account is not on the approved list.** `facebook/OC25` and `facebook/UMA` are both `gated=manual`. Tested 2026-08-20 with a token for account `quark58`: authentication succeeds and repo *metadata* reads fine, but **every** file 403s "not in the authorized list" — including a 1 KB `config.json`. Two traps recorded so a future session does not re-debug them: (a) `model_info()` succeeding does **not** imply file access, metadata is public on gated repos; (b) the token is **not** the problem here — its scopes show `canReadGatedRepos: true`, so a read-only fine-grained token is sufficient *once access is granted*. `[V-9]` | A human must request access at huggingface.co/facebook/OC25 and be approved by Meta (manual review: legal name, DOB, organisation, FAIR Chemistry License). Then stage `checkpoints/esen_sm_conserve.pt` (51 MB, this is the Route-A model) and/or `checkpoints/esen_md_direct.pt` (406 MB). Sunspot compute nodes **do** have outbound network via `http_proxy=proxy.alcf.anl.gov:3128`, so staging can happen in-job. `[V-9]` |
| **G2** `JDFTXOutfile` attrs | CLOSED — `.e .forces .mu .converged .etype .is_gc` confirmed | — |
| **G3** JDFTx GPU on Aurora | CLOSED — internal SYCL port | stage pseudopotentials; confirm `pcm-variant CANDLE` for your build |
| **G4** MLIP on XPU | **CLOSED for Route A — fairchem ported and verified on hardware.** `check_aurora_env.py` passes all four checks on a Sunspot compute node; a full CUDA→XPU port of fairchem runs eSCN-MD fwd+bwd on PVC. `[V-9]` | Route B (CP-MACE) still unrun — it was only ever a source-reading verdict |
| **G5** Agent LLM | restructured — local vLLM-XPU, no Globus, no network | verify the tool-call parser (§7) |
| **G8** CP-MACE licence | open | email the authors for an explicit grant |
| **G10** scoop risk | live | the anchor authors know this gap; move on Phase 0 |

---

## 5. Why co-location matters

JDFTx runs GPU-native on Aurora via the SYCL port, and CP-MACE runs on XPU. So
**both halves of the acquisition loop fit in one allocation**:

```
PBS job (N nodes)
└── BundledLauncher slices $PBS_NODEFILE into disjoint host groups
    ├── group A ──> constant-potential MD  (committee: sigma_F, sigma_mu)
    ├── group B ──> constant-potential MD
    ├── group C ──> JDFTx GC-DFT audit of flagged frames
    └── group D ──> JDFTx GC-DFT audit
            └─> harvest -> recalibrate tau_mu -> next round
```

Split across machines, each turn costs a data stage plus a queue wait — hours —
and the campaign degenerates into a batch sweep with extra steps. Co-located, a
turn is one MD window plus one audit. That is the difference between an
adaptive instrument and a scripted parameter scan, and it is a claim the
manuscript can make concretely.

`BundledLauncher.run_all` throttles to `capacity = len(hosts) // nodes_per_task`.
Verified: 10 tasks at capacity 4 over an 8-host nodefile ran in 3 waves with
disjoint host groups and no oversubscription.

---

## 6. `triton-xpu 3.7.2` — two consequences

**Near term:** fairchem's UMA-S GPU execution mode dispatches Triton kernels
behind a `device.type == "cuda"` guard. With triton-xpu present that path is a
candidate rather than permanently off — `patch_fairchem_xpu.py
--enable-triton-xpu`. Opt-in and experimental on purpose: Triton portability is
a design goal, not a guarantee for a given kernel. CUDA-authored kernels can use
backend-specific intrinsics, and `num_warps`/`num_stages` autotuning configs are
tuned for NVIDIA occupancy, so a kernel can compile on XPU and still be slower
than the portable path. Benchmark against the default; `--revert` if it
regresses.

**Longer term, and more interesting:** the Clebsch-Gordan tensor product has no
Intel-GPU accelerator at all. cuEquivariance and OpenEquivariance are both JIT
kernel generators, and both stop before Intel. Triton is a JIT kernel generator
that works on XPU. **A Triton-XPU CG kernel is feasible and nobody has written
one.** Same shape as NeuralPLexer3's Flash-TriangularAttention (5× peak-memory
reduction, 50% faster forward). Check 4 of the probe tells you whether this is
on the table.

---

## 7. Agent LLM: local vLLM-XPU

Default backend is a **local** vLLM server on the allocation, not the remote
ALCF endpoint. `ELECTROCHEM_LLM_BACKEND=alcf` reverts.

Three failure modes this removes, all of which bite hardest inside a batch job:
compute nodes have no outbound network; `inference_auth_token.py authenticate`
is an interactive browser Globus flow no batch job can perform, with tokens
that expire mid-campaign; and a shared endpoint's models go hot/cold behind
other users. Same shape as Argonne's own Aurora multi-agent work
(arXiv:2604.07681).

Cost, to budget honestly: tiles serving the LLM are tiles not doing science.

**Tool calling is not on by default and this is the trap.** vLLM needs BOTH
`--enable-auto-tool-choice` AND a `--tool-call-parser` matching the model's
chat template. With the wrong parser the server starts, answers chat, and
returns tool invocations as **prose** — the agents produce plausible transcripts
and execute nothing. `VLLMServerConfig` refuses to build a command when no
parser is known, and `VLLMServer.verify_tool_calling()` proves a structured call
comes back before a campaign starts. Use it.

Stage weights locally first; `HF_HUB_OFFLINE=1` is forced so an un-staged model
fails fast instead of hanging on a socket timeout.

---

## 8. Suggested order of work

1. **`check_aurora_env.py`** on a compute node. 5 minutes, settles G4.
2. **Request OC25 checkpoint access** (G1) — it has human latency, start it now.
3. **Email CP-MACE authors** re: licence (G8) — same reason.
4. **CP-MACE end-to-end on their pretrained Au model**, unmodified, to validate
   the toolchain before investing in training.
5. **JDFTx**: one Cu slab grand-canonical single point. Confirm `.mu`, `.etype`
   and `nElectrons` all parse. Then calibrate `target-mu` against the Cu(100)
   work function — `run_campaign calibrate`.
6. **Strong-scaling run**, 200/400/800-atom cells. This replaces the guessed
   `atoms_per_rank=8` and the ×8 GPU multiplier in
   `hpc.aurora.nodes_for_jdftx`, which are the least-grounded numbers in the
   whole stack.
7. **Co-location rehearsal**: `BundledLauncher` running MD and a DFT audit
   concurrently in one allocation.
8. **Phase 1 reproduction** — the credibility floor. Hit `RESEARCH_PLAN.md` §3.1
   (Cu(100) 0.64 eV, Cu(310) 0.57/0.49 eV, ΔG 0.375/0.088 eV) or stop and debug.

Rehearse any campaign with `--dry-run` first: it writes every PBS script and
JDFTx input and submits nothing. A wrong tag replicated across 500 runs is
expensive.

---

## 9. Known-open code items

- `Campaign.md_runner` is deliberately `None`. Wire
  `md.cp_md_driver.run_constant_potential_md` into it once trained committee
  members exist. A stub that manufactured `MDResult`s would corrupt the
  A1-vs-B2 comparison, which is the experiment.
- Validation agent: IF-valid checks are real functions; the
  electrochemistry-specific guardrails (potential referencing, finite-size
  effects, ensemble choice) are still prompt-level.
- `propose_bo_candidates` is a distance heuristic. `scikit-learn` is in the
  module, so swap in a real `GaussianProcessRegressor` once calibrated data
  exists.
- `AseAtomsAdaptor.get_structure()` drops `atoms.info`, so
  `solvent_z_range` / `min_centre_separation` are lost crossing into pymatgen.
  Nothing depends on them yet. Unverified — ASE/pymatgen were not installed
  where this was written.
- PLUMED COLVAR bias units: confirm eV vs kJ/mol on the first real run.
  `data.harvest.read_colvar(bias_units=...)` is explicit rather than guessed
  because a silent 96.485× error rescales every free energy while leaving the
  barriers superficially plausible.

---

## 10. Document map

| File | Contents |
|---|---|
| `PROPOSAL.md` | scientific case — thesis, the gap, verified novelty landscape, the Ulissi/FAIR argument, target venues |
| `RESEARCH_PLAN.md` | architecture, phases, gates, falsifiable metrics, full source ledger |
| `ROADMAP.md` | module-by-module implementation status |
| `AURORA_HANDOFF.md` | this file |
| `archive/session-transcript-2026-08.md` | raw design conversation; provenance only, **not** a specification |

Provenance tags throughout: `[V-n]` verified against a live source ·
`[C]` verified by reading code · `[P]` project capability asserted by the team ·
`[A]` assumption · `[?]` unverified, must not enter a manuscript.
