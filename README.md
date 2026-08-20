# electrochem-agents

Agentic scaffold for **constant-potential** electrochemical interface simulations —
JDFTx grand-canonical DFT, potential-aware MLIPs (eSEN-OC25 / CP-MACE FermiMACE), and
PLUMED-driven enhanced sampling, under LLM-agent control.

Extends [arXiv:2509.17862](https://arxiv.org/abs/2509.17862) — *Insights into CO
dimerization at electrified Cu interfaces from large-scale machine learning
simulations* (Sahoo, Maraschin, Varley, Levine, Ulissi, Zitnick, Takemura, Gauthier,
Govindarajan, Shuaibi).

> ⚠️ **Pre-execution.** No module here has ever run against a real simulation stack.
> Zero MD steps, zero DFT calculations, zero LLM calls. Everything is written and
> logic-tested only. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for exactly what is and
> is not verified.

---

## The idea in one paragraph

Two literatures each solve half the problem. OC25 gives explicit solvent at scale
(7.8M DFT points) but **constant charge**. BEAST DB gives grand-canonical
**constant potential** (20k+ calculations) but **implicit solvent**. The
explicit-solvent × constant-potential cell is empty, because it is expensive — which
is why the contribution here is an *acquisition policy* rather than a bigger
calculation. Constant-potential MLIP committees expose a signal that constant-charge
models structurally cannot: **σ_µ, disagreement about the electrode potential itself**.
This project makes σ_µ the trigger for spending grand-canonical DFT.

Because JDFTx is ported to SYCL and runs GPU-native on Aurora PVC, the MD and the
grand-canonical audits it triggers run **inside one allocation** — so the loop
closes in minutes rather than across queue cycles and machine boundaries. That is
what makes it an adaptive instrument instead of a batch sweep.

---

## Documentation

| Read this | For |
|---|---|
| [`docs/AURORA_HANDOFF.md`](docs/AURORA_HANDOFF.md) | **start here on Sunspot/Aurora** — env setup, the 5-minute probe, gate status, order of work |
| [`docs/PROPOSAL.md`](docs/PROPOSAL.md) | scientific case — thesis, the gap, what is actually novel, collaboration, venues |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | architecture, phases, gates, falsifiable metrics, source ledger |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | module-by-module implementation status |
| [`docs/archive/`](docs/archive/) | raw design conversation — provenance only, **not** a specification |

All claims across these documents carry provenance tags: `[V-n]` verified against a
live source, `[C]` verified by reading code, `[A]` assumption, `[?]` unverified.

---

## Layout

```
hpc/          paths          software locations; /lus/flare project layout
              aurora         node topology, SYCL env, mpiexec (PALS)
              pbs            PBS Pro scripts, qsub, qstat polling
              launcher       local | pbs | bundle (many jobs, one allocation)
data/         schema         StatePoint · CPDFTLabel · MDResult · CommitteeStats
              store          append-only JSONL, flock-safe, compactable
              harvest        the ONLY place that decides a run succeeded
              xyz            CP-MACE extended-XYZ (electron= / potential=)
acquisition/  policies       B0 grid · B1 random · B2 force-only ablation
              sigma_mu       A1 policy + threshold calibration vs realised error
              registry       frozen pre-registration; cost-to-reproduce scoring
cp_dft/       calibration    target-mu <-> U vs SHE; work-function reference
              jdftx_setup    CANDLE / target-mu inputs, potential sweeps
              jdftx_driver   Aurora submission + harvest
              jdftx_interface  low-level pymatgen/ASE paths
mlip/         committee      sigma_F + sigma_mu; implements get_mu()
              xpu            Intel GPU resolution; `doctor` backend report
              esen_oc25 · cp_mace_wrapper · cp_mace_simulation
md/           cp_md_driver   run(1) loop, safety thresholds, agent hooks
              ase_opes_runner · opes_runner · lammps_runner
analysis/     free_energy    reweighting, block convergence
              interface_validity   IF-valid physical-validity checklist
systems/      cu_interface · packing · water_geometry · io_utils
agents/       manager · planner · reasoning · system_messages · llm_backend
              {analysis,cp_dft,md_opes,mlip,system}_agent
workflows/    campaign       setup -> run -> harvest -> decide loop
              run_campaign   CLI (doctor/preregister/calibrate/run/compare)
              aurora_campaign.yaml
tests/        6 modules, pure numpy
```

## Running a campaign

```bash
python -m workflows.run_campaign doctor                                  # on a COMPUTE node
python -m workflows.run_campaign preregister --config workflows/aurora_campaign.yaml
python -m workflows.run_campaign calibrate   --config ... --facet 100
python -m workflows.run_campaign run         --config ... --policy A1_sigma_mu
python -m workflows.run_campaign compare     --config ...
```

`--dry-run` writes every PBS script and JDFTx input but submits nothing. Rehearse
that way first: a wrong tag replicated across 500 runs is expensive.

Run all four policy arms (`B0_grid`, `B1_random`, `B2_force_uncertainty`,
`A1_sigma_mu`). **A1 vs B2 is the experiment** — B2 is A1 with the σ_µ term removed,
so the gap between them is the only evidence that Fermi-level uncertainty adds
anything over ordinary force uncertainty.

---

## Access requirements

Three of these gate real execution and are **not** yet cleared.

| Requirement | Note |
|---|---|
| **OC25 checkpoints** | Gated Hugging Face repo under Meta's FAIR Chemistry License — legal name, DOB, organization, AUP acceptance. The OC25 *dataset* is CC-BY-4.0; the *checkpoints are not*. |
| **CP-MACE checkout** | [github.com/yuanyue-liu-group/CP-MACE](https://github.com/yuanyue-liu-group/CP-MACE). **Ships no LICENSE — all rights reserved.** Fine to clone and run locally; do **not** vendor its code or redistribute its `.model` files. `mlip/cp_mace_simulation.py` deliberately live-imports from your local checkout rather than copying anything. |
| **JDFTx** | Ported to SYCL, runs GPU-native on Aurora PVC. Note upstream JDFTx has **no** SYCL/oneAPI/HIP backend — this port is project-held, so point `ELECTROCHEM_JDFTX_GPU_BIN` at your build. Still needed: pseudopotential staging, confirming `pcm-variant CANDLE` for this build, and a strong-scaling run to replace the `atoms_per_rank` default. |
| **ALCF Inference Endpoints** | One interactive `python inference_auth_token.py authenticate` Globus login. Agents require **tool-calling** models — reasoning-only models are disqualified. |

## Install

```bash
pip install -r requirements.txt      # numpy pyyaml ase pymatgen ag2 requests
```

On Aurora, **`module load frameworks` first** — it supplies the oneAPI SDK and
an XPU-enabled PyTorch. `requirements.txt` deliberately omits `torch`: a PyPI
wheel is CUDA or CPU only and shadows the working build.

Separate envs for the two MLIP arms, because CP-MACE and upstream MACE both
provide the module name `mace`:

```bash
# Route A                       # Route B (from your CP-MACE checkout)
pip install fairchem-core       pip install ./mace
```

PLUMED coupling also needs `py-plumed` and `PLUMED_KERNEL` exported.

### Settle the Intel-GPU question before committing MLIP hours

```bash
python -m workflows.run_campaign doctor --probe --cp-mace-model <path>.model
```

Real forward **and backward** per stack on device. Exit 0 = at least one stack
usable; 1 = device failure; 2 = environment incomplete (not an XPU verdict).

## Tests

```bash
python -m pytest tests/       # pure numpy; no ASE/pymatgen/fairchem/PLUMED needed
```

These test logic, not physics. Passing tests do not mean the workflow runs.
