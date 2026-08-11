# Roadmap

## Phase 1 — Constant-charge MVP (eSEN-OC25 + LAMMPS + OPES)

- Build Cu(100) / Cu(310) slabs + explicit water + Cs+/K+/Li+ using `systems/`.
- Load eSEN-OC25 checkpoint via `fairchem`.
- Run MD + OPES (CO-CO distance CV) via `md/` runners.
- Reproduce OC25 paper barriers/reaction energies as a validation baseline.

## Phase 2 — CP-DFT audit module

- Implement `cp_dft/` wrapper for a constant-potential DFT method (GPAW inner-potential,
  JDFTx grand-canonical, or CP2K FCP algorithm).
- Sample initial/TS/final states from Phase 1 trajectories; label with true electrode
  potential, energies, forces, and workfunction.
- Compare constant-charge vs constant-potential free energies.

## Phase 3 — CP-MACE for Cu interfaces

- Build CP-MACE-format xyz datasets (electron/potential tags) from OC25 + CP-DFT labels.
- Train `FermiMACE` on Cu(100)/Cu(310)/water/cation systems.
- Validate against CP-DFT and eSEN-OC25 baselines.
- Run constant-potential MD (CP-MACE `simulation/` scripts, or via LAMMPS ML-IAP).

## Phase 4 — Agentic orchestration

- Implement `agents/planner.py` and specialized agents to chain Phases 1-3 from a
  single YAML/CLI spec.
- Add convergence-driven OPES hyperparameter control and early stopping.
- Add HPC scheduler abstraction (GH200 node / A100 cluster).
