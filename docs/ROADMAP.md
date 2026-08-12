# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

Status: core geometry, I/O, PLUMED input generation, and free-energy
analysis are implemented and unit-tested (15/15 passing on pure numpy, no
ase/pymatgen/fairchem required for the tested subset).

- systems/water_geometry.py -- rigid TIP3P-like water geometry. Tested.
- systems/packing.py -- dependency-free (Packmol-free) rejection-sampling
  packer for solvent + ions. Tested.
- systems/io_utils.py -- extended-XYZ (CP-MACE format) and LAMMPS data file
  writers, no ase/pymatgen dependency. Tested.
- systems/cu_interface.py -- Cu(100)/Cu(310) slab + explicit water + cation
  interface builder (requires ASE; follows OC25 recipe, Methods 5.1.1-5.1.2).
- mlip/esen_oc25.py -- fairchem OC25 eSEN calculator loader.
- mlip/cp_mace_wrapper.py -- CP-MACE dataset writer and mace_run_train wrapper.
- md/opes_runner.py -- PLUMED OPES input generator matching Methods 5.4. Tested.
- md/ase_opes_runner.py -- ASE Langevin MD + PLUMED bias driver (the actual
  simulation engine described in the paper).
- md/lammps_runner.py -- LAMMPS data/input writer for later ML-IAP integration.
- analysis/free_energy.py -- OPES reweighting, barrier/reaction-energy
  extraction, block-uncertainty convergence check, water-orientation
  analysis. Tested.

Remaining Phase 1 work (requires ase/pymatgen/fairchem + GPU, must run on
GH200/A100):

- End-to-end run: build an 8x8 Cu(100)/water/Cs+ cell, attach eSEN-OC25,
  run md.ase_opes_runner.run_md_with_opes for the full 7.5 ns, and check
  the extracted barrier/reaction energy against the paper's reported
  values (Cu(100): ~0.64 eV barrier, ~0.375 eV reaction energy at PZC).
- Validate estimate_surface_charge_density against the paper's reported
  sigma values for matching cation counts.

## Phase 2 -- CP-DFT audit module

- Implement cp_dft/ wrapper for a constant-potential DFT method (GPAW
  inner-potential, JDFTx grand-canonical, or CP2K FCP algorithm).
- Sample initial/TS/final states from Phase 1 trajectories; label with
  true electrode potential, energies, forces, and workfunction.
- Compare constant-charge vs constant-potential free energies.

## Phase 3 -- CP-MACE for Cu interfaces

- Use mlip/cp_mace_wrapper.py to build CP-MACE-format datasets from OC25 +
  CP-DFT labels.
- Train FermiMACE on Cu(100)/Cu(310)/water/cation systems.
- Validate against CP-DFT and eSEN-OC25 baselines.
- Run constant-potential MD (CP-MACE simulation/ scripts, or via LAMMPS ML-IAP).

## Phase 4 -- Agentic orchestration

- Implement agents/planner.py and specialized agents to chain Phases 1-3
  from a single YAML/CLI spec.
- Wire analysis.free_energy.is_converged into agents/md_opes_agent.py for
  convergence-driven OPES stopping.
- Add HPC scheduler abstraction (GH200 node / A100 cluster).
