# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

Status: core geometry, I/O, PLUMED input generation, and free-energy
analysis are implemented and unit-tested (19/19 passing on pure numpy, no
ase/pymatgen/fairchem/JDFTx required for the tested subset).

- systems/water_geometry.py -- rigid TIP3P-like water geometry. Tested.
- systems/packing.py -- dependency-free (Packmol-free) rejection-sampling
  packer for solvent + ions. Tested.
- systems/io_utils.py -- extended-XYZ (CP-MACE format) and LAMMPS data file
  writers, no ase/pymatgen dependency. Tested.
- systems/cu_interface.py -- Cu(100)/Cu(310) slab + explicit water + cation
  interface builder (requires ASE; follows OC25 recipe, Methods 5.1.1-5.1.2).
- mlip/esen_oc25.py -- fairchem OC25 eSEN calculator loader, using the
  CURRENT fairchem-core (>=2.x) unified API:
  `pretrained_mlip.get_predict_unit(...)` + `FAIRChemCalculator(...)`
  (verified against facebook/OC25 model card and facebookresearch/fairchem
  README, 2026-08-12). A legacy fairchem-v1 `OCPCalculator` path is kept
  as an explicit opt-in for pinned-v1 environments only.
- mlip/cp_mace_wrapper.py -- CP-MACE dataset writer and mace_run_train wrapper.
- md/opes_runner.py -- PLUMED OPES input generator matching Methods 5.4. Tested.
- md/ase_opes_runner.py -- ASE Langevin MD + `ase.calculators.plumed.Plumed`
  bias driver (verified current API, requires ASE >=3.23.0 + py-plumed),
  the actual simulation engine described in the paper.
- md/lammps_runner.py -- LAMMPS data/input writer for later ML-IAP integration.
- analysis/free_energy.py -- OPES reweighting, barrier/reaction-energy
  extraction, block-uncertainty convergence check, water-orientation
  analysis. Tested.

### Access / license gate (IMPORTANT, corrected 2026-08-12)

The OC25 **dataset** is CC-BY-4.0 (open). The OC25 **model checkpoints**
(esen-sm-conserving-all-oc25, esen-md-direct-all-oc25) are distributed
under Meta's "FAIR Chemistry License" via a GATED Hugging Face repo
(huggingface.co/facebook/OC25): you must request access (legal name, DOB,
organization) and accept an Acceptable Use Policy before
`pretrained_mlip.get_predict_unit(...)` will succeed. Budget time for this
approval before your first GH200 run.

Remaining Phase 1 work (requires ase/pymatgen/fairchem + GPU + gated HF
access, must run on GH200/A100):

- End-to-end run: build an 8x8 Cu(100)/water/Cs+ cell, attach eSEN-OC25 via
  the current FAIRChemCalculator API, run
  md.ase_opes_runner.run_md_with_opes for the full 7.5 ns, and check the
  extracted barrier/reaction energy against the paper's reported values
  (Cu(100): ~0.64 eV barrier, ~0.375 eV reaction energy at PZC).
- Validate estimate_surface_charge_density against the paper's reported
  sigma values for matching cation counts.

## Phase 2 -- CP-DFT audit module -- INTERFACE STARTED

- cp_dft/jdftx_interface.py -- wraps the OFFICIAL JDFTx ASE calculator
  (`from JDFTx import JDFTx`, distributed with JDFTx source under
  jdftx/scripts/ase; verified against jdftx.org/ASE.html, 2026-08-12).
  Grand-canonical / constant-potential control uses JDFTx's native
  `target-mu` command, passed through the calculator's `commands=` dict
  (the ASE wrapper has no dedicated constant-potential argument of its
  own -- confirm sign convention/reference level against JDFTx docs before
  reporting calibrated potentials). Command assembly logic
  (`build_jdftx_commands`) is unit-tested (4/4 passing) without requiring
  a JDFTx build.
- Remaining: build JDFTx from source on target HPC, set PYTHONPATH/
  JDFTx/JDFTx_pseudo env vars, calibrate target-mu against a known
  work function for a reference Cu slab, then sample initial/TS/final
  states from Phase 1 trajectories and label with true electrode
  potential, energies, forces, and workfunction.
- Compare constant-charge (Phase 1) vs constant-potential (Phase 2) free
  energies.

## Phase 3 -- CP-MACE for Cu interfaces

- Use mlip/cp_mace_wrapper.py to build CP-MACE-format datasets from OC25 +
  CP-DFT (JDFTx target-mu) labels.
- Train FermiMACE on Cu(100)/Cu(310)/water/cation systems.
- Validate against CP-DFT and eSEN-OC25 baselines.
- Run constant-potential MD (CP-MACE simulation/ scripts, or via LAMMPS ML-IAP).

## Phase 4 -- Agentic orchestration

- Implement agents/planner.py and specialized agents to chain Phases 1-3
  from a single YAML/CLI spec.
- Wire analysis.free_energy.is_converged into agents/md_opes_agent.py for
  convergence-driven OPES stopping.
- Add HPC scheduler abstraction (GH200 node / A100 cluster).
