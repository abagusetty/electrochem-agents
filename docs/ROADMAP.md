# Roadmap

## Phase 1 -- Constant-charge MVP (eSEN-OC25 + ASE/PLUMED OPES) -- IMPLEMENTED

Status: core geometry, I/O, PLUMED input generation, and free-energy
analysis are implemented and unit-tested (23/23 passing on pure numpy, no
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
  README, 2026-08-12).
- mlip/cp_mace_wrapper.py -- CP-MACE dataset writer and mace_run_train wrapper.
- md/opes_runner.py -- PLUMED OPES input generator matching Methods 5.4. Tested.
- md/ase_opes_runner.py -- ASE Langevin MD + `ase.calculators.plumed.Plumed`
  bias driver (verified current API, requires ASE >=3.23.0 + py-plumed).
- md/lammps_runner.py -- LAMMPS data/input writer for later ML-IAP integration.
- analysis/free_energy.py -- OPES reweighting, barrier/reaction-energy
  extraction, block-uncertainty convergence check, water-orientation
  analysis. Tested.

### Access / license gate (corrected 2026-08-12)

OC25 **dataset** = CC-BY-4.0 (open). OC25 model **checkpoints** are gated
under Meta's "FAIR Chemistry License" (huggingface.co/facebook/OC25);
request access + `huggingface-cli login` before your first GH200 run.

## Phase 2 -- CP-DFT audit module -- REVISED 2026-08-12: pymatgen-first design

cp_dft/jdftx_interface.py now splits across two deliberately chosen tools:

1. **pymatgen.io.jdftx (PRIMARY)** -- `JDFTXInfile.from_structure()` for
   typed, validated input construction and `JDFTXOutfile` for structured
   output parsing (`.mu` = Fermi energy/electron chemical potential,
   `.is_gc` = grand-canonical flag, `.forces`, `.structure`, full
   electronic-minimization history). This is the module atomate2 itself
   uses for JDFTx I/O (Ganose et al., atomate2 paper, 2025) and is
   actively maintained (added to pymatgen ~Oct 2024, ongoing PRs through
   2025). No ASE dependency. `run_jdftx_single_point()` writes the input,
   runs JDFTx via subprocess, and parses the output -- this is the
   recommended path for Phase 2's actual need: single-point CP-DFT labels
   {structure, target-mu, energy, forces} for CP-MACE training data.

2. **Official ASE JDFTx calculator (SECONDARY)** -- `load_ase_jdftx_calculator()`,
   kept ONLY for the case where a step-wise ASE Calculator object is
   required, i.e. coupling JDFTx to `ase.calculators.plumed.Plumed` for
   constant-potential enhanced-sampling MD. pymatgen has no MD driver, so
   this path is unavoidable for that specific use case but should not be
   used for plain single-point labeling.

Both paths set grand-canonical / constant-potential control via JDFTx's
native `target-mu` command (electron chemical potential); tag assembly
logic (`_base_tags`) is unit-tested (4/4 passing) without requiring
pymatgen, JDFTx, or ASE to be installed. Confirm target-mu's sign
convention/reference level against JDFTx docs and calibrate against a
known work function before reporting calibrated electrode potentials.

Remaining Phase 2 work (requires pymatgen>=2025.4 + a JDFTx build + GPU,
must run on GH200/A100 or a JDFTx-capable node):

- Build JDFTx from source; confirm `pymatgen.io.jdftx` version compatibility.
- Calibrate target-mu against a known work function for a reference Cu slab.
- Sample initial/TS/final states from Phase 1 trajectories; label with
  `run_jdftx_single_point()`.
- Compare constant-charge (Phase 1) vs constant-potential (Phase 2) free
  energies.

## Phase 3 -- CP-MACE for Cu interfaces

- Use mlip/cp_mace_wrapper.py to build CP-MACE-format datasets from OC25 +
  CP-DFT (pymatgen/JDFTx target-mu) labels.
- Train FermiMACE on Cu(100)/Cu(310)/water/cation systems.
- Validate against CP-DFT and eSEN-OC25 baselines.
- Run constant-potential MD (CP-MACE simulation/ scripts, or via LAMMPS ML-IAP).

## Phase 4 -- Agentic orchestration

- Implement agents/planner.py and specialized agents to chain Phases 1-3
  from a single YAML/CLI spec.
- Wire analysis.free_energy.is_converged into agents/md_opes_agent.py for
  convergence-driven OPES stopping.
- Add HPC scheduler abstraction (GH200 node / A100 cluster).
