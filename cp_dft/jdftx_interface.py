"""
JDFTx interface for constant-potential / grand-canonical DFT, split across
two tools deliberately chosen for what each is actually good at (verified
2026-08-12):

1. pymatgen.io.jdftx (JDFTXInfile / JDFTXOutfile) -- PRIMARY, recommended
   path for single-point CP-DFT labeling (Phase 2's main need: generating
   {structure, electron, potential, energy, forces} labels for CP-MACE
   training data). This is the module atomate2 itself uses for JDFTx I/O
   ("JDFTx output and input files are parsed with code in
   pymatgen.io.jdftx" -- Ganose et al., atomate2 paper). It provides:
     - JDFTXInfile.from_structure(): typed, validated input construction
       directly from a pymatgen Structure.
     - JDFTXOutfile / JDFTXOutfileSlice: structured output parsing that
       exposes `.mu` (Fermi energy / electron chemical potential after
       the run), `.is_gc` (grand-canonical flag), `.forces`, `.structure`,
       and full electronic-minimization history as typed attributes --
       no manual log-file regexing required.
   No ASE dependency; JDFTx is run via subprocess with a plain input file.

2. The official JDFTx ASE calculator (`from JDFTx import JDFTx`, shipped
   under jdftx/scripts/ase, see jdftx.org/ASE.html) -- SECONDARY, used
   only when a step-wise ASE Calculator object is actually required, e.g.
   coupling JDFTx to `ase.calculators.plumed.Plumed` for constant-
   potential enhanced-sampling MD. pymatgen has no MD driver, so this
   path is unavoidable for that specific use case, but it should NOT be
   used for plain single-point labeling where pymatgen is more robust.

Grand-canonical / constant-potential control in JDFTx itself is via the
native `target-mu` command (electron chemical potential) plus an implicit
solvation model (e.g. `fluid LinearPCM`). Both paths below set this the
same way; confirm the sign convention and reference level against the
JDFTx documentation for your version/pseudopotential set before
interpreting target-mu as a calibrated electrode potential -- a
calibration step (matching target-mu to a known work function for a
reference Cu slab) is recommended before reporting potential-dependent
free energies.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class JDFTxRunConfig:
    executable: str = "jdftx"  # or full path; can include an MPI launcher
                                 # prefix, e.g. "mpirun -n 8 jdftx"
    pseudo_set: Optional[str] = "GBRV-pbe"  # or "SG15", "GBRV-lda", "GBRV-pbesol"
    pseudo_dir: Optional[str] = None
    fluid_model: Optional[str] = "LinearPCM"
    target_mu_hartree: Optional[float] = None  # None => neutral, constant-charge
    elec_cutoff: str = "20 100"
    extra_tags: Dict[str, str] = field(default_factory=dict)
    work_dir: str = "."
    input_filename: str = "in"
    output_filename: str = "out"


def _base_tags(config: JDFTxRunConfig) -> Dict[str, str]:
    """Tags shared by both the pymatgen and ASE paths."""
    tags: Dict[str, str] = {"elec-cutoff": config.elec_cutoff}
    if config.fluid_model:
        tags["fluid"] = config.fluid_model
    if config.target_mu_hartree is not None:
        tags["target-mu"] = f"{config.target_mu_hartree:.6f}"
    tags.update(config.extra_tags)
    return tags


# ---------------------------------------------------------------------------
# Primary path: pymatgen.io.jdftx (recommended for single-point CP-DFT
# labeling; no ASE dependency).
# ---------------------------------------------------------------------------

def build_jdftx_infile(structure, config: JDFTxRunConfig):
    """Build a pymatgen JDFTXInfile for `structure` with the tags implied
    by `config` (elec-cutoff, fluid, target-mu, extra_tags), using
    JDFTXInfile.from_structure() plus tag merging.

    Requires pymatgen>=2025.4 (JDFTx I/O support). Returns a JDFTXInfile
    object; call `.write_file(path)` to write it to disk.
    """
    try:
        from pymatgen.io.jdftx.inputs import JDFTXInfile
    except ImportError as exc:
        raise ImportError(
            "pymatgen.io.jdftx requires a reasonably recent pymatgen "
            "(>=2025.4, when JDFTx I/O support was added). "
            "Install/upgrade via `pip install -U pymatgen`."
        ) from exc

    infile = JDFTXInfile.from_structure(structure)
    tags = _base_tags(config)
    for tag, value in tags.items():
        infile.append_tag(tag, value) if tag in infile else infile.update({tag: value})
    return infile


def run_jdftx_single_point(structure, config: JDFTxRunConfig):
    """Write a JDFTXInfile for `structure`, run JDFTx via subprocess, and
    parse the result with pymatgen's JDFTXOutfile. Returns a dict with
    energy (eV), forces, mu (Fermi energy / electron chemical potential,
    eV), and is_gc (whether the run was grand-canonical).

    This is the recommended path for Phase 2 CP-DFT labeling: no ASE
    dependency, structured output parsing via pymatgen.
    """
    try:
        from pymatgen.io.jdftx.outputs import JDFTXOutfile
    except ImportError as exc:
        raise ImportError(
            "pymatgen.io.jdftx requires a reasonably recent pymatgen "
            "(>=2025.4). Install/upgrade via `pip install -U pymatgen`."
        ) from exc

    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    infile = build_jdftx_infile(structure, config)
    input_path = work_dir / config.input_filename
    output_path = work_dir / config.output_filename
    infile.write_file(str(input_path))

    cmd = config.executable.split() + ["-i", str(input_path), "-o", str(output_path)]
    subprocess.run(cmd, check=True, cwd=str(work_dir))

    outfile = JDFTXOutfile.from_file(str(output_path))
    return {
        "energy_ev": outfile.e,
        "forces": outfile.forces,
        "mu_ev": outfile.mu,
        "is_gc": outfile.is_gc,
        "structure": outfile.structure,
        "converged": outfile.converged,
    }


# ---------------------------------------------------------------------------
# Secondary path: official ASE JDFTx calculator, ONLY for coupling to a
# step-wise ASE dynamics loop (e.g. PLUMED-biased MD). Do not use this for
# plain single-point labeling -- prefer run_jdftx_single_point above.
# ---------------------------------------------------------------------------

def load_ase_jdftx_calculator(config: JDFTxRunConfig):
    """Return the official ASE JDFTx calculator, for use only when a
    step-wise ASE Calculator object is required (e.g.
    ase.calculators.plumed.Plumed wrapping JDFTx for constant-potential
    enhanced-sampling MD). Requires jdftx/scripts/ase on PYTHONPATH (see
    https://jdftx.org/ASE.html).
    """
    try:
        from JDFTx import JDFTx
    except ImportError as exc:
        raise ImportError(
            "Could not import the JDFTx ASE interface. Ensure JDFTx is "
            "built from source and jdftx/scripts/ase is on PYTHONPATH: "
            "export PYTHONPATH=/path-to-jdftx/scripts/ase:$PYTHONPATH"
        ) from exc

    tags = _base_tags(config)
    kwargs = {"commands": tags, "executable": config.executable}
    if config.pseudo_set is not None:
        kwargs["pseudoSet"] = config.pseudo_set
    if config.pseudo_dir is not None:
        kwargs["pseudoDir"] = config.pseudo_dir
    return JDFTx(**kwargs)
