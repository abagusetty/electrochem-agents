"""
JDFTx ASE-calculator wrapper for constant-potential / grand-canonical DFT.

This wraps the OFFICIAL JDFTx ASE interface (distributed with the JDFTx
source under `jdftx/scripts/ase`, imported as `from JDFTx import JDFTx`;
see https://jdftx.org/ASE.html, verified 2026-08-12). That interface is a
thin force/energy calculator: JDFTx input-file commands are passed
through as a plain dict via the `commands=` argument, with no dedicated
"constant potential" parameter of its own.

Grand-canonical / constant-potential control in JDFTx is achieved via its
native `target-mu` command (electron chemical potential) together with an
implicit solvation model (e.g. `fluid LinearPCM` or `fluid SaLSA`) and
appropriate `elec-cutoff`/pseudopotential settings -- these are passed
straight through the `commands` dict below. Consult the JDFTx
documentation (jdftx.org) for the exact `target-mu` sign convention and
units (Hartree) relative to the vacuum level for your pseudopotential set.

Setup required before this module will work:
  1. Build JDFTx from source.
  2. Install ASE separately (JDFTx does not bundle it).
  3. `export PYTHONPATH=/path-to-jdftx/scripts/ase:$PYTHONPATH`
  4. `export JDFTx=/path/to/jdftx/executable`
  5. `export JDFTx_pseudo=/path/to/pseudopotential/directory` (or use a
     built-in set via `pseudo_set`, e.g. "GBRV-pbe", "SG15").
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class JDFTxConfig:
    executable: Optional[str] = None  # falls back to $JDFTx env var if None
    pseudo_set: Optional[str] = "GBRV-pbe"  # or "SG15", "GBRV-lda", "GBRV-pbesol"
    pseudo_dir: Optional[str] = None  # falls back to $JDFTx_pseudo if None
    fluid_model: Optional[str] = "LinearPCM"  # implicit solvent; None to disable
    target_mu_hartree: Optional[float] = None  # electron chemical potential
                                                 # target for grand-canonical /
                                                 # constant-potential runs;
                                                 # None => standard neutral,
                                                 # constant-charge calculation
    elec_cutoff: str = "20 100"  # wavefunction / density cutoffs (Hartree)
    extra_commands: Dict[str, str] = field(default_factory=dict)
    mpi_launcher: Optional[str] = None  # e.g. "mpirun -n 8" or "srun -n 8"


def build_jdftx_commands(config: JDFTxConfig) -> Dict[str, str]:
    """Assemble the JDFTx input-command dict from a JDFTxConfig.

    NOTE on target-mu: this is passed through verbatim to JDFTx as the
    `target-mu` command. Confirm the sign convention and reference level
    (absolute vs. vacuum-referenced) against the JDFTx documentation for
    your JDFTx version before interpreting results as calibrated
    electrode potentials; a calibration step (e.g. matching target-mu to
    a known work function for a reference slab) is recommended before
    reporting potential-dependent free energies.
    """
    commands: Dict[str, str] = {"elec-cutoff": config.elec_cutoff}
    if config.fluid_model:
        commands["fluid"] = config.fluid_model
    if config.target_mu_hartree is not None:
        commands["target-mu"] = f"{config.target_mu_hartree:.6f}"
    commands.update(config.extra_commands)
    return commands


def load_jdftx_calculator(config: JDFTxConfig):
    """Return an ASE-compatible JDFTx calculator configured per `config`.

    Raises ImportError with setup instructions if the JDFTx ASE interface
    (`from JDFTx import JDFTx`) is not importable, which typically means
    PYTHONPATH has not been set to jdftx/scripts/ase.
    """
    try:
        from JDFTx import JDFTx
    except ImportError as exc:
        raise ImportError(
            "Could not import the JDFTx ASE interface. Ensure JDFTx is "
            "built from source and that jdftx/scripts/ase is on "
            "PYTHONPATH (see https://jdftx.org/ASE.html): "
            "export PYTHONPATH=/path-to-jdftx/scripts/ase:$PYTHONPATH"
        ) from exc

    commands = build_jdftx_commands(config)

    executable = config.executable
    if config.mpi_launcher and executable:
        executable = f"{config.mpi_launcher} {executable}"

    kwargs = {"commands": commands}
    if executable is not None:
        kwargs["executable"] = executable
    if config.pseudo_set is not None:
        kwargs["pseudoSet"] = config.pseudo_set
    if config.pseudo_dir is not None:
        kwargs["pseudoDir"] = config.pseudo_dir

    return JDFTx(**kwargs)


def attach_calculator(atoms, config: JDFTxConfig):
    """Attach a JDFTx calculator to an ASE Atoms object in place."""
    atoms.calc = load_jdftx_calculator(config)
    return atoms


def single_point_energy_and_forces(atoms, config: JDFTxConfig):
    """Run a single-point JDFTx calculation and return (energy_eV, forces).

    Calls `calculator.clean()` afterwards to remove JDFTx run files from
    the temporary run directory, per the official interface's convention.
    """
    calc = load_jdftx_calculator(config)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    if hasattr(calc, "clean"):
        calc.clean()
    return energy, forces
