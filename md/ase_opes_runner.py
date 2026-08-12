"""
ASE-driven MD + PLUMED/OPES runner, mirroring the actual simulation engine
used in Sahoo et al. (arXiv:2509.17862, Methods 5.4): PLUMED interfaced
with ASE's molecular dynamics engine.
"""

from dataclasses import dataclass
from pathlib import Path

from md.opes_runner import OPESConfig, write_plumed_input


@dataclass
class MDRunConfig:
    timestep_fs: float = 0.5
    temperature_k: float = 300.0
    n_steps: int = 15_000_000
    friction_per_fs: float = 0.002
    trajectory_path: str = "traj.traj"
    log_path: str = "md.log"
    log_interval: int = 1000


def run_md_with_opes(atoms, opes_config: OPESConfig, md_config: MDRunConfig,
                      work_dir: str = "."):
    try:
        from ase import units
        from ase.calculators.plumed import Plumed
        from ase.md.langevin import Langevin
        from ase.io.trajectory import Trajectory
    except ImportError as exc:
        raise ImportError(
            "ASE (with PLUMED bindings) is required for run_md_with_opes. "
            "Install ase and ensure PLUMED was built with "
            "--enable-modules=opes and Python/ASE support."
        ) from exc

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plumed_input_path = work_dir / "plumed.dat"
    write_plumed_input(opes_config, str(plumed_input_path))

    base_calc = atoms.calc
    if base_calc is None:
        raise ValueError("atoms.calc must be set to an MLIP calculator before biasing.")

    with open(plumed_input_path) as fh:
        plumed_lines = fh.read().splitlines()

    biased_calc = Plumed(
        calc=base_calc,
        input=plumed_lines,
        timestep=md_config.timestep_fs * units.fs,
        atoms=atoms,
        kT=md_config.temperature_k * units.kB,
    )
    atoms.calc = biased_calc

    dyn = Langevin(
        atoms,
        timestep=md_config.timestep_fs * units.fs,
        temperature_K=md_config.temperature_k,
        friction=md_config.friction_per_fs / units.fs,
    )

    traj = Trajectory(str(work_dir / md_config.trajectory_path), "w", atoms)
    dyn.attach(traj.write, interval=md_config.log_interval)

    dyn.run(md_config.n_steps)
    traj.close()
    return {
        "plumed_input": str(plumed_input_path),
        "trajectory": str(work_dir / md_config.trajectory_path),
        "colvar": str(Path.cwd() / "COLVAR"),
    }
