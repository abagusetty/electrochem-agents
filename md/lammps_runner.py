"""
LAMMPS MD runner for a later ML-IAP/Kokkos integration. For Phase 1, prefer
md.ase_opes_runner, which matches the paper's actual simulation engine.
"""

from pathlib import Path

from systems.io_utils import write_lammps_data

DEFAULT_MASSES = {"Cu": 63.546, "O": 15.999, "H": 1.008,
                   "Li": 6.94, "K": 39.098, "Cs": 132.905}

LAMMPS_INPUT_TEMPLATE = """\
units           metal
atom_style      full
boundary        p p f
read_data       {data_file}

# NOTE: replace `pair_style none` with an ML-IAP pair_style once the target
# MLIP has been exported to a LAMMPS-compatible format, e.g.:
#   pair_style      mliap unified {model_file}
#   pair_coeff      * *
pair_style      none

timestep        {timestep_fs_over_1000}
velocity        all create {temperature_k} {seed} dist gaussian

fix             thermostat all langevin {temperature_k} {temperature_k} 100.0 {seed}
fix             integrate all nve

thermo          {thermo_interval}
thermo_style    custom step temp pe etotal press

dump            traj all custom {dump_interval} {dump_file} id type x y z fx fy fz

run             {n_steps}
"""


def write_lammps_inputs(atoms_species, atoms_positions, cell, work_dir,
                         data_filename="system.data", input_filename="in.electrochem",
                         model_file="model.pt", temperature_k=300.0, timestep_fs=0.5,
                         n_steps=1_000_000, seed=12345, thermo_interval=1000,
                         dump_interval=1000, dump_file="traj.dump", masses=None):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    masses = masses or DEFAULT_MASSES

    data_path = work_dir / data_filename
    _, type_map = write_lammps_data(str(data_path), atoms_species, atoms_positions, cell, masses)

    input_text = LAMMPS_INPUT_TEMPLATE.format(
        data_file=data_filename,
        timestep_fs_over_1000=timestep_fs / 1000.0,
        temperature_k=temperature_k,
        seed=seed,
        thermo_interval=thermo_interval,
        dump_interval=dump_interval,
        dump_file=dump_file,
        n_steps=n_steps,
        model_file=model_file,
    )
    input_path = work_dir / input_filename
    with open(input_path, "w") as fh:
        fh.write(input_text)

    return {"data_file": str(data_path), "input_file": str(input_path), "type_map": type_map}


def run_md(data_file: str, input_file: str, lammps_executable: str = "lmp", dry_run: bool = True):
    import subprocess
    cmd = [lammps_executable, "-in", input_file]
    if dry_run:
        return cmd
    return subprocess.run(cmd, check=True, cwd=str(Path(input_file).parent))
