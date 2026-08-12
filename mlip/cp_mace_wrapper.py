"""
Wrapper around CP-MACE (github.com/yuanyue-liu-group/CP-MACE) for
constant-potential MLIP dataset preparation, training, and inference.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from systems.io_utils import write_extxyz


@dataclass
class LabeledConfig:
    species: Sequence[str]
    positions: "numpy.ndarray"  # noqa: F821
    cell: "numpy.ndarray"  # noqa: F821
    forces: "numpy.ndarray"  # noqa: F821
    energy: float
    electron: float
    potential: float


def write_cp_mace_dataset(configs: Iterable[LabeledConfig], out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp_frame")
    with open(out_path, "w") as fh:
        for cfg in configs:
            frame_text = write_extxyz(
                path=str(tmp),
                species=cfg.species,
                positions=cfg.positions,
                cell=cfg.cell,
                forces=cfg.forces,
                energy=cfg.energy,
                extra_fields={"electron": cfg.electron, "potential": cfg.potential},
            )
            fh.write(frame_text)
    if tmp.exists():
        tmp.unlink()


@dataclass
class CPMACETrainConfig:
    train_file: str
    name: str = "MACE_model"
    valid_fraction: float = 0.05
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    model: str = "FermiMACE"
    loss: str = "fermi_weighted"
    error_table: str = "Fermi_PerAtomRMSE"
    forces_weight: float = 100.0
    energy_weight: float = 1.0
    potential_weight: float = 10.0
    hidden_irreps: str = "128x0e + 128x1o"
    r_max: float = 5.0
    batch_size: int = 10
    max_num_epochs: int = 300
    device: str = "cuda"
    seed: int = 1

    def to_cli_args(self) -> list:
        return [
            "mace_run_train",
            f"--name={self.name}",
            f"--train_file={self.train_file}",
            f"--valid_fraction={self.valid_fraction}",
            '--config_type_weights={"Default":1.0}',
            f"--energy_key={self.energy_key}",
            f"--forces_key={self.forces_key}",
            "--E0s=average",
            f"--model={self.model}",
            f"--loss={self.loss}",
            f"--error_table={self.error_table}",
            f"--forces_weight={self.forces_weight}",
            f"--energy_weight={self.energy_weight}",
            f"--potential_weight={self.potential_weight}",
            f"--hidden_irreps={self.hidden_irreps}",
            f"--r_max={self.r_max}",
            f"--batch_size={self.batch_size}",
            f"--max_num_epochs={self.max_num_epochs}",
            "--ema",
            "--ema_decay=0.99",
            "--amsgrad",
            f"--device={self.device}",
            f"--seed={self.seed}",
        ]


def run_training(config: CPMACETrainConfig, dry_run: bool = True):
    cmd = config.to_cli_args()
    if dry_run:
        return cmd
    return subprocess.run(cmd, check=True)
