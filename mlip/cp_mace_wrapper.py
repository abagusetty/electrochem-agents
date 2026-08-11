"""
Wrapper around CP-MACE (github.com/yuanyue-liu-group/CP-MACE) for
constant-potential MLIP training and inference.

CP-MACE requires an extended XYZ dataset with `electron=` and `potential=`
tags on each structure (see CP-MACE README). This module handles:

TODO:
- Convert OC25 + CP-DFT labeled configurations into CP-MACE xyz format.
- Wrap `mace_run_train` invocation with FermiMACE model / fermi_weighted loss.
- Load a trained FermiMACE checkpoint for inference (energy, forces, Fermi level).
"""

from dataclasses import dataclass


@dataclass
class CPMACETrainConfig:
    train_file: str
    model: str = "FermiMACE"
    loss: str = "fermi_weighted"
    potential_weight: float = 10.0
    forces_weight: float = 100.0
    energy_weight: float = 1.0
    hidden_irreps: str = "128x0e + 128x1o"
    r_max: float = 5.0
    batch_size: int = 10
    max_num_epochs: int = 300
    device: str = "cuda"


def write_cp_mace_dataset(configs, out_path: str):
    """Write a list of labeled configurations to CP-MACE extended-XYZ format.

    Each config must carry: positions, species, forces, energy, electron
    (net charge or electron count), and potential (Fermi level).
    """
    raise NotImplementedError("Phase 3: implement xyz writer with electron/potential tags")


def run_training(config: CPMACETrainConfig):
    """Invoke mace_run_train with CP-MACE FermiMACE arguments."""
    raise NotImplementedError("Phase 3: shell out to mace_run_train with config")
