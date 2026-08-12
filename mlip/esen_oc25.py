"""
Wrapper around fairchem OC25-trained eSEN models for constant-charge MD.
Checkpoints (e.g. "esen-sm-conserving-all-oc25") should be downloaded from
https://huggingface.co/facebook/OC25.
"""

from dataclasses import dataclass


@dataclass
class ESENOC25Config:
    checkpoint_path: str
    device: str = "cuda"
    cpu: bool = False


def load_esen_oc25_calculator(config: ESENOC25Config):
    calculator_cls = None
    try:
        from fairchem.core import OCPCalculator as calculator_cls  # type: ignore
    except ImportError:
        try:
            from ocpmodels.common.relaxation.ase_utils import OCPCalculator as calculator_cls  # type: ignore
        except ImportError:
            pass

    if calculator_cls is None:
        raise ImportError(
            "Could not import an OCPCalculator from fairchem.core or "
            "ocpmodels. Install fairchem via "
            "https://github.com/facebookresearch/fairchem and download the "
            "OC25 eSEN checkpoint from https://huggingface.co/facebook/OC25."
        )

    return calculator_cls(
        checkpoint_path=config.checkpoint_path,
        cpu=config.cpu or config.device == "cpu",
    )


def attach_calculator(atoms, config: ESENOC25Config):
    atoms.calc = load_esen_oc25_calculator(config)
    return atoms
