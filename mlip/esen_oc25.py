"""
Wrapper around fairchem OC25-trained eSEN models, using the CURRENT
fairchem-core >=2.x unified API (verified against the official
facebook/OC25 model card and facebookresearch/fairchem README, both
accessed 2026-08-12).

IMPORTANT: as of fairchem-core 2.x, OCPCalculator is a fairchem-v1-only
API and is NOT the recommended path for OC25 eSEN checkpoints. The
current, correct usage is:

    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    predictor = pretrained_mlip.get_predict_unit(
        "esen-sm-conserving-all-oc25", device="cuda"
    )
    calc = FAIRChemCalculator(predictor)

This module wraps that call. A legacy fairchem-v1 `OCPCalculator` path is
kept as an explicit opt-in fallback only (`config.use_legacy_v1=True`) for
environments pinned to fairchem-core 1.x.

Access requirements (gated, NOT open like the dataset itself):
  * The OC25 dataset is CC-BY-4.0.
  * The OC25 model CHECKPOINTS (esen-sm-conserving-all-oc25,
    esen-md-direct-all-oc25) are distributed under Meta's "FAIR Chemistry
    License" via a GATED Hugging Face repo (huggingface.co/facebook/OC25).
    You must request access (submit legal name, DOB, organization) and
    run `huggingface-cli login` with an access token before
    `pretrained_mlip.get_predict_unit(...)` will succeed.
"""

from dataclasses import dataclass

# Model reference names as listed on huggingface.co/facebook/OC25
# (checked 2026-08-12).
OC25_ESEN_MODEL_NAMES = {
    "esen-sm-conserving": "esen-sm-conserving-all-oc25",
    "esen-md-direct": "esen-md-direct-all-oc25",
}


@dataclass
class ESENOC25Config:
    model_name: str = "esen-sm-conserving-all-oc25"
    device: str = "cuda"  # "cuda" for GH200/A100, "cpu" otherwise
    task_name: str = "oc20"  # UMA-style task selector; ignored by non-UMA
                              # OC25-specific checkpoints but harmless to pass
    use_legacy_v1: bool = False
    legacy_checkpoint_path: str = ""  # only used if use_legacy_v1=True


def load_esen_oc25_calculator(config: ESENOC25Config):
    """Return an ASE-compatible Calculator backed by an OC25 eSEN
    checkpoint, for use as `atoms.calc = load_esen_oc25_calculator(cfg)`.

    Default path (fairchem-core >=2.x, current as of 2026-08-12):
        pretrained_mlip.get_predict_unit(config.model_name, device=...)
        -> FAIRChemCalculator(predictor)

    Requires prior `huggingface-cli login` with a token that has been
    granted access to huggingface.co/facebook/OC25.
    """
    if config.use_legacy_v1:
        try:
            from ocpmodels.common.relaxation.ase_utils import OCPCalculator
        except ImportError as exc:
            raise ImportError(
                "use_legacy_v1=True requires the fairchem-core v1 / "
                "ocpmodels package to be installed. Prefer the default "
                "(fairchem-core>=2.x) path unless you have a specific "
                "reason to pin to v1."
            ) from exc
        if not config.legacy_checkpoint_path:
            raise ValueError(
                "legacy_checkpoint_path must be set when use_legacy_v1=True."
            )
        return OCPCalculator(
            checkpoint_path=config.legacy_checkpoint_path,
            cpu=config.device == "cpu",
        )

    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as exc:
        raise ImportError(
            "fairchem-core (>=2.x) is required. Install via "
            "`pip install fairchem-core`, then run `huggingface-cli login` "
            "and request access at https://huggingface.co/facebook/OC25 "
            "before loading OC25 eSEN checkpoints."
        ) from exc

    predictor = pretrained_mlip.get_predict_unit(config.model_name, device=config.device)
    return FAIRChemCalculator(predictor, task_name=config.task_name)


def attach_calculator(atoms, config: ESENOC25Config):
    """Attach an eSEN-OC25 calculator to an ASE Atoms object in place."""
    atoms.calc = load_esen_oc25_calculator(config)
    return atoms
