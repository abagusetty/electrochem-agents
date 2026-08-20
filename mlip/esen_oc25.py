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

import os
from pathlib import Path
from dataclasses import dataclass

# Model reference names as listed on huggingface.co/facebook/OC25
# (checked 2026-08-12).
OC25_ESEN_MODEL_NAMES = {
    "esen-sm-conserving": "esen-sm-conserving-all-oc25",
    "esen-md-direct": "esen-md-direct-all-oc25",
}


# ---------------------------------------------------------------------------
# XPU / AURORA STATUS -- READ BEFORE PLANNING AROUND THIS MODULE
# ---------------------------------------------------------------------------
# fairchem-core does NOT run on Intel GPUs as shipped. Verified against
# fairchem main, 2026-08-20:
#
#   src/fairchem/core/units/mlip_unit/predict.py
#       assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"
#       self.device = get_device_for_local_rank() if device == "cuda" else "cpu"
#       backend = "gloo" if device == "cpu" else "nccl"
#       num_gpu_per_worker = 1 if device == "cuda" else 0
#
# `device="xpu"` therefore raises AssertionError immediately. It is a runtime
# assert, not merely the `Literal["cuda","cpu"]` type hint on
# get_predict_unit() -- passing "xpu" does not degrade gracefully, it stops.
#
# The good news is that the block is shallow: the repository contains no .cu,
# .cpp or .cuh sources, so this is pure Python over torch. The CUDA-specific
# paths that do exist (Triton kernels for UMA-S GPU mode, tf32 context, NCCL)
# are already gated behind `device == "cuda"` and would simply be skipped. A
# working XPU path needs: the assert widened, `self.device` resolved for xpu,
# and a distributed backend other than NCCL (ccl/gloo).
#
# Until that patch exists, the options on Aurora are, in order:
#   1. Route B (CP-MACE) instead -- its mace/tools/torch_tools.py init_device
#      has an explicit `elif device_str == "xpu"` branch, so it is the
#      Aurora-native path.
#   2. fairchem on CPU here. Xeon CPU Max with HBM is not absurd for
#      inference, but it is far off GPU throughput.
#   3. fairchem on NVIDIA, JDFTx on Aurora -- costs the co-located loop.
#
# See RESEARCH_PLAN.md G4.
SUPPORTED_DEVICES = ("cuda", "cpu")


def fairchem_accepts_xpu() -> bool:
    """Whether the INSTALLED fairchem has been taught about XPU.

    Read from the installed source rather than tracked as a constant, so this
    stays correct after `scripts/patch_fairchem_xpu.py` runs (or after a pip
    upgrade silently reverts it). Cheap: one find_spec plus one file read.
    """
    import importlib.util

    override = os.environ.get("ELECTROCHEM_FAIRCHEM_XPU")
    if override is not None:
        return override.lower() in ("1", "true", "yes")
    try:
        spec = importlib.util.find_spec("fairchem.core.units.mlip_unit.predict")
        if spec is None or not spec.origin:
            return False
        text = Path(spec.origin).read_text()
    except Exception:                                           # noqa: BLE001
        return False
    # The patched assert lists "xpu" among the accepted devices.
    return '"xpu",' in text or "'xpu'," in text or '"xpu"]' in text


@dataclass
class ESENOC25Config:
    model_name: str = "esen-sm-conserving-all-oc25"
    device: str = "cuda"  # "cuda" | "cpu" ONLY -- see XPU note above
    task_name: str = "oc20"  # UMA-style task selector; ignored by non-UMA
                              # OC25-specific checkpoints but harmless to pass
    use_legacy_v1: bool = False
    legacy_checkpoint_path: str = ""  # only used if use_legacy_v1=True


def _check_device(device: str) -> None:
    """Fail here, with an explanation, rather than inside fairchem's assert.

    Stock fairchem raises a bare `AssertionError: device must be either 'cpu'
    or 'cuda'` with no hint that Intel GPUs are unimplemented rather than
    misconfigured -- which on Aurora reads as the user's mistake.
    """
    base = device.split(":")[0]
    if base in SUPPORTED_DEVICES:
        return
    if base == "xpu" and fairchem_accepts_xpu():
        return                      # patched install; let fairchem handle it
    raise ValueError(
        f"fairchem-core does not support device={device!r}. Upstream asserts "
        f"device in {list(SUPPORTED_DEVICES)} "
        "(src/fairchem/core/units/mlip_unit/predict.py), so Intel GPUs are "
        "unimplemented -- not merely untested.\n"
        "On Aurora, choose one:\n"
        "  * use CP-MACE (Route B), whose init_device has an explicit xpu "
        "branch;\n"
        "  * run fairchem on 'cpu' here (Xeon CPU Max, far below GPU "
        "throughput);\n"
        "  * run fairchem on an NVIDIA system and keep JDFTx on Aurora, "
        "which costs the co-located acquisition loop;\n"
        "  * patch fairchem: run scripts/patch_fairchem_xpu.py. Six exact "
        "edits across two files (no CUDA sources exist, so the block is "
        "shallow); it verifies every target before writing, keeps .orig "
        "backups, and supports --dry-run / --diff / --revert. This function "
        "re-reads the installed source, so it starts accepting 'xpu' as soon "
        "as the patch is applied."
    )


def load_esen_oc25_calculator(config: ESENOC25Config):
    """Return an ASE-compatible Calculator backed by an OC25 eSEN
    checkpoint, for use as `atoms.calc = load_esen_oc25_calculator(cfg)`.

    Default path (fairchem-core >=2.x, current as of 2026-08-12):
        pretrained_mlip.get_predict_unit(config.model_name, device=...)
        -> FAIRChemCalculator(predictor)

    Requires prior `huggingface-cli login` with a token that has been
    granted access to huggingface.co/facebook/OC25.
    """
    _check_device(config.device)

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
