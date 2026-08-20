#!/usr/bin/env bash
# Build the Python environment for this project on Sunspot / Aurora.
#
#   bash scripts/setup_sunspot_env.sh [VENV_DIR]
#
# Idempotent: safe to re-run. Verified on a Sunspot compute node 2026-08-20
# against frameworks/2026.1.0 (torch 2.13.0a0+gitcf30153, 12 PVC tiles).
#
# ---------------------------------------------------------------------------
# The one rule: never let pip touch torch or numpy.
# ---------------------------------------------------------------------------
# The frameworks module ships an ALCF source build of torch with the XPU
# backend. A PyPI wheel is CUDA- or CPU-only and would shadow it, silently
# killing every XPU path.
#
# This is not hypothetical. fairchem-core pins `torch~=2.13.0`, and under
# PEP 440 an alpha sorts BELOW the release -- 2.13.0a0 < 2.13.0 -- so the
# module's torch does NOT satisfy fairchem's own pin, and an unconstrained
# `pip install fairchem-core` genuinely resolves a replacement from PyPI.
# Reproduced live.
#
# Hence: a constraints file pinning torch and numpy to the module's exact
# versions. pip then fails loudly instead of swapping them out.
set -eo pipefail
# NOTE: deliberately no `set -u`. Lmod's init script dereferences unset
# variables (ZSH_EVAL_CONTEXT), so `module load` aborts under nounset.

VENV="${1:-/tegu/Performance/abagusetty/electrochem-agents/elechem-venv}"

# shellcheck disable=SC1091
source /etc/profile.d/modules.sh 2>/dev/null || true
module load frameworks

if [[ ! -d "$VENV" ]]; then
    echo ">> creating venv (with system site packages) at $VENV"
    python -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

TORCH_V=$(python -c 'import torch; print(torch.__version__)')
NUMPY_V=$(python -c 'import numpy; print(numpy.__version__)')
echo ">> module provides torch=$TORCH_V numpy=$NUMPY_V"

CONSTRAINTS=$(mktemp /tmp/aurora-constraints.XXXXXX.txt)
printf 'torch==%s\nnumpy==%s\n' "$TORCH_V" "$NUMPY_V" > "$CONSTRAINTS"
trap 'rm -f "$CONSTRAINTS"' EXIT

# --- absent from the module, and torch-free: install with deps under constraints
echo ">> installing science stack"
python -m pip install -q -c "$CONSTRAINTS" \
    'ase>=3.26.0' 'pymatgen>=2025.4' 'e3nn>=0.5' \
    ase-db-backends 'clusterscope==0.0.18' hydra-core 'lmdb>=1.6.2,<=1.7.3' \
    monty orjson opt_einsum_fx pymatgen-core submitit wandb \
    pyre-extensions tensorboard \
    pytest pytest-cov coverage syrupy

# --- declares a torch dependency -> --no-deps so pip cannot reconsider torch
python -m pip install -q --no-deps torchtnt

# --- fairchem: the XPU-ported fork, else the PyPI build (CPU/CUDA only)
FAIRCHEM_SRC="${ELECTROCHEM_FAIRCHEM_SRC:-/tegu/Performance/abagusetty/electrochem-agents/fairchem}"
if [[ -d "$FAIRCHEM_SRC/packages/fairchem-core" ]]; then
    echo ">> installing fairchem from $FAIRCHEM_SRC (editable, XPU-ported fork)"
    HATCH_VCS_PRETEND_VERSION=${HATCH_VCS_PRETEND_VERSION:-2.22.0} \
        python -m pip install -q --no-deps -e "$FAIRCHEM_SRC/packages/fairchem-core"
else
    echo ">> WARNING: $FAIRCHEM_SRC not found; installing stock fairchem-core."
    echo "   Stock fairchem asserts device in ['cpu','cuda'] and will NOT run on XPU."
    echo "   Clone github.com/abagusetty/fairchem (branch xpu-support) for Route A."
    python -m pip install -q --no-deps fairchem-core
fi

# --- the check that matters: did anything replace torch?
echo
python - <<'PY'
import torch, sys
print(f"torch  : {torch.__version__}")
print(f"xpu    : available={torch.xpu.is_available()} devices={torch.xpu.device_count()}")
if "+git" not in torch.__version__:
    sys.exit("FATAL: torch was replaced by a PyPI wheel. Re-create the venv.")
if not torch.xpu.is_available():
    sys.exit("FATAL: no XPU. Are you on a compute node? (qsub -I ...)")
try:
    from fairchem.core.common.device_utils import SUPPORTED_DEVICE_TYPES
    print(f"fairchem: XPU-capable, supports {list(SUPPORTED_DEVICE_TYPES)}")
except ImportError:
    import fairchem.core  # noqa: F401
    print("fairchem: stock build -- CPU/CUDA only, no XPU support")
print("\nOK. Next: python scripts/check_aurora_env.py")
PY
