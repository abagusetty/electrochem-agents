#!/usr/bin/env python3
"""
Teach an installed fairchem-core to run on Intel GPUs (XPU / Aurora PVC).

    python patch_fairchem_xpu.py --dry-run     # show the diff, change nothing
    python patch_fairchem_xpu.py               # apply, with .orig backups
    python patch_fairchem_xpu.py --revert      # restore from backups
    python patch_fairchem_xpu.py --diff > fairchem-xpu.patch   # for upstreaming

WHY A SOURCE PATCH AND NOT A MONKEYPATCH
-----------------------------------------
The primary blocker is a bare statement inside a function body:

    assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"

No runtime shim can reach a line inside a compiled function. You would have to
replace the entire enclosing method, which means vendoring upstream logic and
silently diverging from it at the next release. Editing the installed source is
smaller, inspectable with `diff`, reversible, and produces something you can
send upstream verbatim.

WHAT IT CHANGES -- 6 sites, 2 files
------------------------------------
fairchem/core/common/distutils.py
  1. assign_device_for_local_rank  -- add an XPU branch beside the CUDA one
  2. get_device_for_local_rank     -- resolve and return "xpu:N"

fairchem/core/units/mlip_unit/predict.py
  3. assert device in [...]        -- accept "xpu"
  4. self.device = ...             -- route xpu to get_device_for_local_rank
  5. backend = gloo/nccl           -- oneCCL for xpu (NCCL is NVIDIA-only).
                                      Selectable with --ccl-backend:
                                        xccl  oneCCL native in torch (default)
                                        ccl   external torch-ccl bindings
  6. num_gpu_per_worker            -- an xpu worker also needs a GPU slot

WHAT IT LEAVES ALONE BY DEFAULT -- and the --enable-triton-xpu opt-in
----------------------------------------------------------------------
Two `if torch.device(self._requested_device).type == "cuda":` guards exist.
By default both stay untouched: once the resolved device is `xpu:N` they are
simply skipped, which is correct and keeps the diff small. They need DIFFERENT
treatment, which is why they are not one flag:

  GUARD A  selects UMA-S "GPU execution mode", which dispatches Triton kernels.
           Skipping it costs performance, not correctness -- you get the
           portable path instead.
           With `triton-xpu` installed (Aurora ships 3.7.2), those kernels
           MIGHT run on Intel GPUs: Triton is designed to be portable, and the
           XPU backend compiles the same `tl.*` IR. "Might" is doing real work
           in that sentence -- kernels written against CUDA can still use
           backend-specific intrinsics, and `num_warps`/`num_stages` autotuning
           configs are tuned for NVIDIA SM occupancy, so a kernel can compile
           and be slower than the portable path. This is an OPT-IN
           (`--enable-triton-xpu`) precisely because it must be measured, not
           assumed. Benchmark against the default before keeping it.

  GUARD B  calls `torch.cuda.empty_cache()` on a fallback path. Widening this
           guard to xpu would be a BUG -- the call itself is CUDA-specific and
           would raise. `--enable-triton-xpu` therefore rewrites the call to
           dispatch per device type rather than just widening the condition.

>>> VERIFY THIS ASSUMPTION ON YOUR INSTALL. <<< If `_requested_device` holds the
raw *argument* ("cuda") rather than the resolved device, the Triton path would
activate on XPU and fail. `--check-requested-device` reports which it is.

SAFETY
------
Every edit is an exact full-line string replacement. If any target string is
absent -- because upstream moved on -- the script reports exactly which and
applies NOTHING. A partial patch is worse than none.

Not run against a real fairchem install.
"""

import argparse
import difflib
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

BACKUP_SUFFIX = ".orig-prexpu"


@dataclass
class Edit:
    """One exact-string replacement."""
    label: str
    old: str
    new: str
    required: bool = True


@dataclass
class FilePatch:
    module: str                      # dotted module path, for locating the file
    edits: List[Edit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The patch
# ---------------------------------------------------------------------------

DISTUTILS = FilePatch(
    module="fairchem.core.common.distutils",
    edits=[
        Edit(
            label="assign_device_for_local_rank: add XPU branch",
            old="""    if cpu:
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cpu"
    else:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cuda"
        torch.cuda.set_device(local_rank)""",
            new="""    if cpu:
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cpu"
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        # Intel GPU (Aurora PVC). Checked before CUDA so a machine with both
        # still lands on the accelerator the caller actually asked for; on
        # Aurora only this branch can ever be true.
        os.environ[CURRENT_DEVICE_TYPE_STR] = "xpu"
        torch.xpu.set_device(local_rank)
    else:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cuda"
        torch.cuda.set_device(local_rank)""",
        ),
        Edit(
            label="get_device_for_local_rank: auto-detect XPU",
            old="""        os.environ[CURRENT_DEVICE_TYPE_STR] = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )""",
            new="""        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            os.environ[CURRENT_DEVICE_TYPE_STR] = f"xpu:{torch.xpu.current_device()}"
        else:
            os.environ[CURRENT_DEVICE_TYPE_STR] = (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )""",
        ),
        Edit(
            label="get_device_for_local_rank: resolve XPU",
            old="""    if "cuda" in os.environ[CURRENT_DEVICE_TYPE_STR]:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        return f"cuda:{torch.cuda.current_device()}"
    elif os.environ[CURRENT_DEVICE_TYPE_STR] == "cpu":
        return "cpu\"""",
            new="""    if "cuda" in os.environ[CURRENT_DEVICE_TYPE_STR]:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        return f"cuda:{torch.cuda.current_device()}"
    elif "xpu" in os.environ[CURRENT_DEVICE_TYPE_STR]:
        assert (
            getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
        ), "device type is xpu but no XPU is available!"
        return f"xpu:{torch.xpu.current_device()}"
    elif os.environ[CURRENT_DEVICE_TYPE_STR] == "cpu":
        return "cpu\"""",
        ),
    ],
)

PREDICT = FilePatch(
    module="fairchem.core.units.mlip_unit.predict",
    edits=[
        Edit(
            label="accept 'xpu' as a device",
            old="""        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'\"""",
            new="""        assert device in [
            "cpu",
            "cuda",
            "xpu",
        ], "device must be one of 'cpu', 'cuda', 'xpu'\"""",
        ),
        Edit(
            label="resolve accelerator device (cuda or xpu)",
            old="""        self.device = get_device_for_local_rank() if device == "cuda" else "cpu\"""",
            new="""        self.device = (
            get_device_for_local_rank() if device in ("cuda", "xpu") else "cpu"
        )""",
        ),
        Edit(
            label="distributed backend: oneCCL for XPU",
            old="""        backend = "gloo" if device == "cpu" else "nccl\"""",
            new="""        # NCCL is NVIDIA-only. Intel GPUs collectives go through oneCCL,
        # reachable two ways:
        #   "xccl" -- oneCCL upstreamed into PyTorch as a native backend.
        #             Preferred: no extra package, and it IS oneCCL underneath.
        #   "ccl"  -- the older external bindings (torch-ccl /
        #             oneccl_bindings_for_pytorch), which must be imported
        #             before init_process_group to register the backend.
        if device == "cpu":
            backend = "gloo"
        elif device == "xpu":
            backend = "__CCL_BACKEND__"
        else:
            backend = "nccl\"""",
        ),
        Edit(
            label="reserve a GPU slot for XPU workers too",
            old="""        num_gpu_per_worker = 1 if device == "cuda" else 0""",
            new="""        num_gpu_per_worker = 1 if device in ("cuda", "xpu") else 0""",
        ),
    ],
)

# Opt-in, --enable-triton-xpu only. Off by default: needs measurement.
TRITON_XPU = FilePatch(
    module="fairchem.core.units.mlip_unit.predict",
    edits=[
        Edit(
            label="GUARD A: allow UMA-S Triton path on XPU (needs triton-xpu)",
            old="""        if torch.device(self._requested_device).type == "cuda":
            self.inference_settings = maybe_update_settings_backend(""",
            new="""        if torch.device(self._requested_device).type in ("cuda", "xpu"):
            # XPU included: triton-xpu compiles the same Triton IR. Portability
            # is a design goal of Triton, NOT a guarantee for a given kernel --
            # measure this against the default path before trusting it.
            self.inference_settings = maybe_update_settings_backend(""",
        ),
        Edit(
            label="GUARD B: device-correct empty_cache (NOT a guard widening)",
            old="""        if torch.device(self._requested_device).type == "cuda":
            torch.cuda.empty_cache()""",
            new="""        # Dispatch on device type. Widening the condition alone would be a
        # bug: torch.cuda.empty_cache() raises on an XPU device.
        _dev_type = torch.device(self._requested_device).type
        if _dev_type == "cuda":
            torch.cuda.empty_cache()
        elif _dev_type == "xpu":
            torch.xpu.empty_cache()""",
        ),
    ],
)

PATCHES = [DISTUTILS, PREDICT]

# Placeholder resolved from --ccl-backend before the edits are applied.
CCL_BACKEND_PLACEHOLDER = "__CCL_BACKEND__"
CCL_BACKENDS = {
    "xccl": ("oneCCL via PyTorch's native XCCL backend. No extra package. "
             "Requires a torch built with XCCL support (torch >= ~2.7)."),
    "ccl": ("oneCCL via the external torch-ccl / oneccl_bindings_for_pytorch "
            "bindings. Those must be imported before init_process_group or "
            "the backend name is unregistered."),
}


def resolve_ccl_backend(name: str) -> None:
    """Substitute the chosen oneCCL backend name into the pending edits."""
    for patch in PATCHES:
        for edit in patch.edits:
            if CCL_BACKEND_PLACEHOLDER in edit.new:
                edit.new = edit.new.replace(CCL_BACKEND_PLACEHOLDER, name)


# ---------------------------------------------------------------------------

def locate(module: str) -> Path:
    """Find an installed module's source file without importing fairchem."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        # find_spec raises ModuleNotFoundError when a PARENT package is absent,
        # rather than returning None. Without this the script dies with a
        # traceback instead of the actionable message below.
        spec = None
    if spec is None or not spec.origin:
        raise FileNotFoundError(
            f"Cannot locate {module}. Is fairchem-core installed in this "
            "environment? Activate the venv that has it, then re-run."
        )
    return Path(spec.origin)


def apply_edits(text: str, edits: List[Edit]) -> Tuple[str, List[str], List[str]]:
    applied, missing = [], []
    for edit in edits:
        if edit.new in text and edit.old not in text:
            applied.append(f"{edit.label} (already patched)")
            continue
        if edit.old not in text:
            if edit.required:
                missing.append(edit.label)
            continue
        if text.count(edit.old) > 1:
            missing.append(f"{edit.label} (AMBIGUOUS: {text.count(edit.old)} matches)")
            continue
        text = text.replace(edit.old, edit.new, 1)
        applied.append(edit.label)
    return text, applied, missing


def unified(path: Path, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path.name}", tofile=f"b/{path.name}", n=3))


def check_requested_device(predict_path: Path) -> str:
    """Report what `_requested_device` is assigned from.

    Decides whether the untouched `torch.device(self._requested_device).type
    == "cuda"` guards stay correctly inactive on XPU. If it holds the raw
    argument rather than the resolved device, those Triton/CUDA paths would
    fire and the patch is incomplete.
    """
    text = predict_path.read_text()
    hits = re.findall(r"^\s*self\._requested_device\s*=\s*(.+)$", text, re.MULTILINE)
    if not hits:
        return ("could not find an assignment to self._requested_device -- "
                "inspect the two `torch.device(self._requested_device).type == "
                "\"cuda\"` guards manually.")
    joined = "; ".join(h.strip() for h in hits)
    if any("self.device" in h for h in hits):
        return (f"OK: _requested_device = {joined} -> resolves to 'xpu:N', so the "
                "CUDA-only Triton/empty_cache guards stay inactive. No further "
                "change needed.")
    return (f"CHECK THIS: _requested_device = {joined}. If that is the raw "
            "'cuda'/'xpu' argument rather than the resolved device, the "
            "`.type == \"cuda\"` guards behave on the argument. That is fine "
            "for device='xpu' (type is 'xpu', guards stay off) but would be "
            "wrong if you patched by passing device='cuda' instead.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diff", action="store_true",
                        help="print a unified diff to stdout and exit")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--check-requested-device", action="store_true")
    parser.add_argument("--enable-triton-xpu", action="store_true",
                        help="EXPERIMENTAL. Also route the UMA-S Triton path to "
                             "XPU (needs triton-xpu) and make empty_cache "
                             "device-correct. Measure against the default "
                             "before keeping it.")
    parser.add_argument("--ccl-backend", default="xccl", choices=sorted(CCL_BACKENDS),
                        help="oneCCL backend name for XPU collectives. "
                             "xccl = native in torch (default); "
                             "ccl = external torch-ccl bindings.")
    args = parser.parse_args()
    if args.enable_triton_xpu and TRITON_XPU not in PATCHES:
        PATCHES.append(TRITON_XPU)
    resolve_ccl_backend(args.ccl_backend)

    try:
        paths = {p.module: locate(p.module) for p in PATCHES}
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.check_requested_device:
        print(check_requested_device(paths[PREDICT.module]))
        return 0

    if args.revert:
        reverted = 0
        for path in paths.values():
            backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
            if backup.exists():
                shutil.copy2(backup, path)
                backup.unlink()
                print(f"reverted {path}")
                reverted += 1
            else:
                print(f"no backup for {path} (nothing to revert)")
        return 0 if reverted else 1

    results, blocked = [], []
    for patch in PATCHES:
        path = paths[patch.module]
        before = path.read_text()
        after, applied, missing = apply_edits(before, patch.edits)
        results.append((path, before, after, applied, missing))
        if missing:
            blocked.append((path, missing))

    if blocked:
        print("REFUSING TO PATCH -- these targets were not found:\n", file=sys.stderr)
        for path, missing in blocked:
            print(f"  {path}", file=sys.stderr)
            for label in missing:
                print(f"      - {label}", file=sys.stderr)
        print("\nUpstream has changed. Nothing was written: a partially applied "
              "patch is worse than none, because fairchem would accept 'xpu' "
              "and then route it somewhere wrong. Re-derive the edits against "
              "your installed version.", file=sys.stderr)
        return 1

    if args.diff or args.dry_run:
        for path, before, after, applied, _ in results:
            if before == after:
                print(f"# {path}: no change (already patched)")
                continue
            print(unified(path, before, after), end="")
        if args.dry_run:
            print("\n# --dry-run: nothing written.", file=sys.stderr)
        return 0

    for path, before, after, applied, _ in results:
        if before == after:
            print(f"unchanged (already patched): {path}")
            continue
        backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(after)
        print(f"patched {path}")
        for label in applied:
            print(f"    + {label}")
        print(f"    backup: {backup}")

    print("\n" + check_requested_device(paths[PREDICT.module]))
    print(f"\nXPU collectives use backend={args.ccl_backend!r}: "
          f"{CCL_BACKENDS[args.ccl_backend]}")
    print("Single-device inference -- what the acquisition loop needs -- never "
          "reaches this line, so an unregistered backend surfaces only under "
          "Ray-distributed multi-worker use.")
    if args.enable_triton_xpu:
        print("\nTriton-XPU path ENABLED (experimental). Benchmark it against "
              "the default before keeping it: a CUDA-authored Triton kernel can "
              "compile on XPU and still be slower than the portable path, "
              "because its autotuning configs were tuned for NVIDIA occupancy. "
              "Revert with --revert if it regresses.")
    print("\nNext:")
    print("  python -c \"import torch;print(torch.__version__, torch.xpu.is_available())\"")
    print("  python check_aurora_env.py --deep      # forward AND backward on XPU")
    print("\nForward alone proves nothing -- a missing XPU autograd kernel gives "
          "a model that loads, returns an energy, and cannot run MD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
