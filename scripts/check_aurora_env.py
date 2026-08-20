#!/usr/bin/env python3
"""
Standalone Aurora/XPU environment probe. No repo dependency -- scp this one
file to Aurora and run it.

    module load frameworks
    python check_aurora_env.py                 # login node: env only
    python check_aurora_env.py --deep          # compute node: real fwd+bwd
    python check_aurora_env.py --deep --cp-mace-model /path/to/x.model

WHAT IT ANSWERS, IN ORDER OF DECISIVENESS

  1. torch + XPU alive?              cheap, settles the runtime
  2. e3nn tensor product fwd+BWD?    THE CRUX -- see below
  3. scatter ops on device?          message passing needs them
  4. Triton kernel compiles on XPU?  gates fairchem's UMA-S Triton path, and
                                     tells you whether a Triton-XPU CG kernel
                                     is worth writing
  5. fairchem eSEN loads + runs?     Route A
  6. CP-MACE FermiMACE loads+runs?   Route B, plus get_mu()

WHY THE e3nn TEST IS THE CRUX
-----------------------------
Equivariant MLIPs spend most of their time in Clebsch-Gordan tensor products.
Two libraries accelerate that op, and NEITHER helps on Aurora:

  cuEquivariance    NVIDIA only.
  OpenEquivariance  CUDA + HIP only. Checked 2026-08-20: no SYCL/oneAPI/XPU
                    backend in the repo, and no open issue requesting one.

So on Intel GPUs both eSEN and MACE must run e3nn's native pure-PyTorch path.
Pure PyTorch means it *should* work -- but "should" is what this script exists
to replace. If the e3nn tensor product runs forward and backward on XPU, the
models will run; if it does not, nothing above it will either, and you have
your answer in two seconds instead of after a model download.

Expect the native path to be SLOWER than an accelerated NVIDIA run. Slower is
fine. Broken is not. The timing this prints is the number to weigh against
"put MLIP inference on NVIDIA instead".

FORWARD IS NOT ENOUGH
---------------------
Every check runs a BACKWARD pass too. A missing XPU autograd kernel yields a
model that imports, loads, returns an energy, and cannot do MD. That failure
reports as forward-ok / backward-fail here, explicitly.

Exit codes:  0 usable   1 device failure   2 environment incomplete
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

RESULTS: List[Dict[str, Any]] = []


def record(name: str, fn: Callable[[], Dict[str, Any]],
           kind: str = "device") -> Dict[str, Any]:
    """Run one check. Never raises -- one broken package must not hide the
    state of the others, which is the whole point of a probe."""
    entry: Dict[str, Any] = {"check": name, "kind": kind}
    start = time.time()
    try:
        entry.update(fn())
        entry.setdefault("ok", True)
    except ImportError as exc:
        entry.update({"ok": False, "stage": "import", "kind": "environment",
                      "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:                                    # noqa: BLE001
        entry.update({"ok": False, "stage": entry.get("stage", "run"),
                      "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(limit=3)})
    entry["seconds"] = round(time.time() - start, 3)
    RESULTS.append(entry)
    return entry


# ---------------------------------------------------------------------------
# 1. runtime
# ---------------------------------------------------------------------------

def check_torch() -> Dict[str, Any]:
    import torch
    xpu = getattr(torch, "xpu", None)
    available = bool(xpu is not None and xpu.is_available())
    out: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "xpu_available": available,
        "cuda_available": bool(torch.cuda.is_available()),
        "xpu_device_count": int(xpu.device_count()) if available else 0,
    }
    if not available:
        out.update({"ok": False, "stage": "xpu_unavailable", "kind": "environment",
                    "hint": "Expected on a login node. Re-run inside a PBS job. "
                            "If it persists on a compute node, check "
                            "`module load frameworks` and that no pip torch "
                            "wheel is shadowing the ALCF build."})
        return out
    out["xpu_device_name"] = torch.xpu.get_device_name(0)
    a = torch.randn(2048, 2048, device="xpu", requires_grad=True)
    loss = (a @ a).sum()
    loss.backward()                       # exercises autograd on device
    torch.xpu.synchronize()
    out["matmul_fwd_bwd_ok"] = bool(a.grad is not None)
    out["fp64_ok"] = _fp64(torch)
    return out


def _fp64(torch) -> Any:
    """fp64 is available on PVC but slow. Some MLIP paths default to it."""
    try:
        x = torch.ones(256, 256, device="xpu", dtype=torch.float64)
        return bool(float((x @ x).sum()) > 0)
    except Exception as exc:                                    # noqa: BLE001
        return f"<{type(exc).__name__}>"


# ---------------------------------------------------------------------------
# 2. e3nn tensor product -- the crux
# ---------------------------------------------------------------------------

def check_e3nn_tensor_product() -> Dict[str, Any]:
    import torch
    from e3nn import o3

    device = "xpu" if torch.xpu.is_available() else "cpu"
    # Irreps typical of a MACE/eSEN layer: scalars + vectors + rank-2, L=2.
    irreps_in1 = o3.Irreps("32x0e + 32x1o + 16x2e")
    irreps_in2 = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = o3.Irreps("32x0e + 32x1o + 16x2e")
    tp = o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=True).to(device)

    n = 4096                                   # ~edges in an 800-atom cell
    x1 = irreps_in1.randn(n, -1).to(device).requires_grad_(True)
    x2 = irreps_in2.randn(n, -1).to(device).requires_grad_(True)

    out = tp(x1, x2)                           # forward
    if device == "xpu":
        torch.xpu.synchronize()

    loss = out.pow(2).sum()
    loss.backward()                            # backward -- the real test
    if device == "xpu":
        torch.xpu.synchronize()

    start = time.time()
    for _ in range(10):
        tp(x1, x2).pow(2).sum().backward()
    if device == "xpu":
        torch.xpu.synchronize()
    per_iter_ms = (time.time() - start) / 10 * 1000

    return {
        "e3nn_version": __import__("e3nn").__version__,
        "device": device,
        "n_edges": n,
        "forward_ok": bool(out.shape[0] == n),
        "backward_ok": bool(x1.grad is not None and x2.grad is not None),
        "fwd_bwd_ms_per_iter": round(per_iter_ms, 2),
        "note": ("Native e3nn path. Neither cuEquivariance nor "
                 "OpenEquivariance has an Intel backend, so this is the only "
                 "path available on Aurora. Compare this timing against an "
                 "NVIDIA run before deciding where MLIP inference lives."),
    }


def check_scatter() -> Dict[str, Any]:
    """Message passing needs scatter. Native torch ops are preferred; the
    torch_scatter package historically shipped CUDA-only wheels."""
    import torch
    device = "xpu" if torch.xpu.is_available() else "cpu"
    src = torch.randn(8192, 64, device=device, requires_grad=True)
    index = torch.randint(0, 512, (8192,), device=device)

    native = torch.zeros(512, 64, device=device)
    native = native.index_add(0, index, src)
    native.sum().backward()
    out: Dict[str, Any] = {"device": device,
                           "native_index_add_fwd_bwd_ok": bool(src.grad is not None)}

    reduce_target = torch.zeros(512, 64, device=device)
    reduce_target.scatter_reduce_(0, index.unsqueeze(-1).expand(-1, 64),
                                  src.detach(), reduce="amax",
                                  include_self=False)
    out["native_scatter_reduce_ok"] = True

    try:
        import torch_scatter
        got = torch_scatter.scatter_add(src.detach(), index, dim=0, dim_size=512)
        out["torch_scatter_version"] = torch_scatter.__version__
        out["torch_scatter_ok"] = bool(got.shape == (512, 64))
    except ImportError:
        out["torch_scatter"] = "absent (fine -- native ops are used above)"
    except Exception as exc:                                    # noqa: BLE001
        out["torch_scatter_ok"] = f"<{type(exc).__name__}: {exc}>"
    return out


# ---------------------------------------------------------------------------
# 3. accelerators that should be ABSENT on Aurora
# ---------------------------------------------------------------------------

def check_triton_xpu() -> Dict[str, Any]:
    """Does a Triton kernel actually COMPILE AND RUN on XPU?

    Aurora ships triton-xpu (3.7.2). That matters twice:

      1. fairchem's UMA-S "GPU execution mode" dispatches Triton kernels behind
         a `device.type == "cuda"` guard. If Triton works here, that path is a
         candidate (patch_fairchem_xpu.py --enable-triton-xpu) rather than
         permanently off.
      2. Bigger: the Clebsch-Gordan tensor product has no Intel-GPU accelerator
         at all -- cuEquivariance is NVIDIA-only and OpenEquivariance is
         CUDA+HIP-only. Both are JIT kernel generators. A Triton-XPU CG kernel
         is therefore feasible, and nobody has written one.

    Importing triton proves nothing; a kernel must compile for the XPU target
    and produce correct numbers. That is what this runs.
    """
    import torch

    out: Dict[str, Any] = {}
    import triton
    import triton.language as tl
    out["triton_version"] = triton.__version__

    if not torch.xpu.is_available():
        return {**out, "ok": False, "stage": "xpu_unavailable", "kind": "environment",
                "hint": "Run on a compute node."}

    @triton.jit
    def _axpy(x_ptr, y_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a * x + y, mask=mask)

    n = 4096
    x = torch.randn(n, device="xpu")
    y = torch.randn(n, device="xpu")
    got = torch.empty_like(x)

    start = time.time()
    _axpy[(triton.cdiv(n, 256),)](x, y, got, 2.5, n, BLOCK=256)
    torch.xpu.synchronize()
    out["compile_and_run_s"] = round(time.time() - start, 3)

    expected = 2.5 * x + y
    out["max_abs_error"] = float((got - expected).abs().max())
    out["numerically_correct"] = bool(out["max_abs_error"] < 1e-4)
    out["ok"] = out["numerically_correct"]
    out["implication"] = (
        "Triton compiles for XPU. fairchem's UMA-S Triton path becomes a "
        "candidate (--enable-triton-xpu, measure it), and a Triton-XPU "
        "Clebsch-Gordan kernel is feasible -- the gap OpenEquivariance leaves "
        "on Intel GPUs."
        if out["ok"] else
        "Triton did NOT produce correct results on XPU. Keep the fairchem "
        "Triton path disabled (the default).")
    return out


def check_accelerators() -> Dict[str, Any]:
    """cuEquivariance and OpenEquivariance are NVIDIA(+HIP) only.

    Their presence on Aurora is a warning, not a win: a stack that detects one
    and tries to use it will fail at runtime rather than fall back cleanly.
    Absence is the expected, correct state.
    """
    out: Dict[str, Any] = {}
    for module, note in (("cuequivariance", "NVIDIA only"),
                         ("cuequivariance_torch", "NVIDIA only"),
                         ("openequivariance", "CUDA + HIP only; no SYCL/XPU backend")):
        try:
            m = __import__(module)
            out[module] = {"present": True,
                           "version": getattr(m, "__version__", "unknown"),
                           "warning": f"{note} -- should NOT be active on XPU. "
                                      "Ensure the model config selects the "
                                      "native e3nn path."}
        except ImportError:
            out[module] = {"present": False, "expected": True, "note": note}
    for module in ("flash_attn",):
        try:
            __import__(module)
            out[module] = {"present": True, "warning": "CUDA only"}
        except ImportError:
            out[module] = {"present": False, "expected": True}
    return out


# ---------------------------------------------------------------------------
# 4/5. real models
# ---------------------------------------------------------------------------

def _tiny_cu_water(n_water: int = 8):
    import numpy as np
    from ase import Atoms
    a = 3.615
    positions = [[x * a / 2, y * a / 2, 6.0 - i * a / 2]
                 for i in range(2) for x in range(3) for y in range(3)]
    symbols = ["Cu"] * len(positions)
    rng = np.random.default_rng(0)
    for _ in range(n_water):
        o = np.array([rng.uniform(0, 1.5 * a), rng.uniform(0, 1.5 * a),
                      rng.uniform(9.0, 13.0)])
        positions += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
        symbols += ["O", "H", "H"]
    return Atoms(symbols, positions=np.array(positions),
                 cell=[1.5 * a, 1.5 * a, 20.0], pbc=True)


def _run_calculator(build, device: str) -> Dict[str, Any]:
    atoms = _tiny_cu_water()
    out: Dict[str, Any] = {"device": device, "n_atoms": len(atoms)}
    out["stage"] = "load_model"
    atoms.calc = build(device)
    out["stage"] = "forward"
    start = time.time()
    out["energy_ev"] = float(atoms.get_potential_energy())
    out["forward_s"] = round(time.time() - start, 3)
    out["stage"] = "backward_forces"
    start = time.time()
    forces = atoms.get_forces()
    out["max_force_ev_per_ang"] = float(abs(forces).max())
    out["forces_s"] = round(time.time() - start, 3)
    getter = getattr(atoms.calc, "get_mu", None)
    if callable(getter):
        out["mu_ev"] = float(getter())
        out["get_mu_ok"] = True
    out["stage"] = "done"
    return out


def check_fairchem(model: str, device: str) -> Dict[str, Any]:
    def build(dev):
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device=dev))
    result = _run_calculator(build, device)
    result["model"] = model
    return result


def check_cp_mace(model_path: str, repo: Optional[str], device: str) -> Dict[str, Any]:
    def build(dev):
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=[model_path], device=dev)
    result = _run_calculator(build, device)
    result["model_path"] = model_path
    return result


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deep", action="store_true",
                        help="also load and run the real MLIP models")
    parser.add_argument("--esen-model", default="esen-sm-conserving-all-oc25")
    parser.add_argument("--cp-mace-model", default=None)
    parser.add_argument("--cp-mace-repo", default=os.environ.get("ELECTROCHEM_CP_MACE_REPO"))
    parser.add_argument("--json", action="store_true", help="JSON only")
    args = parser.parse_args()

    env = {k: os.environ.get(k) for k in
           ("ZE_FLAT_DEVICE_HIERARCHY", "ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR",
            "PBS_JOBID", "PALS_LOCAL_RANKID")}

    torch_entry = record("torch_xpu", check_torch)
    record("e3nn_tensor_product", check_e3nn_tensor_product)
    record("scatter_ops", check_scatter)
    record("triton_xpu", check_triton_xpu)
    record("accelerator_libs", check_accelerators, kind="info")

    if args.deep:
        device = "xpu" if torch_entry.get("xpu_available") else "cpu"
        record("fairchem_esen", lambda: check_fairchem(args.esen_model, device))
        if args.cp_mace_model:
            record("cp_mace_fermimace",
                   lambda: check_cp_mace(args.cp_mace_model, args.cp_mace_repo, device))
        else:
            RESULTS.append({"check": "cp_mace_fermimace", "ok": None,
                            "skipped": "pass --cp-mace-model <path>.model"})

    failures = [r for r in RESULTS if r.get("ok") is False]
    device_fail = [r for r in failures if r.get("kind") != "environment"]
    env_fail = [r for r in failures if r.get("kind") == "environment"]

    if device_fail:
        code, verdict = 1, ("DEVICE FAILURE: " +
                            ", ".join(f"{r['check']}@{r.get('stage')}" for r in device_fail) +
                            ". forward-ok + backward-fail means a missing XPU "
                            "autograd kernel: the model loads and cannot do MD.")
    elif env_fail:
        code, verdict = 2, ("ENVIRONMENT INCOMPLETE: " +
                            ", ".join(r["check"] for r in env_fail) +
                            ". Nothing reached the device -- this is NOT an "
                            "XPU verdict. Load the module / install the stack "
                            "and re-run on a COMPUTE node.")
    else:
        code, verdict = 0, "ALL CHECKS PASSED on this node."

    payload = {"host": platform.node(), "python": sys.version.split()[0],
               "env": env, "results": RESULTS, "verdict": verdict,
               "exit_code": code}

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return code

    print(f"host={payload['host']}  python={payload['python']}")
    print("env: " + "  ".join(f"{k}={v}" for k, v in env.items() if v))
    print("-" * 74)
    for r in RESULTS:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r.get("ok")]
        print(f"[{mark}] {r['check']:<22} {r.get('seconds', 0):>6}s")
        for key, value in r.items():
            if key in ("check", "ok", "seconds", "kind", "traceback"):
                continue
            text = str(value)
            print(f"        {key}: {text[:150]}")
    print("-" * 74)
    print(verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
