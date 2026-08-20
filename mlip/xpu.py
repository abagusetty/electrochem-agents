"""
Device resolution for Intel GPUs on Aurora.

Aurora has no CUDA. PyTorch reaches Intel Data Center GPU Max (Ponte Vecchio)
through the `xpu` backend. Device strings are "xpu", "xpu:0", ...

With ZE_FLAT_DEVICE_HIERARCHY=FLAT each TILE is a device, so a node exposes
12; with COMPOSITE each CARD is one device, so 6. hpc.aurora sets this.

ENVIRONMENT ASSUMPTION
----------------------
`module load frameworks` on Aurora provides the oneAPI SDK and an
ALCF-built PyTorch with XPU support, by default. Do NOT pip-install torch --
PyPI wheels are CUDA or CPU builds with no XPU backend, and installing one
into the module environment shadows the working build with a broken one.

WHAT IS AND IS NOT SETTLED (RESEARCH_PLAN G4)
---------------------------------------------
SETTLED:      torch exists, torch.xpu works, tensors compute on PVC.
NOT SETTLED:  whether fairchem eSEN/UMA and CP-MACE actually RUN there.

Those are different questions and conflating them is how an allocation gets
burned. A working torch-XPU build says nothing about op coverage for the
pieces these stacks need. Concrete risks, in rough order of likelihood:

  * cuEquivariance          CUDA-only accelerator path in MACE/fairchem;
                            must fall back to the e3nn/native path
  * torch_scatter/_sparse   historically CUDA-only wheels. Modern fairchem
                            and MACE mostly use native torch scatter
                            (index_add_, scatter_reduce) -- verify, do not assume
  * flash-attention         CUDA-only; must be off or absent
  * custom fused kernels    any .cu extension in the dependency tree
  * torch.compile           partial coverage on XPU; may need to be disabled
  * float64                 PVC fp64 is available but slow; check dtype paths

`report_backend()` checks the environment. `probe_mlip_on_xpu()` is the test
that actually decides G4: it runs a real forward AND backward for each stack
on device. Run both on a COMPUTE node before committing MLIP hours.

Nothing in this module has been run on Aurora.
"""

import os
from typing import Any, Dict, List, Optional


def torch_module():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required. On Aurora load the ALCF-provided stack "
            "(`module load frameworks`) rather than pip-installing a CUDA "
            "build, which has no XPU support."
        ) from exc
    return torch


def xpu_available() -> bool:
    try:
        torch = torch_module()
    except ImportError:
        return False
    backend = getattr(torch, "xpu", None)
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        torch = torch_module()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_count() -> int:
    if not xpu_available():
        return 0
    torch = torch_module()
    try:
        return int(torch.xpu.device_count())
    except Exception:
        return 0


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick a torch device string.

    Order: explicit `preferred` > XPU (Aurora) > CUDA > CPU. The rank-local
    index comes from PALS/MPI environment variables so that
    one-rank-per-tile launches spread across devices instead of all piling
    onto device 0 -- a failure mode that looks like "it works" on one rank
    and out-of-memories at scale.
    """
    if preferred:
        return preferred
    if xpu_available():
        count = device_count()
        return f"xpu:{local_rank() % count}" if count > 1 else "xpu"
    if cuda_available():
        torch = torch_module()
        count = torch.cuda.device_count()
        return f"cuda:{local_rank() % count}" if count > 1 else "cuda"
    return "cpu"


def local_rank() -> int:
    """Rank within the node, from whichever launcher set it.

    PALS (Aurora's mpiexec) sets PALS_LOCAL_RANKID; MPICH sets
    MPI_LOCALRANKID; torchrun sets LOCAL_RANK.
    """
    for name in ("PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "LOCAL_RANK",
                 "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def global_rank() -> int:
    for name in ("PMI_RANK", "PALS_RANKID", "RANK", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def world_size() -> int:
    for name in ("PMI_SIZE", "PALS_NRANKS", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 1


def try_import_ipex() -> Optional[Any]:
    """Import intel_extension_for_pytorch if present.

    The ALCF `frameworks` module ships PyTorch with XPU upstreamed, so IPEX is
    generally not required. Some stacks still expect the import for extra op
    coverage. Absence is not an error.
    """
    try:
        import intel_extension_for_pytorch as ipex
        return ipex
    except ImportError:
        return None


def report_backend() -> Dict[str, Any]:
    """Diagnostic to run on an Aurora compute node before committing hours.

    Reports what actually loads and where tensors can live. Import failures
    are captured as strings rather than raised, so one broken package does not
    hide the state of the others.
    """
    report: Dict[str, Any] = {
        "env": {k: os.environ.get(k) for k in
                ("ZE_FLAT_DEVICE_HIERARCHY", "ZE_AFFINITY_MASK",
                 "PALS_LOCAL_RANKID", "PBS_JOBID")},
        "local_rank": local_rank(),
        "global_rank": global_rank(),
        "world_size": world_size(),
    }
    try:
        torch = torch_module()
        report["torch_version"] = torch.__version__
        report["xpu_available"] = xpu_available()
        report["xpu_device_count"] = device_count()
        report["cuda_available"] = cuda_available()
        report["resolved_device"] = resolve_device()
        if report["xpu_available"]:
            try:
                report["xpu_device_name"] = torch.xpu.get_device_name(0)
            except Exception as exc:
                report["xpu_device_name"] = f"<error: {exc}>"
            # The real test: can a tensor round-trip and compute on device?
            try:
                t = torch.ones(1024, 1024, device=report["resolved_device"])
                report["xpu_matmul_ok"] = bool(float((t @ t).sum()) > 0)
            except Exception as exc:
                report["xpu_matmul_ok"] = f"<error: {exc}>"
    except ImportError as exc:
        report["torch"] = f"<not importable: {exc}>"

    ipex = try_import_ipex()
    report["ipex_version"] = getattr(ipex, "__version__", None) if ipex else None

    for name in ("fairchem.core", "mace", "e3nn", "torch_scatter"):
        try:
            module = __import__(name, fromlist=["__version__"])
            report[f"{name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            report[f"{name}_version"] = f"<not importable: {type(exc).__name__}>"
    return report


def print_backend_report() -> None:
    import json
    print(json.dumps(report_backend(), indent=2, default=str))


# ---------------------------------------------------------------------------
# G4: does the MLIP stack actually run on XPU?
# ---------------------------------------------------------------------------

def _tiny_cu_water(n_water: int = 8):
    """Small Cu slab + water, as an ASE Atoms. Big enough to exercise the
    message-passing/scatter path, small enough to run anywhere."""
    import numpy as np
    from ase import Atoms

    a = 3.615
    cu = [[x * a / 2, y * a / 2, 6.0 - i * a / 2]
          for i in range(2) for x in range(3) for y in range(3)]
    rng = np.random.default_rng(0)
    water, symbols = [], ["Cu"] * len(cu)
    for _ in range(n_water):
        o = np.array([rng.uniform(0, 3 * a / 2), rng.uniform(0, 3 * a / 2),
                      rng.uniform(9.0, 13.0)])
        water += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
        symbols += ["O", "H", "H"]
    positions = np.array(list(cu) + list(water))
    return Atoms(symbols, positions=positions,
                 cell=[3 * a / 2, 3 * a / 2, 20.0], pbc=True)


def _probe_calculator(name: str, build_calc, device: str) -> Dict[str, Any]:
    """Run one calculator through energy AND forces on `device`.

    Forces matter more than energy: they exercise the backward pass, which is
    where a missing XPU autograd kernel shows up. An energy-only probe passes
    on stacks that cannot actually run MD.
    """
    import time
    entry: Dict[str, Any] = {"stack": name, "device": device}
    try:
        atoms = _tiny_cu_water()
        entry["n_atoms"] = len(atoms)
    except Exception as exc:
        return {**entry, "ok": False, "stage": "build_atoms",
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        atoms.calc = build_calc(device)
    except Exception as exc:
        return {**entry, "ok": False, "stage": "load_model",
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        start = time.time()
        energy = float(atoms.get_potential_energy())
        entry["energy_ev"] = energy
        entry["energy_s"] = round(time.time() - start, 3)
    except Exception as exc:
        return {**entry, "ok": False, "stage": "forward",
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        start = time.time()
        forces = atoms.get_forces()
        entry["max_force_ev_per_angstrom"] = float(abs(forces).max())
        entry["forces_s"] = round(time.time() - start, 3)
    except Exception as exc:
        return {**entry, "ok": False, "stage": "backward_forces",
                "error": f"{type(exc).__name__}: {exc}"}

    # Fermi level: required by CP-MACE's NoseHoover integrator.
    getter = getattr(atoms.calc, "get_mu", None)
    if callable(getter):
        try:
            entry["mu_ev"] = float(getter())
        except Exception as exc:
            entry["mu_ev"] = f"<error: {type(exc).__name__}>"
    entry["ok"] = True
    return entry


def probe_mlip_on_xpu(esen_model: str = "esen-sm-conserving-all-oc25",
                      cp_mace_model_path: Optional[str] = None,
                      cp_mace_repo: Optional[str] = None,
                      device: Optional[str] = None) -> Dict[str, Any]:
    """THE G4 TEST. Run on an Aurora COMPUTE node.

    `report_backend()` only proves torch works. This proves the models do.
    Each stack is probed independently and failures are captured, not raised,
    so one broken stack still yields a verdict on the other -- which matters,
    because Route A (eSEN mu-head) and Route B (FermiMACE) are separate arms
    and either alone is enough to proceed.

    A stack that fails at `backward_forces` but passes `forward` is the
    characteristic missing-autograd-kernel signature: it will look loadable
    and be useless for MD.
    """
    device = device or resolve_device()
    results: List[Dict[str, Any]] = []

    def build_esen(dev):
        from mlip.esen_oc25 import ESENOC25Config, load_esen_oc25_calculator
        return load_esen_oc25_calculator(
            ESENOC25Config(model_name=esen_model, device=dev))

    results.append(_probe_calculator("fairchem-esen-oc25", build_esen, device))

    if cp_mace_model_path:
        def build_cpmace(dev):
            import sys
            if cp_mace_repo and cp_mace_repo not in sys.path:
                sys.path.insert(0, cp_mace_repo)
            from mace.calculators import MACECalculator
            return MACECalculator(model_paths=[cp_mace_model_path], device=dev)
        results.append(_probe_calculator("cp-mace-fermimace",
                                         build_cpmace, device))
    else:
        results.append({"stack": "cp-mace-fermimace", "ok": None,
                        "skipped": "pass cp_mace_model_path= to probe "
                                   "(e.g. the upstream MACE_model_compiled_1.model)"})

    usable = [r["stack"] for r in results if r.get("ok")]
    # Distinguish "the model cannot run on this device" from "this environment
    # is not set up". Reporting a missing ASE install as G4 FAIL would send the
    # MLIP work to another machine for no reason -- the exact wrong decision.
    ENV_STAGES = {"build_atoms", "load_model"}
    device_failures = [r for r in results
                       if r.get("ok") is False and r.get("stage") not in ENV_STAGES]
    env_failures = [r for r in results
                    if r.get("ok") is False and r.get("stage") in ENV_STAGES]

    if usable:
        verdict = (f"G4 PASS for: {', '.join(usable)}. "
                   "Co-located acquisition loop is viable on Aurora.")
    elif device_failures:
        stages = {r["stack"]: r.get("stage") for r in device_failures}
        verdict = (f"G4 FAIL on device ({stages}). A failure at "
                   "'backward_forces' with a clean 'forward' means a missing "
                   "XPU autograd kernel -- the model loads and is useless for "
                   "MD. Fall back to MLIP inference on NVIDIA with JDFTx "
                   "labelling on Aurora. That costs the co-located loop "
                   "(RESEARCH_PLAN 4.5), not the project.")
    elif env_failures:
        stages = {r["stack"]: (r.get("stage"), str(r.get("error"))[:80])
                  for r in env_failures}
        verdict = (f"NOT A G4 ANSWER -- environment incomplete ({stages}). "
                   "Nothing reached the device. Install/enable the stack "
                   "(module load frameworks; the fairchem and CP-MACE envs) "
                   "and re-run on a COMPUTE node. Do NOT read this as XPU "
                   "unsupported.")
    else:
        verdict = "Inconclusive -- nothing was actually probed."

    return {"device": device, "backend": report_backend(),
            "probes": results, "usable_stacks": usable,
            "device_failures": [r["stack"] for r in device_failures],
            "environment_failures": [r["stack"] for r in env_failures],
            "g4_answered": bool(usable or device_failures),
            "verdict": verdict}


def print_mlip_probe(**kwargs) -> None:
    import json
    print(json.dumps(probe_mlip_on_xpu(**kwargs), indent=2, default=str))



if __name__ == "__main__":
    print_backend_report()
