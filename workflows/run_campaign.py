"""
CLI for the constant-potential acquisition campaign on Aurora.

    # 0. Check the machine BEFORE requesting an allocation (G4).
    python -m workflows.run_campaign doctor

    # 1. Freeze the comparison. Must happen before any adaptive round.
    python -m workflows.run_campaign preregister --config campaign.yaml

    # 2. Calibrate target-mu against a reference Cu slab work function.
    python -m workflows.run_campaign calibrate --config campaign.yaml --facet 100

    # 3. Run one policy arm. Run all four to get the comparison.
    python -m workflows.run_campaign run --config campaign.yaml --policy A1_sigma_mu

    # 4. Score.
    python -m workflows.run_campaign compare --config campaign.yaml

`--dry-run` writes every PBS script and input file but submits nothing.
Rehearse a campaign that way before spending allocation; the scripts are
readable and a wrong tag replicated over 500 runs is expensive.

Nothing here has been executed on Aurora.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from acquisition.policies import (
    ForceUncertaintyPolicy, GridPolicy, RandomPolicy, enumerate_candidates,
)
from acquisition.registry import (
    PolicyRun, PreRegistration, compare_policies, write_comparison,
)
from acquisition.sigma_mu import SigmaMuConfig, SigmaMuPolicy
from cp_dft.calibration import (
    CU_WORK_FUNCTION_EV, PotentialCalibration, calibrate_from_reference_slab,
)
from cp_dft.jdftx_driver import JDFTxDriver, JDFTxDriverConfig
from cp_dft.jdftx_setup import JDFTxProtocol, build_pzc_reference
from data.harvest import harvest_jdftx_run
from hpc.aurora import AuroraConfig
from hpc.launcher import make_launcher
from hpc.paths import ProjectPaths, SoftwareStack
from workflows.campaign import Campaign, CampaignConfig

POLICIES = {
    "B0_grid": lambda cfg: GridPolicy(),
    "B1_random": lambda cfg: RandomPolicy(seed=cfg.get("seed", 0)),
    "B2_force_uncertainty": lambda cfg: ForceUncertaintyPolicy(
        sigma_force_threshold=cfg.get("tau_force", 0.05)),
    "A1_sigma_mu": lambda cfg: SigmaMuPolicy(
        SigmaMuConfig(tau_mu_ev=cfg.get("tau_mu", 0.02))),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_paths(config: Dict[str, Any]) -> ProjectPaths:
    hpc_cfg = config.get("hpc", {})
    if hpc_cfg.get("root"):
        return ProjectPaths(root=hpc_cfg["root"], project=hpc_cfg.get("project"))
    return ProjectPaths.on_flare(
        project=hpc_cfg.get("project", "UNSET"),
        campaign=config.get("name", "campaign"),
        base=hpc_cfg.get("flare_base", "/lus/flare/projects"),
    )


def build_aurora(config: Dict[str, Any]) -> AuroraConfig:
    hpc_cfg = config.get("hpc", {})
    return AuroraConfig(
        project=hpc_cfg.get("project", "UNSET"),
        queue=hpc_cfg.get("queue", "prod"),
        nodes=int(hpc_cfg.get("nodes", 1)),
        walltime=hpc_cfg.get("walltime", "01:00:00"),
        filesystems=hpc_cfg.get("filesystems", "flare:home"),
        ranks_per_node=hpc_cfg.get("ranks_per_node"),
        modules=hpc_cfg.get("modules", ["frameworks"]),
        env=hpc_cfg.get("env", {}),
        affinity_script=hpc_cfg.get("affinity_script"),
    )


def build_protocol(config: Dict[str, Any]) -> JDFTxProtocol:
    return JDFTxProtocol(**(config.get("jdftx", {}) or {}))


def build_registration(config: Dict[str, Any]) -> PreRegistration:
    space = config.get("space", {})
    budget = config.get("budget", {})
    return PreRegistration(
        campaign=config.get("name", "campaign"),
        created_utc=config.get("created_utc", "unset"),
        facets=space.get("facets", ["100", "310"]),
        controls=space.get("controls", [-4.4, -4.2, -4.0, -3.8]),
        control_kind=space.get("control_kind", "target_mu_ev"),
        cations=space.get("cations", [None, "K"]),
        dft_budget_total=int(budget.get("dft_calls", 200)),
        md_budget_ns_total=float(budget.get("md_ns", 50.0)),
        rounds=int(budget.get("rounds", 5)),
        budget_per_round=int(budget.get("per_round", 20)),
        tolerance_ev=float(config.get("tolerance_ev", 0.05)),
        notes=config.get("notes", ""),
    )


def load_calibration(paths: ProjectPaths) -> PotentialCalibration:
    path = paths.state / "potential_calibration.json"
    if not path.exists():
        return PotentialCalibration(
            note="No calibration on disk. Run `calibrate` first.")
    return PotentialCalibration.from_dict(json.loads(path.read_text()))


def build_jdftx_driver(config: Dict[str, Any], dry_run: bool) -> JDFTxDriver:
    paths, aurora = build_paths(config), build_aurora(config)
    stack = SoftwareStack.from_env()
    for key, value in (config.get("software", {}) or {}).items():
        if hasattr(stack, key) and value:
            setattr(stack, key, value)
    launcher = make_launcher(
        config.get("hpc", {}).get("launcher", "pbs"),
        aurora=aurora,
        nodes_per_task=int(config.get("hpc", {}).get("nodes_per_task", 1)),
        dry_run=dry_run,
    )
    return JDFTxDriver(JDFTxDriverConfig(
        stack=stack, paths=paths, aurora=aurora, launcher=launcher,
        use_gpu_build=config.get("hpc", {}).get("jdftx_gpu", True),
        sycl=config.get("hpc", {}).get("jdftx_sycl", True),
        atoms_per_rank=int(config.get("hpc", {}).get("atoms_per_rank", 8)),
    ))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    """Report the machine's actual state. Run this on a COMPUTE node."""
    from mlip.xpu import probe_mlip_on_xpu, report_backend

    if args.probe:
        # The G4 test: real forward + backward per stack, on device.
        stack = SoftwareStack.from_env()
        result = probe_mlip_on_xpu(
            esen_model=args.esen_model,
            cp_mace_model_path=args.cp_mace_model,
            cp_mace_repo=stack.cp_mace_repo)
        print(json.dumps(result, indent=2, default=str))
        if not result["g4_answered"]:
            print("\nG4 NOT ANSWERED -- see verdict. Nothing reached the "
                  "device; this is an environment problem, not an XPU verdict.",
                  file=sys.stderr)
            return 2
        return 0 if result["usable_stacks"] else 1

    report: Dict[str, Any] = {"backend": report_backend()}
    if args.config:
        config = load_config(args.config)
        stack = SoftwareStack.from_env()
        report["software_paths"] = {
            k: v for k, v in stack.__dict__.items() if v}
        report["paths_exist"] = stack.check_exists(
            "jdftx_bin", "jdftx_gpu_bin", "jdftx_pseudo_dir",
            "plumed_kernel", "cp_mace_repo", "model_dir")
        report["project_root"] = build_paths(config).root
    print(json.dumps(report, indent=2, default=str))

    backend = report["backend"]
    if not backend.get("xpu_available"):
        print("\nNOTE: XPU not available here. That is expected on a login "
              "node. Re-run inside a PBS job before concluding anything about "
              "gate G4 (fairchem/MACE on Intel GPUs).", file=sys.stderr)
    return 0


def cmd_preregister(args) -> int:
    config = load_config(args.config)
    paths = build_paths(config).create()
    registration = build_registration(config)
    path = registration.write(paths.state / "preregistration.json")
    print(f"Pre-registration written: {path}")
    print(f"  fingerprint: {registration.fingerprint()}")
    print(f"  primary comparison: {registration.primary_comparison}")
    print("\nThis file is now frozen. Editing it invalidates the comparison; "
          "start a new campaign directory instead.")
    return 0


def cmd_calibrate(args) -> int:
    """Run the PZC reference slab and fit the target-mu calibration."""
    config = load_config(args.config)
    paths = build_paths(config).create()
    driver = build_jdftx_driver(config, dry_run=args.dry_run)

    spec = build_pzc_reference(facet=args.facet, nx=args.nx, ny=args.ny,
                               protocol=build_protocol(config))
    print(f"Submitting PZC reference: Cu({args.facet}) {args.nx}x{args.ny}, neutral")
    submission = driver.submit_one(spec)
    submission.handle.wait(interval_s=args.poll)

    label = harvest_jdftx_run(submission.run_dir, sp_id=spec.calc_id,
                              pseudo_set=spec.protocol.pseudo_set)
    if not label.converged:
        print(f"PZC reference did NOT converge (status={label.status.value}). "
              f"Inspect {submission.run_dir}. Not writing a calibration.",
              file=sys.stderr)
        return 1

    calibration = calibrate_from_reference_slab(
        mu_pzc_ev=label.mu_ev, facet=args.facet,
        work_function_ev=args.work_function)
    out = paths.state / "potential_calibration.json"
    out.write_text(json.dumps(calibration.to_dict(), indent=2))
    print(calibration.describe())
    print(f"\nWritten: {out}")
    print("Sanity-check the sign before trusting absolute potentials: a more "
          "negative target-mu must correspond to a more negative U vs SHE.")
    return 0


def cmd_run(args) -> int:
    config = load_config(args.config)
    paths = build_paths(config).create()
    registration = build_registration(config)
    calibration = load_calibration(paths)

    if not calibration.calibrated and not args.allow_uncalibrated:
        print("REFUSING: no potential calibration on disk. Run "
              "`calibrate` first, or pass --allow-uncalibrated to proceed "
              "with RELATIVE potentials only (results not reportable as "
              "absolute).", file=sys.stderr)
        return 1

    factory = POLICIES.get(args.policy)
    if factory is None:
        print(f"Unknown policy {args.policy!r}. Choose from {sorted(POLICIES)}.",
              file=sys.stderr)
        return 1

    campaign = Campaign(
        CampaignConfig(
            name=registration.campaign, paths=paths,
            stack=SoftwareStack.from_env(), registration=registration,
            protocol=build_protocol(config), potential_calibration=calibration,
        ),
        policy=factory(config.get("policy", {}) or {}),
        jdftx=build_jdftx_driver(config, dry_run=args.dry_run),
        md_runner=None,   # wire md.cp_md_driver here once models are trained
    )

    space = config.get("space", {})
    candidates = enumerate_candidates(
        facets=registration.facets, controls=registration.controls,
        cations=registration.cations, control_kind=registration.control_kind,
        n_cation=int(space.get("n_cation", 2)),
        nx=int(space.get("nx", 8)), ny=int(space.get("ny", 8)),
    )
    print(f"Policy {args.policy}: {len(candidates)} candidate state points, "
          f"budget {registration.dft_budget_total} DFT calls / "
          f"{registration.md_budget_ns_total} ns")

    if args.status_only:
        print(json.dumps(campaign.status(), indent=2, default=str))
        return 0

    run = campaign.run(candidates, bootstrap_rounds=args.bootstrap_rounds)
    print(json.dumps(run.to_dict(), indent=2, default=str))
    return 0


def cmd_compare(args) -> int:
    config = load_config(args.config)
    paths = build_paths(config)
    registration = PreRegistration.load(paths.state / "preregistration.json")

    runs: List[PolicyRun] = []
    for name in registration.policies:
        path = paths.state / name / "policy_run.json"
        if not path.exists():
            print(f"  missing arm: {name} (no {path})", file=sys.stderr)
            continue
        payload = json.loads(path.read_text())
        runs.append(PolicyRun(
            policy_name=payload["policy_name"],
            n_dft_calls=payload["n_dft_calls"],
            md_ns=payload["md_ns"],
            reproduced=payload.get("reproduced", {}),
            best_error_ev=payload.get("best_error_ev", {}),
            rounds_used=payload.get("rounds_used", 0),
            notes=payload.get("notes", ""),
        ))

    comparison = compare_policies(runs, registration)
    out = write_comparison(paths.state / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2, default=str))
    print(f"\nWritten: {out}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_campaign", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="report machine/software state")
    doctor.add_argument("--config")
    doctor.add_argument("--probe", action="store_true",
                        help="run the G4 test: real MLIP forward+backward on "
                             "device. Use on a COMPUTE node.")
    doctor.add_argument("--esen-model", default="esen-sm-conserving-all-oc25")
    doctor.add_argument("--cp-mace-model", default=None,
                        help="path to a FermiMACE .model to probe Route B")
    doctor.set_defaults(func=cmd_doctor)

    prereg = sub.add_parser("preregister", help="freeze the comparison")
    prereg.add_argument("--config", required=True)
    prereg.set_defaults(func=cmd_preregister)

    calib = sub.add_parser("calibrate", help="fit target-mu vs work function")
    calib.add_argument("--config", required=True)
    calib.add_argument("--facet", default="100", choices=sorted(CU_WORK_FUNCTION_EV))
    calib.add_argument("--nx", type=int, default=4)
    calib.add_argument("--ny", type=int, default=4)
    calib.add_argument("--work-function", type=float, default=None)
    calib.add_argument("--poll", type=float, default=60.0)
    calib.add_argument("--dry-run", action="store_true")
    calib.set_defaults(func=cmd_calibrate)

    run = sub.add_parser("run", help="run one policy arm")
    run.add_argument("--config", required=True)
    run.add_argument("--policy", required=True, choices=sorted(POLICIES))
    run.add_argument("--bootstrap-rounds", type=int, default=1)
    run.add_argument("--allow-uncalibrated", action="store_true")
    run.add_argument("--status-only", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="score all arms")
    compare.add_argument("--config", required=True)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
