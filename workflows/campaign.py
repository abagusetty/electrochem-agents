"""
End-to-end campaign driver: setup -> run -> harvest -> decide -> repeat.

One `Campaign` object owns one pre-registered comparison on one filesystem
root, and can be resumed from that root alone. The loop:

    round 0   bootstrap with the grid policy (B0) so there ARE labels to
              calibrate a threshold against
    calibrate fit tau_mu against realised |mu_MLIP - mu_JDFTx|, and check
              whether sigma_mu is informative at all
    round n   policy selects candidates -> MD -> harvest -> CP-DFT audit of
              flagged frames -> harvest labels -> retrain/extend
    score     cost-to-reproduce vs the frozen targets, per policy

The same driver runs every policy (B0/B1/B2/A1), which is what makes the
comparison fair: identical machinery, identical budget accounting, identical
harvesting, only the `select` call differs.

Nothing in this file has been executed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from acquisition.policies import (
    AcquisitionDecision, AcquisitionPolicy, Candidate, GridPolicy,
)
from acquisition.registry import PolicyRun, PreRegistration, score_reproduction
from acquisition.sigma_mu import (
    SigmaMuPolicy, ThresholdCalibration, calibrate_threshold,
)
from cp_dft.calibration import PotentialCalibration, UNCALIBRATED
from cp_dft.jdftx_driver import JDFTxDriver, JDFTxDriverConfig, summarise_labels
from cp_dft.jdftx_setup import JDFTxCalcSpec, JDFTxProtocol
from data.schema import CPDFTLabel, MDResult, StatePoint, state_point_id
from data.store import LabelStore, RecordStore, JSONLStore
from hpc.paths import ProjectPaths, SoftwareStack


@dataclass
class CampaignConfig:
    name: str
    paths: ProjectPaths
    stack: SoftwareStack
    registration: PreRegistration
    protocol: JDFTxProtocol = field(default_factory=JDFTxProtocol)
    potential_calibration: PotentialCalibration = UNCALIBRATED
    # Frames per flagged trajectory handed to grand-canonical audit.
    audit_frames_per_trajectory: int = 4
    min_frame_separation: int = 100
    # Stop when every target is reproduced, rather than burning the full
    # budget. The metric is cost-to-reproduce, so continuing past success
    # would only inflate the number.
    stop_when_reproduced: bool = True


class Campaign:
    """Owns campaign state; one instance per policy arm."""

    def __init__(self, config: CampaignConfig, policy: AcquisitionPolicy,
                 jdftx: Optional[JDFTxDriver] = None,
                 md_runner: Optional[Any] = None):
        self.config = config
        self.policy = policy
        self.jdftx = jdftx
        # md_runner(state_point, run_dir, committee) -> MDResult.
        # Injected so the campaign can be exercised against a stub without a
        # GPU, and so the CP-MD driver stays swappable.
        self.md_runner = md_runner

        config.paths.create()
        arm = config.paths.state / policy.name
        arm.mkdir(parents=True, exist_ok=True)
        self.arm_dir = arm

        self.records = RecordStore(arm / "md_results.jsonl")
        self.labels = LabelStore(arm / "cp_dft_labels.jsonl")
        self.decisions = JSONLStore(arm / "decisions.jsonl")

        self.calibration: Optional[ThresholdCalibration] = None
        self._committee_mu: Dict[str, float] = {}

        config.registration.write(config.paths.state / "preregistration.json")

    # -- budget ------------------------------------------------------------

    @property
    def dft_calls_used(self) -> int:
        return len(self.labels.labels())

    @property
    def md_ns_used(self) -> float:
        return sum(r.sampled_ns for r in self.records.results())

    def budget_remaining(self) -> Dict[str, float]:
        reg = self.config.registration
        return {
            "dft_calls": max(0, reg.dft_budget_total - self.dft_calls_used),
            "md_ns": max(0.0, reg.md_budget_ns_total - self.md_ns_used),
        }

    def budget_exhausted(self) -> bool:
        remaining = self.budget_remaining()
        return remaining["dft_calls"] <= 0 or remaining["md_ns"] <= 0

    # -- stages ------------------------------------------------------------

    def run_md_for(self, candidates: Sequence[Candidate]) -> List[MDResult]:
        """MD at each selected state point."""
        if self.md_runner is None:
            raise RuntimeError(
                "No md_runner injected. Provide a callable "
                "(state_point, run_dir) -> MDResult, e.g. one wrapping "
                "md.cp_md_driver.run_constant_potential_md."
            )
        results = []
        for candidate in candidates:
            sp = candidate.state_point
            sid = state_point_id(sp)
            run_dir = self.config.paths.run_dir(sid)
            result = self.md_runner(sp, run_dir)
            self.records.append_result(result, state_point=sp)
            if result.committee is not None and result.committee.mean_mu_ev is not None:
                self._committee_mu[sid] = result.committee.mean_mu_ev
            results.append(result)
        return results

    def audit_with_cp_dft(self, md_results: Sequence[MDResult],
                          budget: int) -> List[CPDFTLabel]:
        """Grand-canonical DFT on the frames the policy flagged.

        For SigmaMuPolicy the frames come from the per-step sigma_mu trace;
        for the other policies they are evenly spaced, which is the honest
        equivalent when no frame-level signal exists.
        """
        if self.jdftx is None:
            raise RuntimeError("No JDFTxDriver configured; cannot audit.")
        if budget <= 0:
            return []

        sp_map = self.records.state_point_map()
        specs: List[JDFTxCalcSpec] = []
        per_traj = max(1, min(self.config.audit_frames_per_trajectory,
                              budget // max(1, len(md_results))))

        for result in md_results:
            sp = sp_map.get(result.state_point_id)
            if sp is None or result.trajectory_path is None:
                continue
            frames = self._frames_to_audit(result, per_traj)
            for index in frames:
                specs.append(JDFTxCalcSpec(
                    state_point=sp, protocol=self.config.protocol,
                    target_mu_hartree=(
                        self.config.potential_calibration.mu_ev_to_target_mu_hartree(
                            sp.target_mu_ev) if sp.target_mu_ev is not None else None),
                    frame_index=int(index),
                    parent_trajectory=result.trajectory_path,
                ))
            if len(specs) >= budget:
                break

        specs = specs[:budget]
        if not specs:
            return []

        labels = self.jdftx.run_and_harvest(specs, parallel=True)
        for label in labels:
            self.labels.append_label(label)
        return labels

    def _frames_to_audit(self, result: MDResult, count: int) -> List[int]:
        sigma_log = Path(result.run_dir or ".") / "sigma.jsonl"
        if isinstance(self.policy, SigmaMuPolicy) and sigma_log.exists():
            trace, steps = [], []
            for line in sigma_log.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                trace.append(entry.get("sigma_mu_ev", 0.0))
                steps.append(entry.get("step", 0))
            picked = self.policy.select_frames(
                trace, count, min_separation=self.config.min_frame_separation)
            return [steps[i] for i in picked if i < len(steps)]

        # No frame-level signal: spread evenly over the trajectory.
        n = max(1, result.n_steps)
        return [int(n * (i + 1) / (count + 1)) for i in range(count)]

    def calibrate(self) -> Optional[ThresholdCalibration]:
        """Fit tau_mu against realised DFT error and report informativeness."""
        sigmas, errors = [], []
        stats = {r.state_point_id: r.committee for r in self.records.results()}
        for label in self.labels.usable_labels():
            committee = stats.get(label.state_point_id)
            predicted = self._committee_mu.get(label.state_point_id)
            if committee is None or predicted is None:
                continue
            sigmas.append(committee.sigma_mu_ev)
            errors.append(abs(predicted - label.mu_ev))

        if len(sigmas) < 4:
            return None
        self.calibration = calibrate_threshold(sigmas, errors)
        if isinstance(self.policy, SigmaMuPolicy):
            self.policy.calibration = self.calibration
            if self.calibration.calibrated:
                self.policy.config.tau_mu_ev = self.calibration.tau_mu_ev
        (self.arm_dir / "calibration.json").write_text(
            json.dumps(self.calibration.to_dict(), indent=2))
        return self.calibration

    # -- loop --------------------------------------------------------------

    def run_round(self, candidates: Sequence[Candidate],
                  round_index: int) -> AcquisitionDecision:
        reg = self.config.registration
        budget = min(reg.budget_per_round, int(self.budget_remaining()["dft_calls"]))

        decision = self.policy.select(
            candidates, budget,
            labelled=self.labels.labelled_ids() | self.records.attempted_ids(),
            md_results=self.records.results(),
            labels=self.labels.labels(),
            round_index=round_index,
        )
        self.decisions.append(decision.to_dict())
        if not decision.selected:
            return decision

        md_results = self.run_md_for(decision.selected)
        self.audit_with_cp_dft(md_results, budget)
        return decision

    def run(self, candidates: Sequence[Candidate],
            bootstrap_rounds: int = 1) -> PolicyRun:
        """Full campaign for this policy arm.

        Bootstrap rounds always use the grid policy: an adaptive policy has
        nothing to be adaptive about before any labels exist, and SigmaMuPolicy
        deliberately refuses to select on an uncalibrated threshold.
        """
        reg = self.config.registration
        original_policy = self.policy

        for round_index in range(reg.rounds):
            if self.budget_exhausted():
                break
            if round_index < bootstrap_rounds:
                self.policy = GridPolicy()
            else:
                self.policy = original_policy
                if self.calibration is None:
                    self.calibrate()
            self.run_round(candidates, round_index)
            if self.config.stop_when_reproduced and self.score().all_reproduced:
                break

        self.policy = original_policy
        run = self.score()
        (self.arm_dir / "policy_run.json").write_text(
            json.dumps(run.to_dict(), indent=2))
        return run

    # -- scoring -----------------------------------------------------------

    def observed_targets(self) -> Dict[str, Dict[str, float]]:
        """Map converged MD results onto the pre-registered target keys.

        Only converged results count. Scoring an unconverged barrier against
        the anchor value would let noise reproduce a target by accident.
        """
        sp_map = self.records.state_point_map()
        observed: Dict[str, Dict[str, float]] = {}
        for result in self.records.results():
            if not result.converged or result.barrier_ev is None:
                continue
            sp = sp_map.get(result.state_point_id)
            if sp is None:
                continue
            key = _target_key(sp)
            if key is None:
                continue
            observed[key] = {"barrier_ev": result.barrier_ev,
                             "reaction_energy_ev": result.reaction_energy_ev}
        return observed

    def score(self) -> PolicyRun:
        reg = self.config.registration
        scored = score_reproduction(self.observed_targets(), reg)
        return PolicyRun(
            policy_name=self.policy.name,
            n_dft_calls=self.dft_calls_used,
            md_ns=self.md_ns_used,
            reproduced=scored["reproduced"],
            best_error_ev=scored["worst_error_ev"],
            rounds_used=len(self.decisions.read_dicts()),
            notes=(self.calibration.describe() if self.calibration else
                   "no sigma_mu calibration performed"),
        )

    def status(self) -> Dict[str, Any]:
        return {
            "campaign": self.config.name,
            "policy": self.policy.name,
            "root": self.config.paths.root,
            "dft_calls_used": self.dft_calls_used,
            "md_ns_used": round(self.md_ns_used, 4),
            "budget_remaining": self.budget_remaining(),
            "labels": self.labels.summary(),
            "records": self.records.summary(),
            "calibration": (self.calibration.to_dict() if self.calibration else None),
            "reproduction": score_reproduction(self.observed_targets(),
                                               self.config.registration),
        }


def _target_key(sp: StatePoint) -> Optional[str]:
    """Map a state point onto a pre-registered target key, or None."""
    condition = sp.condition_label
    if sp.facet == "100" and condition == "neutral":
        return "cu100_neutral"
    if sp.facet == "310" and condition == "neutral":
        return "cu310_neutral"
    if sp.facet == "310" and condition.startswith("-2"):
        return "cu310_m23"
    return None
