"""
A1 -- the sigma_mu acquisition policy.

THE IDEA
--------
CP-MACE's own metadynamics driver already runs a committee of trained models
and reports both force standard deviation AND variation in chemical potential
across members. That second quantity, sigma_mu, is disagreement about the
ELECTRONIC BOUNDARY CONDITION -- about what potential the system is at.

A constant-charge committee cannot produce it: with the charge fixed there is
no mu for members to disagree about. So sigma_mu is an uncertainty signal
native to constant-potential simulation, and the proposal is to use it to
decide when a configuration has earned an expensive grand-canonical DFT label.

    High sigma_F  -> "the models disagree about the forces here."
                     Often fixable by more sampling. Cheap remedies first.
    High sigma_mu -> "the models disagree about the potential here."
                     No amount of MLIP sampling fixes that. Only DFT resolves
                     the electronic boundary condition. Spend the label.

CALIBRATION IS MANDATORY
------------------------
An uncertainty is not an acquisition signal until it predicts error. The
threshold tau_mu must be fitted against REALISED error -- |mu_committee -
mu_JDFTx| on labelled points -- not guessed. `ThresholdCalibration` does that
and reports whether sigma_mu actually correlates with error. If it does not,
the policy says so instead of quietly selecting noise.

This mirrors how NeuralPLexer3 (NeurIPS 2025) trains a confidence module
against realised structural error (their Algorithm S4) rather than trusting
raw model variance -- the same discipline, applied to mu.

FALSIFIABILITY
--------------
This policy is only interesting if it beats acquisition.policies
.ForceUncertaintyPolicy (B2), which is this policy with the sigma_mu term
removed. Run both. Report the gap either way.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from acquisition.policies import AcquisitionDecision, AcquisitionPolicy, Candidate
from data.schema import CPDFTLabel, MDResult, state_point_id


@dataclass
class SigmaMuConfig:
    """Policy thresholds and weights.

    Defaults are PLACEHOLDERS. tau_mu especially must come from
    ThresholdCalibration on real labels before any production round -- a
    guessed threshold either spends the whole budget on noise or never
    triggers at all, and both look like "the policy did something".
    """

    tau_mu_ev: float = 0.02             # sigma_mu above this -> DFT audit
    tau_force_ev_per_angstrom: float = 0.05
    # Ranking weights once candidates are past threshold.
    w_sigma_mu: float = 1.0
    w_sigma_force: float = 0.2          # deliberately small: sigma_F is B2's job
    w_novelty: float = 0.3              # spread over the state-point space
    w_mu_tracking_error: float = 0.5    # runs that drifted off targetmu
    # Do not spend more than this fraction of a round's budget on frames from
    # one trajectory. Without it, one pathological trajectory monopolises the
    # round and the campaign stops exploring.
    max_fraction_per_trajectory: float = 0.4
    min_selected: int = 1
    require_calibration: bool = True

    # HARD physical veto, after NeuralPLexer3's sample ranking (S.9), which
    # scores a ligand as `pLDDT - 1000 * (is_clash + is_chirality_violation)`:
    # physical validity is a veto, not a weighted term. The same logic applies
    # here. A trajectory that left physical validity (water inside the slab,
    # dissociated solvent, collapsed slab) produces configurations whose DFT
    # label describes a system nobody asked about -- high sigma_mu on a broken
    # geometry is the model correctly reporting that the geometry is broken,
    # and paying for a label there is the worst possible use of the budget.
    # A soft weight would let an extreme sigma_mu outbid the penalty; -1000
    # cannot be outbid.
    veto_penalty: float = 1000.0
    min_if_valid_fraction: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdCalibration:
    """Fit of sigma_mu against realised DFT error.

    `spearman` and `auc` answer the only question that matters: does high
    sigma_mu actually mark configurations the MLIP gets wrong? If not, the
    policy has no basis and the honest move is to report that.
    """

    tau_mu_ev: float
    n_points: int
    spearman: Optional[float] = None
    pearson: Optional[float] = None
    auc: Optional[float] = None            # ranking AUC vs "error above median"
    mean_error_above_tau: Optional[float] = None
    mean_error_below_tau: Optional[float] = None
    calibrated: bool = False
    note: str = ""

    # Stratified discrimination. See `calibrate_threshold`.
    #   between_spearman -- does sigma_mu rank ERROR across DIFFERENT state
    #                       points? (easy; a committee usually gets this)
    #   within_spearman  -- does it rank error across FRAMES OF THE SAME
    #                       trajectory? (hard; this is what frame selection
    #                       actually needs)
    between_spearman: Optional[float] = None
    within_spearman: Optional[float] = None
    n_within_groups: int = 0

    @property
    def frame_selection_valid(self) -> bool:
        """Whether `select_frames` is justified.

        NeuralPLexer3 (NeurIPS 2025, S.8) reports exactly the failure this
        guards against: their confidence scores correlated well with accuracy
        ACROSS diverse targets while being nearly insensitive to different
        sampled conformations of the SAME topology. They attribute it to
        under-training plus insufficient paired conformational data per
        topology, and fix it by generating multiple hypotheses per topology and
        adding a contrastive objective.

        The analogue here is direct and dangerous: sigma_mu can look strongly
        predictive because Cu(100) and Cu(310) differ, while carrying no
        information about which FRAME of one trajectory deserves a DFT audit.
        Between-group signal justifies picking state points; only within-group
        signal justifies picking frames. Conflating them spends the audit
        budget on noise while the calibration report looks healthy.
        """
        return (self.calibrated and self.n_within_groups >= 3
                and self.within_spearman is not None
                and self.within_spearman >= 0.25)

    @property
    def is_informative(self) -> bool:
        """Whether sigma_mu carries usable signal.

        Deliberately strict. A policy built on an uninformative signal is
        worse than no policy: it spends real allocation and produces a
        confident-looking result.
        """
        if not self.calibrated or self.n_points < 8:
            return False
        if self.spearman is None:
            return False
        if self.spearman < 0.3:
            return False
        if (self.mean_error_above_tau is not None
                and self.mean_error_below_tau is not None):
            return self.mean_error_above_tau > 1.5 * self.mean_error_below_tau
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_informative"] = self.is_informative
        d["frame_selection_valid"] = self.frame_selection_valid
        return d

    def describe(self) -> str:
        if not self.calibrated:
            return f"ThresholdCalibration [UNCALIBRATED] tau_mu={self.tau_mu_ev:.4f} eV"
        verdict = "INFORMATIVE" if self.is_informative else "NOT INFORMATIVE"
        frames = "OK" if self.frame_selection_valid else "NOT JUSTIFIED"
        def _f(x):
            return "n/a" if x is None else f"{x:+.3f}"
        return (
            f"ThresholdCalibration [{verdict}] n={self.n_points}\n"
            f"  tau_mu           = {self.tau_mu_ev:.4f} eV\n"
            f"  spearman (all)   = {_f(self.spearman)}\n"
            f"  between-group    = {_f(self.between_spearman)}   "
            f"(justifies STATE-POINT selection)\n"
            f"  within-group     = {_f(self.within_spearman)}   "
            f"(justifies FRAME selection) -> {frames}"
            f"  [{self.n_within_groups} group(s)]\n"
            f"  mean |err| above/below tau = "
            f"{self.mean_error_above_tau:.4f} / {self.mean_error_below_tau:.4f} eV\n"
            f"  {self.note}"
        )


def calibrate_threshold(sigma_mu_values: Sequence[float],
                        realised_error_ev: Sequence[float],
                        target_recall: float = 0.8,
                        groups: Optional[Sequence[str]] = None
                        ) -> ThresholdCalibration:
    """Fit tau_mu so the policy catches `target_recall` of the worst errors.

    `realised_error_ev[i]` is |mu_committee - mu_JDFTx| for a labelled
    configuration whose committee sigma_mu was `sigma_mu_values[i]`. "Worst"
    means above the median error.

    `groups[i]` names the state point (or trajectory) each point came from.
    PASS IT. With groups the fit is stratified into:

      * BETWEEN-group -- correlation of per-group mean sigma_mu against
        per-group mean error. Justifies choosing which STATE POINTS to audit.
      * WITHIN-group  -- correlation computed after removing each group's mean
        from both quantities. Justifies choosing which FRAMES to audit.

    The distinction is not pedantic. A committee will usually show strong
    between-group signal simply because Cu(100) and Cu(310) are different
    problems, while carrying no within-trajectory information at all -- the
    exact failure NeuralPLexer3 diagnoses for confidence heads (S.8). Without
    stratifying, an aggregate Spearman of 0.9 can hide a within-group Spearman
    of 0.0, and `select_frames` would then spend the audit budget on noise.

    Returns a calibration whose `is_informative` flag gates state-point
    selection and whose `frame_selection_valid` flag gates frame selection.
    """
    sigma = np.asarray(sigma_mu_values, dtype=float)
    error = np.asarray(realised_error_ev, dtype=float)
    if sigma.size != error.size:
        raise ValueError("sigma_mu_values and realised_error_ev must be equal length.")
    if sigma.size < 4:
        return ThresholdCalibration(
            tau_mu_ev=float(np.median(sigma)) if sigma.size else 0.02,
            n_points=int(sigma.size), calibrated=False,
            note=("Too few labelled points to calibrate (need >= 4, ideally "
                  ">= 20). Run the bootstrap grid first."))

    finite = np.isfinite(sigma) & np.isfinite(error)
    sigma, error = sigma[finite], error[finite]
    if sigma.size < 4 or np.allclose(sigma, sigma[0]):
        return ThresholdCalibration(
            tau_mu_ev=float(np.median(sigma)), n_points=int(sigma.size),
            calibrated=False,
            note="sigma_mu has no spread -- committee members are too similar. "
                 "Retrain members with different seeds/splits.")

    pearson = float(np.corrcoef(sigma, error)[0, 1])
    spearman = float(np.corrcoef(_rank(sigma), _rank(error))[0, 1])

    between_rho, within_rho, n_groups = _stratified_spearman(
        sigma, error, [groups[i] for i in np.flatnonzero(finite)] if groups else None)

    # tau at the quantile that captures target_recall of high-error points.
    high = error > np.median(error)
    tau = (float(np.quantile(sigma[high], 1.0 - target_recall))
           if high.sum() >= 2 else float(np.median(sigma)))

    above, below = sigma >= tau, sigma < tau
    return ThresholdCalibration(
        tau_mu_ev=tau,
        n_points=int(sigma.size),
        spearman=spearman,
        pearson=pearson,
        between_spearman=between_rho,
        within_spearman=within_rho,
        n_within_groups=n_groups,
        auc=_ranking_auc(sigma, high),
        mean_error_above_tau=float(error[above].mean()) if above.any() else None,
        mean_error_below_tau=float(error[below].mean()) if below.any() else None,
        calibrated=True,
        note=(f"tau set to capture ~{target_recall:.0%} of above-median errors "
              f"({int(above.sum())}/{sigma.size} candidates would trigger)."),
    )


def _stratified_spearman(sigma: np.ndarray, error: np.ndarray,
                         groups: Optional[Sequence[str]]
                         ) -> Tuple[Optional[float], Optional[float], int]:
    """Split rank correlation into between-group and within-group parts.

    Within-group is computed on group-mean-centred values, so it measures
    whether sigma_mu orders error AMONG members of the same group, with the
    easy between-group variance removed.
    """
    if not groups or len(groups) != sigma.size:
        return None, None, 0

    keys = list(dict.fromkeys(groups))
    index = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    usable = [k for k in keys if len(index[k]) >= 3]

    between = None
    if len(keys) >= 3:
        gs = np.array([sigma[index[k]].mean() for k in keys])
        ge = np.array([error[index[k]].mean() for k in keys])
        if not (np.allclose(gs, gs[0]) or np.allclose(ge, ge[0])):
            between = float(np.corrcoef(_rank(gs), _rank(ge))[0, 1])

    within = None
    if usable:
        cs, ce = [], []
        for k in usable:
            idx = index[k]
            cs.append(sigma[idx] - sigma[idx].mean())
            ce.append(error[idx] - error[idx].mean())
        cs, ce = np.concatenate(cs), np.concatenate(ce)
        if not (np.allclose(cs, 0) or np.allclose(ce, 0)):
            within = float(np.corrcoef(_rank(cs), _rank(ce))[0, 1])

    return between, within, len(usable)


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _ranking_auc(scores: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rank(scores) + 1.0
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def realised_errors_from_labels(labels: Sequence[CPDFTLabel],
                                committee_mu_by_key: Dict[str, float]
                                ) -> Tuple[List[float], List[float], List[str]]:
    """Pair up committee sigma_mu with DFT-realised error.

    Returns (sigma_mu, |mu_committee - mu_dft|, keys) over labels that are
    usable AND have a recorded committee prediction. Unconverged labels are
    excluded -- calibrating against a DFT number that never converged would
    teach the threshold to chase DFT failures rather than MLIP failures.
    """
    sigmas, errors, keys = [], [], []
    for label in labels:
        if not label.is_usable():
            continue
        predicted = committee_mu_by_key.get(label.state_point_id)
        if predicted is None or not math.isfinite(label.mu_ev):
            continue
        sigmas.append(committee_mu_by_key.get(
            f"sigma::{label.state_point_id}", float("nan")))
        errors.append(abs(predicted - label.mu_ev))
        keys.append(label.state_point_id)
    return sigmas, errors, keys


class SigmaMuPolicy(AcquisitionPolicy):
    """A1 -- select configurations by Fermi-level committee disagreement."""

    name = "A1_sigma_mu"

    def __init__(self, config: Optional[SigmaMuConfig] = None,
                 calibration: Optional[ThresholdCalibration] = None):
        self.config = config or SigmaMuConfig()
        self.calibration = calibration
        if calibration is not None and calibration.calibrated:
            self.config.tau_mu_ev = calibration.tau_mu_ev

    # -- scoring -----------------------------------------------------------

    def _novelty(self, candidate: Candidate, labelled_points: Sequence) -> float:
        """Normalised distance to the nearest already-labelled state point.

        Keeps the policy from spending an entire round inside one small,
        high-sigma region of the space.
        """
        if not labelled_points:
            return 1.0
        sp = candidate.state_point
        control = (sp.target_mu_ev if sp.target_mu_ev is not None
                   else (sp.surface_charge_uc_cm2 or 0.0))
        distances = []
        for other in labelled_points:
            other_control = (other.target_mu_ev if other.target_mu_ev is not None
                             else (other.surface_charge_uc_cm2 or 0.0))
            d = abs(control - other_control) / 1.0
            d += 0.0 if other.facet == sp.facet else 1.0
            d += 0.0 if other.cation == sp.cation else 0.5
            distances.append(d)
        return float(min(1.0, min(distances) / 2.0))

    def select(self, candidates, budget, labelled=None, md_results=None,
               labels=None, round_index=0) -> AcquisitionDecision:
        config = self.config
        fresh = self._fresh(candidates, labelled)
        md_results = list(md_results or [])

        if config.require_calibration and (
                self.calibration is None or not self.calibration.calibrated):
            return AcquisitionDecision(
                round_index=round_index, policy_name=self.name, selected=[],
                considered=len(fresh), budget=budget,
                notes=("REFUSED: tau_mu is not calibrated against realised DFT "
                       "error. Bootstrap with GridPolicy, call "
                       "calibrate_threshold(), then re-run. Selecting on an "
                       "uncalibrated threshold would spend allocation on noise."),
            )
        if (self.calibration is not None and self.calibration.calibrated
                and not self.calibration.is_informative):
            return AcquisitionDecision(
                round_index=round_index, policy_name=self.name, selected=[],
                considered=len(fresh), budget=budget,
                notes=("REFUSED: sigma_mu is calibrated but NOT informative "
                       f"(spearman={self.calibration.spearman}). This is a "
                       "publishable negative result for C3 -- report it and "
                       "fall back to B2 rather than proceeding."),
            )

        stats = _stats_by_state_point(md_results)
        labelled_points = [r.state_point_id for r in md_results]
        del labelled_points  # scoring uses state points, resolved by caller

        for candidate in fresh:
            sid = state_point_id(candidate.state_point)
            entry = stats.get(sid, {})
            sigma_mu = entry.get("sigma_mu", 0.0)
            sigma_force = entry.get("sigma_force", 0.0)
            mu_drift = entry.get("mu_tracking_error", 0.0)
            novelty = self._novelty(candidate, [])
            if_valid = entry.get("if_valid_fraction")

            candidate.score = (
                config.w_sigma_mu * (sigma_mu / max(config.tau_mu_ev, 1e-9))
                + config.w_sigma_force * (sigma_force
                                          / max(config.tau_force_ev_per_angstrom, 1e-9))
                + config.w_mu_tracking_error * (mu_drift / max(config.tau_mu_ev, 1e-9))
                + config.w_novelty * novelty
            )
            triggered = sigma_mu >= config.tau_mu_ev
            vetoed = (if_valid is not None
                      and if_valid < config.min_if_valid_fraction)
            if vetoed:
                candidate.score -= config.veto_penalty
                candidate.strategy = "vetoed_if_invalid"
            else:
                candidate.strategy = ("sigma_mu_audit" if triggered
                                      else "sigma_mu_below_threshold")
            candidate.reason = (
                f"sigma_mu={sigma_mu:.4f} eV (tau={config.tau_mu_ev:.4f}), "
                f"sigma_F={sigma_force:.4f} eV/A, "
                f"mu_drift={mu_drift:.4f} eV, novelty={novelty:.2f}"
                + ("" if if_valid is None else f", IF-valid={if_valid:.2f}")
                + (" -- VETOED: physically invalid geometry" if vetoed else ""))

        ordered = sorted(fresh, key=lambda c: (-c.score, c.key))
        # Vetoed candidates are removed outright, not merely ranked last: a
        # round with nothing else to do must spend less, not spend it on
        # broken geometry.
        allowed = [c for c in ordered if c.strategy != "vetoed_if_invalid"]
        n_vetoed = len(ordered) - len(allowed)
        triggered = [c for c in allowed if c.strategy == "sigma_mu_audit"]
        pool = triggered if len(triggered) >= config.min_selected else allowed
        selected = _cap_per_trajectory(pool, budget,
                                       config.max_fraction_per_trajectory)

        return AcquisitionDecision(
            round_index=round_index, policy_name=self.name, selected=selected,
            considered=len(fresh), budget=budget,
            notes=(f"{len(triggered)} candidate(s) above tau_mu="
                   f"{config.tau_mu_ev:.4f} eV; "
                   f"{'threshold' if triggered else 'top-score fallback'} used; "
                   f"{n_vetoed} vetoed as physically invalid; "
                   f"per-trajectory cap "
                   f"{config.max_fraction_per_trajectory:.0%}."),
        )

    # -- frame-level selection ---------------------------------------------

    def select_frames(self, sigma_mu_trace: Sequence[float], budget: int,
                      min_separation: int = 100,
                      allow_uncalibrated: bool = False) -> List[int]:
        """Pick trajectory frames to audit from a per-step sigma_mu trace.

        `min_separation` enforces temporal spacing. Consecutive MD frames are
        nearly identical configurations, so auditing a contiguous block buys
        one label's worth of information at N labels' cost.

        REFUSES unless the calibration demonstrates WITHIN-group discrimination
        (`frame_selection_valid`). Between-group signal -- sigma_mu telling
        Cu(100) apart from Cu(310) -- says nothing about which frame of one
        trajectory is worth auditing, and acting on it would spend the budget
        on noise while the headline correlation looked excellent. Falls back to
        evenly-spaced frames, which is the honest default when no frame-level
        signal has been established.
        """
        trace = np.asarray(sigma_mu_trace, dtype=float)
        if trace.size == 0 or budget <= 0:
            return []

        calibration = self.calibration
        if not allow_uncalibrated and not (
                calibration is not None and calibration.frame_selection_valid):
            n = trace.size
            return [int(n * (i + 1) / (budget + 1)) for i in range(budget)]
        chosen: List[int] = []
        for index in np.argsort(-trace):
            index = int(index)
            if trace[index] < self.config.tau_mu_ev and chosen:
                break
            if all(abs(index - picked) >= min_separation for picked in chosen):
                chosen.append(index)
            if len(chosen) >= budget:
                break
        return sorted(chosen)


def _stats_by_state_point(md_results: Sequence[MDResult]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for result in md_results:
        entry = out.setdefault(result.state_point_id,
                               {"sigma_mu": 0.0, "sigma_force": 0.0,
                                "mu_tracking_error": 0.0})
        if result.if_valid_fraction is not None:
            entry["if_valid_fraction"] = min(
                entry.get("if_valid_fraction", 1.0), result.if_valid_fraction)
        if result.committee is not None:
            entry["sigma_mu"] = max(entry["sigma_mu"], result.committee.sigma_mu_ev)
            entry["sigma_force"] = max(
                entry["sigma_force"], result.committee.sigma_force_ev_per_angstrom)
        if result.achieved_mu_std_ev is not None:
            entry["mu_tracking_error"] = max(
                entry["mu_tracking_error"], result.achieved_mu_std_ev)
    return out


def _cap_per_trajectory(candidates: Sequence[Candidate], budget: int,
                        max_fraction: float) -> List[Candidate]:
    if budget <= 0:
        return []
    cap = max(1, int(math.floor(budget * max_fraction)))
    counts: Dict[str, int] = {}
    selected: List[Candidate] = []
    for candidate in candidates:
        key = candidate.parent_trajectory or "<none>"
        if key != "<none>" and counts.get(key, 0) >= cap:
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= budget:
            break
    return selected


def make_audit_hook(policy: SigmaMuPolicy, every: int = 100):
    """An md.cp_md_driver hook that flags high-sigma_mu frames in-flight.

    Lets a long trajectory nominate its own audit frames as it runs, instead
    of post-processing the whole trace afterwards. The frames land in
    `ConstantPotentialMD.audit_frames`.
    """
    def hook(ctx) -> None:
        if ctx.step % every != 0:
            return
        if ctx.sigma_mu >= policy.config.tau_mu_ev:
            ctx.request["audit_frame"] = True
    return hook
