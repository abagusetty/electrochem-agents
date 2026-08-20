"""
Potential-response benchmarking: does the model get the CHANGE right?

MOTIVATION
----------
NeuralPLexer3 (NeurIPS 2025) built a new benchmark, ConfBench, because the
existing ones did not measure the thing that mattered for their application:
ligand-INDUCED conformational change. A model can score well on absolute
structure prediction while being blind to the apo->holo transition, and that
blindness is invisible to any absolute-accuracy metric.

The same hole exists here, and it is worse, because response to potential is
the entire reason a constant-potential model exists. A potential-conditioned
MLIP can be trained to low force MAE at every potential in its training set
and still have learned essentially no dependence on the conditioning variable:
absolute barriers land near the ensemble mean at each potential, energy/force
MAE looks excellent, and dG/dU -- the quantity every electrochemical
conclusion rests on -- is wrong. Absolute-accuracy metrics cannot see this.

So: measure the response, not the value.

    d_obs(U1 -> U2)  vs  d_ref(U1 -> U2)

TWO SCORES
----------
`response_score` follows ConfBench's construction (their S.11.2): a symmetric,
bounded, normalised differential score, so systems whose response is large and
systems whose response is small are directly comparable. Their form, adapted:

    score = (d_far - d_near) / sqrt( (d_far^2 + d_near^2 + d_ref^2) / 2 )

where d_near/d_far are the observed distances to the two reference states and
d_ref is the reference separation between them. +1 means the prediction sits
exactly on the correct endpoint; -1 means it sits on the wrong one; 0 means it
is uninformative about which state it is in.

`response_slope_error` is the blunter, more directly interpretable companion:
compare fitted dG/dU against the reference slope. Report both. The normalised
score is what makes different systems comparable; the slope is what an
electrochemist will actually ask about.

A MODEL THAT PASSES ABSOLUTE ACCURACY AND FAILS THIS IS NOT USABLE for
potential-dependent conclusions, however good its MAE table looks.

Geometry only; no calculator required.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalised differential score (ConfBench-style)
# ---------------------------------------------------------------------------

def response_score(d_near: float, d_far: float, d_ref: float) -> float:
    """Bounded, normalised score for "did it move the right way?".

    d_near : observed distance to the state it SHOULD resemble
    d_far  : observed distance to the other reference state
    d_ref  : reference separation between the two states

    Normalisation by the RMS of all three distances is what makes the score
    comparable across systems whose responses differ in magnitude -- a 0.05 eV
    shift captured perfectly should score as well as a 0.5 eV shift captured
    perfectly. Without it, big-response systems dominate any average.
    """
    denominator = np.sqrt(0.5 * (d_far ** 2 + d_near ** 2 + d_ref ** 2))
    if denominator < 1e-12:
        return 0.0
    return float((d_far - d_near) / denominator)


@dataclass
class ResponseResult:
    quantity: str
    u_low_v: float
    u_high_v: float
    observed_delta: float
    reference_delta: float
    score_low: float
    score_high: float
    absolute_error_low: float
    absolute_error_high: float

    @property
    def mean_score(self) -> float:
        return 0.5 * (self.score_low + self.score_high)

    @property
    def captured_sign(self) -> bool:
        """Did the model even get the DIRECTION of the response right?

        The weakest possible bar. Failing it means the model is worse than
        useless for potential-dependent conclusions -- it would point an
        experimentalist the wrong way.
        """
        if abs(self.reference_delta) < 1e-9:
            return True
        return (self.observed_delta * self.reference_delta) > 0

    @property
    def relative_magnitude(self) -> Optional[float]:
        """observed / reference. 1.0 is perfect; 0.1 means the model is
        ~10x under-responsive, which is the characteristic failure of a model
        that ignored its conditioning variable."""
        if abs(self.reference_delta) < 1e-9:
            return None
        return float(self.observed_delta / self.reference_delta)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"mean_score": self.mean_score,
                  "captured_sign": self.captured_sign,
                  "relative_magnitude": self.relative_magnitude})
        return d


def score_response_pair(observed_low: float, observed_high: float,
                        reference_low: float, reference_high: float,
                        u_low_v: float, u_high_v: float,
                        quantity: str = "barrier_ev") -> ResponseResult:
    """Score a model's response between two potentials for one observable.

    `observed_*` are the model's values at the two potentials; `reference_*`
    are the ground-truth values (CP-DFT, or the anchor paper).
    """
    d_ref = abs(reference_high - reference_low)

    # At U_low the prediction should be near reference_low and far from
    # reference_high; at U_high the reverse.
    result = ResponseResult(
        quantity=quantity, u_low_v=u_low_v, u_high_v=u_high_v,
        observed_delta=observed_high - observed_low,
        reference_delta=reference_high - reference_low,
        score_low=response_score(abs(observed_low - reference_low),
                                 abs(observed_low - reference_high), d_ref),
        score_high=response_score(abs(observed_high - reference_high),
                                  abs(observed_high - reference_low), d_ref),
        absolute_error_low=abs(observed_low - reference_low),
        absolute_error_high=abs(observed_high - reference_high),
    )
    return result


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

@dataclass
class SlopeResult:
    quantity: str
    observed_slope: float          # e.g. d(barrier)/dU, eV/V
    reference_slope: float
    n_points: int
    observed_intercept: float = 0.0
    reference_intercept: float = 0.0
    observed_r2: Optional[float] = None

    @property
    def slope_error(self) -> float:
        return self.observed_slope - self.reference_slope

    @property
    def relative_slope_error(self) -> Optional[float]:
        if abs(self.reference_slope) < 1e-9:
            return None
        return float(self.slope_error / self.reference_slope)

    @property
    def responsive(self) -> bool:
        """Did the model learn ANY dependence on potential?

        Guards the specific failure this module exists to catch: a model that
        fit every training potential to low MAE by predicting near the
        ensemble mean, so its slope is ~0 while its MAE table looks fine.
        """
        if abs(self.reference_slope) < 1e-9:
            return True
        return abs(self.observed_slope) > 0.2 * abs(self.reference_slope)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"slope_error": self.slope_error,
                  "relative_slope_error": self.relative_slope_error,
                  "responsive": self.responsive})
        return d


def fit_response_slope(potentials_v: Sequence[float],
                       observed: Sequence[float],
                       reference: Sequence[float],
                       quantity: str = "barrier_ev") -> SlopeResult:
    """Linear fit of observable vs potential, model and reference."""
    u = np.asarray(potentials_v, dtype=float)
    y = np.asarray(observed, dtype=float)
    r = np.asarray(reference, dtype=float)
    if not (u.size == y.size == r.size) or u.size < 2:
        raise ValueError(
            "Need >= 2 matched (potential, observed, reference) triples of "
            f"equal length; got {u.size}/{y.size}/{r.size}.")

    obs_slope, obs_intercept = np.polyfit(u, y, 1)
    ref_slope, ref_intercept = np.polyfit(u, r, 1)
    predicted = obs_slope * u + obs_intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = None if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot

    return SlopeResult(
        quantity=quantity, observed_slope=float(obs_slope),
        reference_slope=float(ref_slope), n_points=int(u.size),
        observed_intercept=float(obs_intercept),
        reference_intercept=float(ref_intercept), observed_r2=r2,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def potential_response_report(potentials_v: Sequence[float],
                              observed: Sequence[float],
                              reference: Sequence[float],
                              quantity: str = "barrier_ev") -> Dict[str, Any]:
    """Full response benchmark over a potential sweep.

    Reports absolute accuracy AND response quality side by side, deliberately,
    so a good MAE cannot stand in for a good slope. That separation is the
    whole point of the benchmark -- exactly as NeuralPLexer3 reports
    PoseBusters accuracy and PB-valid as two numbers rather than one.
    """
    u = np.asarray(potentials_v, dtype=float)
    y = np.asarray(observed, dtype=float)
    r = np.asarray(reference, dtype=float)

    order = np.argsort(u)
    u, y, r = u[order], y[order], r[order]

    pairs: List[ResponseResult] = []
    for i in range(len(u) - 1):
        pairs.append(score_response_pair(
            observed_low=float(y[i]), observed_high=float(y[i + 1]),
            reference_low=float(r[i]), reference_high=float(r[i + 1]),
            u_low_v=float(u[i]), u_high_v=float(u[i + 1]), quantity=quantity))

    slope = fit_response_slope(u, y, r, quantity=quantity)
    mae = float(np.mean(np.abs(y - r)))
    endpoint = pairs[0] if len(pairs) == 1 else score_response_pair(
        float(y[0]), float(y[-1]), float(r[0]), float(r[-1]),
        float(u[0]), float(u[-1]), quantity=quantity)

    mean_pair_score = float(np.mean([p.mean_score for p in pairs])) if pairs else 0.0
    sign_ok = all(p.captured_sign for p in pairs)

    if not slope.responsive:
        verdict = ("FAILS: model is unresponsive to potential (slope "
                   f"{slope.observed_slope:+.4f} vs reference "
                   f"{slope.reference_slope:+.4f}). Absolute MAE of "
                   f"{mae:.4f} eV is misleading -- this model cannot support "
                   "any potential-dependent conclusion.")
    elif not sign_ok:
        verdict = ("FAILS: response has the wrong SIGN over at least one "
                   "interval. Would point an experiment the wrong way.")
    elif mean_pair_score < 0.3:
        verdict = (f"WEAK: response direction correct but poorly resolved "
                   f"(mean pair score {mean_pair_score:+.3f}).")
    else:
        verdict = (f"PASSES: response captured (mean pair score "
                   f"{mean_pair_score:+.3f}, slope ratio "
                   f"{slope.observed_slope / slope.reference_slope:.2f}).")

    return {
        "quantity": quantity,
        "n_potentials": int(u.size),
        "potential_range_v": [float(u[0]), float(u[-1])],
        "absolute_mae_ev": mae,
        "max_absolute_error_ev": float(np.max(np.abs(y - r))),
        "mean_pair_response_score": mean_pair_score,
        "endpoint_response_score": endpoint.mean_score,
        "all_signs_correct": sign_ok,
        "slope": slope.to_dict(),
        "pairs": [p.to_dict() for p in pairs],
        "verdict": verdict,
    }


def compare_models(results_by_model: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Rank models by RESPONSE quality, not absolute accuracy.

    Ranking by MAE is what lets an unresponsive model win. Models that fail
    the responsiveness check are listed separately rather than ranked, for the
    same reason a policy that did not reproduce all targets is not ranked by
    cost in acquisition.registry.
    """
    responsive, unresponsive, wrong_sign = [], [], []
    for name, report in results_by_model.items():
        entry = {"model": name,
                 "mean_pair_response_score": report["mean_pair_response_score"],
                 "absolute_mae_ev": report["absolute_mae_ev"],
                 "slope_ratio": (report["slope"]["observed_slope"]
                                 / report["slope"]["reference_slope"]
                                 if abs(report["slope"]["reference_slope"]) > 1e-9
                                 else None),
                 "verdict": report["verdict"]}
        if not report["slope"]["responsive"]:
            unresponsive.append(entry)
        elif not report["all_signs_correct"]:
            # A model with the right response MAGNITUDE and the wrong SIGN is
            # not a mediocre model, it is an actively misleading one. Rank it
            # nowhere.
            wrong_sign.append(entry)
        else:
            responsive.append(entry)

    responsive.sort(key=lambda e: -e["mean_pair_response_score"])
    best_mae = min(results_by_model.items(),
                   key=lambda kv: kv[1]["absolute_mae_ev"])[0] if results_by_model else None
    note = ""
    if best_mae and responsive and best_mae != responsive[0]["model"]:
        note = (f"NOTE: '{best_mae}' has the lowest absolute MAE but "
                f"'{responsive[0]['model']}' captures the potential response "
                "better. Rank on response -- MAE cannot distinguish a model "
                "that learned the physics from one that learned the mean.")
    return {"ranked_by_response": responsive,
            "wrong_sign_models": wrong_sign,
            "unresponsive_models": unresponsive,
            "note": note}
