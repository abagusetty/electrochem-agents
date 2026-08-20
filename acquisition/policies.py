"""
Acquisition baselines.

A claim that an adaptive policy is better is only as good as what it is
compared against. Three baselines, in increasing order of how hard they are
to beat:

  B0 GridPolicy              uniform sweep over (facet x potential x cation).
                             The obvious thing a careful person would do.
  B1 RandomPolicy            random subsample at matched budget. Guards
                             against the classic failure of beating a grid
                             only because the grid was badly resolved.
  B2 ForceUncertaintyPolicy  committee force disagreement as the trigger.
                             THE ABLATION THAT MATTERS: it is the sigma_mu
                             policy with the sigma_mu term removed. If A1
                             does not beat B2, the Fermi-level signal adds
                             nothing over ordinary MLIP uncertainty.

All policies share one interface, take the same budget, and are evaluated on
the same metric (cost to reproduce the anchor paper's values within
tolerance), so the comparison is apples to apples by construction.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from data.schema import CPDFTLabel, MDResult, StatePoint, state_point_id


@dataclass
class Candidate:
    """A state point (optionally a specific frame) that could be labelled."""
    state_point: StatePoint
    frame_index: Optional[int] = None
    parent_trajectory: Optional[str] = None
    score: float = 0.0
    reason: str = ""
    strategy: str = "unspecified"        # traceability tag, per A-Lab GPSS

    @property
    def key(self) -> str:
        base = state_point_id(self.state_point)
        return base if self.frame_index is None else f"{base}_f{self.frame_index:05d}"

    def to_dict(self) -> Dict[str, Any]:
        d = {"key": self.key, "score": self.score, "reason": self.reason,
             "strategy": self.strategy, "frame_index": self.frame_index,
             "parent_trajectory": self.parent_trajectory}
        d["state_point"] = self.state_point.to_dict()
        return d


@dataclass
class AcquisitionDecision:
    """What a policy chose this round, and why. Written to the campaign log so
    the sequence of decisions can be audited after the fact."""
    round_index: int
    policy_name: str
    selected: List[Candidate] = field(default_factory=list)
    considered: int = 0
    budget: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_index": self.round_index,
            "policy_name": self.policy_name,
            "budget": self.budget,
            "considered": self.considered,
            "n_selected": len(self.selected),
            "selected": [c.to_dict() for c in self.selected],
            "notes": self.notes,
        }


class AcquisitionPolicy:
    """Base interface.

    `select` receives everything known so far and returns at most `budget`
    candidates. Policies must not select an already-labelled key -- that
    would inflate their apparent efficiency by re-buying labels they already
    have.
    """

    name = "base"

    def select(self, candidates: Sequence[Candidate], budget: int,
               labelled: Optional[Set[str]] = None,
               md_results: Optional[Sequence[MDResult]] = None,
               labels: Optional[Sequence[CPDFTLabel]] = None,
               round_index: int = 0) -> AcquisitionDecision:
        raise NotImplementedError

    @staticmethod
    def _fresh(candidates: Sequence[Candidate],
               labelled: Optional[Set[str]]) -> List[Candidate]:
        if not labelled:
            return list(candidates)
        return [c for c in candidates if c.key not in labelled]


class GridPolicy(AcquisitionPolicy):
    """B0 -- uniform sweep in a deterministic, reproducible order.

    Ordering is by (facet, control variable, cation) rather than the order
    candidates were enumerated, so the baseline does not depend on how the
    candidate list happened to be built.
    """

    name = "B0_grid"

    def select(self, candidates, budget, labelled=None, md_results=None,
               labels=None, round_index=0) -> AcquisitionDecision:
        fresh = self._fresh(candidates, labelled)

        def sort_key(c: Candidate):
            sp = c.state_point
            control = (sp.target_mu_ev if sp.target_mu_ev is not None
                       else (sp.surface_charge_uc_cm2 or 0.0))
            return (sp.facet, control, sp.cation or "", c.frame_index or 0)

        ordered = sorted(fresh, key=sort_key)
        for c in ordered:
            c.score, c.strategy = 1.0, "grid_sweep"
            c.reason = "uniform grid enumeration"
        return AcquisitionDecision(
            round_index=round_index, policy_name=self.name,
            selected=ordered[:budget], considered=len(fresh), budget=budget,
            notes="Deterministic uniform sweep. The baseline to beat.",
        )


class RandomPolicy(AcquisitionPolicy):
    """B1 -- random subsample at matched budget, fixed seed."""

    name = "B1_random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def select(self, candidates, budget, labelled=None, md_results=None,
               labels=None, round_index=0) -> AcquisitionDecision:
        fresh = self._fresh(candidates, labelled)
        # Sort before sampling so the result depends only on the seed, not on
        # dict/set iteration order.
        pool = sorted(fresh, key=lambda c: c.key)
        chosen = self._rng.sample(pool, min(budget, len(pool))) if pool else []
        for c in chosen:
            c.score, c.strategy = 0.0, "random"
            c.reason = f"random draw, seed={self.seed}"
        return AcquisitionDecision(
            round_index=round_index, policy_name=self.name, selected=chosen,
            considered=len(pool), budget=budget,
            notes=f"Uniform random at matched budget (seed={self.seed}).",
        )


class ForceUncertaintyPolicy(AcquisitionPolicy):
    """B2 -- trigger on committee FORCE disagreement only.

    THE CRITICAL ABLATION. Identical machinery to SigmaMuPolicy with the
    Fermi-level term deleted. Any advantage A1 shows over B0/B1 that B2 also
    shows is an advantage of adaptive acquisition in general, not of sigma_mu.
    The publishable claim is the A1 - B2 gap, and nothing else.
    """

    name = "B2_force_uncertainty"

    def __init__(self, sigma_force_threshold: float = 0.05):
        self.sigma_force_threshold = sigma_force_threshold

    def select(self, candidates, budget, labelled=None, md_results=None,
               labels=None, round_index=0) -> AcquisitionDecision:
        fresh = self._fresh(candidates, labelled)
        sigma_by_sp = _sigma_force_by_state_point(md_results or [])

        for c in fresh:
            sigma = sigma_by_sp.get(state_point_id(c.state_point), 0.0)
            c.score = sigma
            c.strategy = ("force_uncertainty" if sigma >= self.sigma_force_threshold
                          else "force_uncertainty_below_threshold")
            c.reason = (f"sigma_F = {sigma:.4f} eV/A "
                        f"(threshold {self.sigma_force_threshold:.4f})")

        ordered = sorted(fresh, key=lambda c: (-c.score, c.key))
        above = [c for c in ordered if c.score >= self.sigma_force_threshold]
        # Fall back to the highest-sigma candidates rather than spending
        # nothing: a policy that declines to spend its budget would win the
        # cost metric trivially without producing any labels.
        selected = (above or ordered)[:budget]
        return AcquisitionDecision(
            round_index=round_index, policy_name=self.name, selected=selected,
            considered=len(fresh), budget=budget,
            notes=(f"{len(above)} candidate(s) above sigma_F threshold; "
                   f"{'threshold' if above else 'top-score fallback'} used."),
        )


def _sigma_force_by_state_point(md_results: Sequence[MDResult]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for result in md_results:
        if result.committee is None:
            continue
        value = result.committee.sigma_force_ev_per_angstrom
        out[result.state_point_id] = max(out.get(result.state_point_id, 0.0), value)
    return out


def enumerate_candidates(facets: Sequence[str],
                         controls: Sequence[float],
                         cations: Sequence[Optional[str]],
                         control_kind: str = "target_mu_ev",
                         n_cation: int = 2,
                         coverages: Optional[Sequence[float]] = None,
                         **state_point_kwargs) -> List[Candidate]:
    """Full Cartesian product of the exploration space.

    This is the ground set every policy draws from, so all of them see the
    same universe -- a policy cannot win by being handed a better menu.
    """
    if control_kind not in ("target_mu_ev", "surface_charge_uc_cm2"):
        raise ValueError(
            "control_kind must be 'target_mu_ev' (constant potential) or "
            "'surface_charge_uc_cm2' (constant charge).")
    coverages = coverages if coverages is not None else [None]

    candidates: List[Candidate] = []
    for facet in facets:
        for control in controls:
            for cation in cations:
                for coverage in coverages:
                    kwargs = dict(state_point_kwargs)
                    kwargs[control_kind] = control
                    sp = StatePoint(
                        facet=facet, cation=cation,
                        n_cation=(n_cation if cation else 0),
                        co_coverage_ml=coverage, **kwargs)
                    candidates.append(Candidate(state_point=sp))
    return candidates
