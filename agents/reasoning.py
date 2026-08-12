"""
Deterministic reasoning support for the electrochemistry exploration
agents, following the two-mode structure of Fei et al., "Agentic LLM
Reasoning in a Self-Driving Laboratory for Air-Sensitive Lithium Halide
Spinel Conductors" (arXiv:2604.11957): an abductive (abnormality-chasing)
mode and an inductive (pattern-distilling) mode, plus a BO-assisted
variant of the latter once enough data has accumulated.

None of this module calls an LLM. It exists to give the LLM agents
concrete, computed evidence to reason over -- comparisons against
literature reference values, deviations from LOCAL chemical neighbors
(not just fixed literature values, since exploration moves beyond what
arXiv:2509.17862 studied), distilled statistical trends, and BO-style
candidate proposals -- mirroring how the anchor paper's own BO model
constrains, rather than replaces, LLM judgment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REFERENCE_VALUES = {
    ("100", "neutral"): {"barrier_ev": 0.64, "reaction_energy_ev": 0.375},
    ("310", "neutral"): {"barrier_ev": 0.57, "reaction_energy_ev": 0.088},
    ("310", "-23"): {"barrier_ev": 0.49, "reaction_energy_ev": -0.037},
}

CATION_EFFECT_MAX_EV = 0.05
CHARGE_EFFECT_TYPICAL_EV = 0.17


@dataclass
class SimulationRecord:
    facet: str
    surface_charge_density: float
    cation: Optional[str]
    barrier_ev: float
    reaction_energy_ev: float
    barrier_ev_std: Optional[float] = None
    reaction_energy_ev_std: Optional[float] = None
    converged: bool = True
    potential_v: Optional[float] = None


@dataclass
class AnalysisContext:
    facet: str
    condition_label: str
    barrier_ev: float
    reaction_energy_ev: float
    barrier_ev_std: Optional[float] = None
    reaction_energy_ev_std: Optional[float] = None
    converged: Optional[bool] = None
    cation_effect_ev: Optional[float] = None


def build_comparison_report(context: AnalysisContext) -> dict:
    key = (context.facet, context.condition_label)
    reference = REFERENCE_VALUES.get(key)
    flags = []
    d_barrier = None
    d_rxn = None

    if context.converged is False:
        flags.append(
            "NOT CONVERGED: block-uncertainty check failed. Do not report "
            "this barrier/reaction energy as final; recommend continued "
            "sampling or investigate slow interfacial relaxation."
        )

    if reference is not None:
        d_barrier = context.barrier_ev - reference["barrier_ev"]
        d_rxn = context.reaction_energy_ev - reference["reaction_energy_ev"]
        if abs(d_barrier) > 0.15:
            flags.append(
                f"Barrier deviates from anchor-paper reference by {d_barrier:+.3f} eV "
                f"(reference {reference['barrier_ev']:.3f} eV for Cu({context.facet}), "
                f"{context.condition_label})."
            )
        if abs(d_rxn) > 0.15:
            flags.append(
                f"Reaction energy deviates from anchor-paper reference by {d_rxn:+.3f} eV "
                f"(reference {reference['reaction_energy_ev']:.3f} eV)."
            )
    else:
        flags.append(
            f"No anchor-paper reference point for facet={context.facet}, "
            f"condition={context.condition_label}. May be a genuinely new regime."
        )

    if context.cation_effect_ev is not None and context.cation_effect_ev > CATION_EFFECT_MAX_EV:
        flags.append(
            f"Observed cation-identity effect ({context.cation_effect_ev:.3f} eV) exceeds "
            f"the anchor paper's typical range (<= {CATION_EFFECT_MAX_EV} eV)."
        )

    if context.barrier_ev_std is not None and context.barrier_ev_std > 0.05:
        flags.append(
            f"Barrier block-uncertainty ({context.barrier_ev_std:.3f} eV) is larger than "
            "typical converged values in the anchor study; treat as provisional."
        )

    return {
        "facet": context.facet, "condition_label": context.condition_label,
        "observed": {"barrier_ev": context.barrier_ev, "reaction_energy_ev": context.reaction_energy_ev},
        "reference": reference, "delta_barrier_ev": d_barrier,
        "delta_reaction_energy_ev": d_rxn, "flags": flags,
    }


def _charge_distance(a: SimulationRecord, b: SimulationRecord) -> float:
    a_val = a.potential_v if a.potential_v is not None else a.surface_charge_density
    b_val = b.potential_v if b.potential_v is not None else b.surface_charge_density
    return abs(a_val - b_val)


def find_local_abnormalities(records: Sequence[SimulationRecord], barrier_tol_ev: float = 0.12,
                              rxn_tol_ev: float = 0.12, max_neighbor_charge_distance: float = 15.0,
                              min_neighbors: int = 2) -> List[dict]:
    results = []
    for i, record in enumerate(records):
        neighbors = [
            r for j, r in enumerate(records)
            if j != i and r.facet == record.facet and r.converged
            and _charge_distance(r, record) <= max_neighbor_charge_distance
        ]
        if len(neighbors) < min_neighbors:
            continue

        neighbor_barriers = np.array([n.barrier_ev for n in neighbors])
        neighbor_rxns = np.array([n.reaction_energy_ev for n in neighbors])
        d_barrier = record.barrier_ev - neighbor_barriers.mean()
        d_rxn = record.reaction_energy_ev - neighbor_rxns.mean()

        if abs(d_barrier) > barrier_tol_ev or abs(d_rxn) > rxn_tol_ev:
            results.append({
                "index": i, "record": record,
                "delta_barrier_ev": float(d_barrier), "delta_reaction_energy_ev": float(d_rxn),
                "n_neighbors": len(neighbors),
                "neighbor_barrier_mean": float(neighbor_barriers.mean()),
                "neighbor_rxn_mean": float(neighbor_rxns.mean()),
            })
    return results


def distill_patterns(records: Sequence[SimulationRecord]) -> dict:
    converged = [r for r in records if r.converged]
    if not converged:
        return {"facets": {}, "cation_effect_ev": None, "n_records": 0}

    facets: Dict[str, dict] = {}
    for facet in sorted({r.facet for r in converged}):
        facet_records = [r for r in converged if r.facet == facet]
        charges = np.array([
            r.potential_v if r.potential_v is not None else r.surface_charge_density
            for r in facet_records
        ])
        barriers = np.array([r.barrier_ev for r in facet_records])
        rxns = np.array([r.reaction_energy_ev for r in facet_records])

        slope_barrier = None
        slope_rxn = None
        if len(facet_records) >= 2 and np.ptp(charges) > 1e-9:
            slope_barrier = float(np.polyfit(charges, barriers, 1)[0])
            slope_rxn = float(np.polyfit(charges, rxns, 1)[0])

        facets[facet] = {
            "n_records": len(facet_records),
            "mean_barrier_ev": float(barriers.mean()),
            "mean_reaction_energy_ev": float(rxns.mean()),
            "barrier_vs_charge_slope": slope_barrier,
            "reaction_energy_vs_charge_slope": slope_rxn,
        }

    facet_ranking = sorted(facets.items(), key=lambda kv: kv[1]["mean_reaction_energy_ev"])

    cation_groups: Dict[str, List[float]] = {}
    for r in converged:
        if r.cation:
            cation_groups.setdefault(r.cation, []).append(r.reaction_energy_ev)
    cation_effect_ev = None
    if len(cation_groups) >= 2:
        cation_means = [np.mean(v) for v in cation_groups.values()]
        cation_effect_ev = float(max(cation_means) - min(cation_means))

    return {
        "facets": facets,
        "facet_ranking_most_to_least_favorable": [name for name, _ in facet_ranking],
        "cation_effect_ev": cation_effect_ev,
        "n_records": len(converged),
    }


def _encode_candidate(facet: str, charge_or_potential: float, cation: Optional[str],
                       facet_order: Sequence[str], cation_order: Sequence[str]) -> np.ndarray:
    facet_idx = facet_order.index(facet) if facet in facet_order else -1
    cation_idx = cation_order.index(cation) if cation in cation_order else -1
    return np.array([facet_idx, charge_or_potential, cation_idx], dtype=float)


def propose_bo_candidates(records: Sequence[SimulationRecord], candidate_pool: Sequence[dict],
                           n_candidates: int = 5, favor_exergonic: bool = True) -> List[dict]:
    converged = [r for r in records if r.converged]
    if not converged:
        return list(candidate_pool)[:n_candidates]

    patterns = distill_patterns(converged)
    facet_order = sorted({r.facet for r in converged} | {c["facet"] for c in candidate_pool})
    cation_order = sorted({r.cation for r in converged if r.cation}
                           | {c.get("cation") for c in candidate_pool if c.get("cation")})

    explored_points = np.array([
        _encode_candidate(r.facet, r.potential_v if r.potential_v is not None else r.surface_charge_density,
                           r.cation, facet_order, cation_order)
        for r in converged
    ])
    explored_scale = np.maximum(explored_points.std(axis=0), 1e-6)

    scored = []
    for cand in candidate_pool:
        facet = cand["facet"]
        charge = cand.get("potential_v", cand.get("surface_charge_density"))
        cation = cand.get("cation")

        point = _encode_candidate(facet, charge, cation, facet_order, cation_order)
        distances = np.linalg.norm((explored_points - point) / explored_scale, axis=1)
        novelty = float(distances.min())

        facet_stats = patterns["facets"].get(facet)
        predicted_favorability = 0.0
        if facet_stats and facet_stats["reaction_energy_vs_charge_slope"] is not None:
            extrapolated_rxn = (
                facet_stats["mean_reaction_energy_ev"]
                + facet_stats["reaction_energy_vs_charge_slope"]
                * (charge - np.mean([
                    r.potential_v if r.potential_v is not None else r.surface_charge_density
                    for r in converged if r.facet == facet
                ]))
            )
            predicted_favorability = -extrapolated_rxn if favor_exergonic else -abs(extrapolated_rxn)

        score = novelty + predicted_favorability
        scored.append({**cand, "novelty": novelty, "predicted_favorability": predicted_favorability,
                        "score": score})

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:n_candidates]
