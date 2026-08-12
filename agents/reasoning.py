"""
Deterministic reasoning support for the Results Analyst Agent.

This module does NOT call an LLM. It pre-computes structured comparisons
against the reference values from Sahoo et al. (arXiv:2509.17862) so that
the LLM agent's judgment is grounded in concrete numeric context rather
than being asked to recall literature values from memory. The agent still
makes the final call (e.g. "is this deviation meaningful or a convergence
artifact?"), but it does so with this comparison table in front of it,
mirroring how LAMMPS-Agents' Results Analyser is given concrete tool
outputs (OVITO images, thermo data) rather than being asked to reason in
a vacuum.
"""

from dataclasses import dataclass
from typing import Optional

# Reference values digitized directly from Sahoo et al., arXiv:2509.17862.
REFERENCE_VALUES = {
    ("100", "neutral"): {"barrier_ev": 0.64, "reaction_energy_ev": 0.375},
    ("310", "neutral"): {"barrier_ev": 0.57, "reaction_energy_ev": 0.088},
    ("310", "-23"): {"barrier_ev": 0.49, "reaction_energy_ev": -0.037},
}

CATION_EFFECT_MAX_EV = 0.05  # anchor paper: cation identity shifts <=~0.03-0.05 eV
CHARGE_EFFECT_TYPICAL_EV = 0.17  # anchor paper: 0 -> -31.3 uC/cm^2 shifts rxn energy ~0.173 eV


@dataclass
class AnalysisContext:
    facet: str  # "100" or "310"
    condition_label: str  # "neutral", "-23", etc. -- must match a REFERENCE_VALUES key
    barrier_ev: float
    reaction_energy_ev: float
    barrier_ev_std: Optional[float] = None
    reaction_energy_ev_std: Optional[float] = None
    converged: Optional[bool] = None
    cation_effect_ev: Optional[float] = None  # observed spread across cation species,
                                                 # if this run is part of a cation sweep


def build_comparison_report(context: AnalysisContext) -> dict:
    """Compare a completed simulation's results against the closest
    anchor-paper reference point and flag anything that warrants the
    Results Analyst Agent's closer attention before reporting.

    Returns a dict with the raw comparison plus a list of `flags` (human-
    readable strings) that the LLM agent should specifically address in
    its final summary, rather than silently passing over.
    """
    key = (context.facet, context.condition_label)
    reference = REFERENCE_VALUES.get(key)
    flags = []

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
                f"{context.condition_label}). Check MLIP validity and convergence "
                "before treating this as a new finding."
            )
        if abs(d_rxn) > 0.15:
            flags.append(
                f"Reaction energy deviates from anchor-paper reference by {d_rxn:+.3f} eV "
                f"(reference {reference['reaction_energy_ev']:.3f} eV). Same caveats as above."
            )
    else:
        d_barrier = None
        d_rxn = None
        flags.append(
            f"No anchor-paper reference point for facet={context.facet}, "
            f"condition={context.condition_label}. This may be a genuinely new "
            "regime (e.g. a facet/condition not studied in arXiv:2509.17862) -- "
            "note this explicitly rather than implying a mismatch."
        )

    if context.cation_effect_ev is not None and context.cation_effect_ev > CATION_EFFECT_MAX_EV:
        flags.append(
            f"Observed cation-identity effect ({context.cation_effect_ev:.3f} eV) exceeds "
            f"the anchor paper's typical range (<= {CATION_EFFECT_MAX_EV} eV). Re-check "
            "sampling per-cation before concluding cation identity matters here."
        )

    if context.barrier_ev_std is not None and context.barrier_ev_std > 0.05:
        flags.append(
            f"Barrier block-uncertainty ({context.barrier_ev_std:.3f} eV) is larger than "
            "typical converged values in the anchor study; treat the barrier estimate "
            "as provisional."
        )

    return {
        "facet": context.facet,
        "condition_label": context.condition_label,
        "observed": {
            "barrier_ev": context.barrier_ev,
            "reaction_energy_ev": context.reaction_energy_ev,
        },
        "reference": reference,
        "delta_barrier_ev": d_barrier,
        "delta_reaction_energy_ev": d_rxn,
        "flags": flags,
    }
