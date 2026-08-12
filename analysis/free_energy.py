"""
Free-energy and structural analysis utilities for OPES-biased trajectories,
following Methods 5.4 of arXiv:2509.17862.
"""

import numpy as np

KB_EV_PER_K = 8.617333262e-5


def reweighted_fes(cv_values, bias_values_ev, temperature_k=300.0, bins=100, cv_range=None):
    cv_values = np.asarray(cv_values, dtype=float)
    bias_values_ev = np.asarray(bias_values_ev, dtype=float)
    kt = KB_EV_PER_K * temperature_k
    weights = np.exp(bias_values_ev / kt)

    if cv_range is None:
        cv_range = (cv_values.min(), cv_values.max())
    hist, edges = np.histogram(cv_values, bins=bins, range=cv_range, weights=weights)
    hist = np.where(hist > 0, hist, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fes = -kt * np.log(hist)
    fes = fes - np.nanmin(fes)
    return centers, fes


def extract_barrier_and_reaction_energy(cv_values, bias_values_ev, temperature_k=300.0,
                                         bins=100, cv_range=None):
    centers, fes = reweighted_fes(cv_values, bias_values_ev, temperature_k, bins, cv_range)
    valid = ~np.isnan(fes)
    if valid.sum() < 3:
        return {"barrier_ev": None, "reaction_energy_ev": None}
    barrier = float(np.nanmax(fes[valid]))
    reaction_energy = float(fes[valid][-1])
    return {"barrier_ev": barrier, "reaction_energy_ev": reaction_energy,
            "cv_centers": centers, "fes_ev": fes}


def block_free_energy_convergence(cv_values, bias_values_ev, n_blocks=5,
                                   temperature_k=300.0, bins=100):
    cv_values = np.asarray(cv_values, dtype=float)
    bias_values_ev = np.asarray(bias_values_ev, dtype=float)
    n = len(cv_values)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)

    barriers, reaction_energies = [], []
    for i in range(n_blocks):
        sl = slice(edges[i], edges[i + 1])
        result = extract_barrier_and_reaction_energy(cv_values[sl], bias_values_ev[sl],
                                                       temperature_k, bins)
        if result["barrier_ev"] is None:
            continue
        barriers.append(result["barrier_ev"])
        reaction_energies.append(result["reaction_energy_ev"])

    def _mean_std(values):
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    barrier_mean, barrier_std = _mean_std(barriers)
    rxn_mean, rxn_std = _mean_std(reaction_energies)
    return {
        "barrier_ev_mean": barrier_mean,
        "barrier_ev_std": barrier_std,
        "reaction_energy_ev_mean": rxn_mean,
        "reaction_energy_ev_std": rxn_std,
        "n_blocks_used": len(barriers),
    }


def is_converged(cv_values, bias_values_ev, tol_ev=0.02, n_blocks=5,
                  temperature_k=300.0, bins=100):
    stats = block_free_energy_convergence(cv_values, bias_values_ev, n_blocks,
                                           temperature_k, bins)
    if stats["barrier_ev_std"] is None or stats["reaction_energy_ev_std"] is None:
        return False
    return stats["barrier_ev_std"] < tol_ev and stats["reaction_energy_ev_std"] < tol_ev


def water_orientation_distribution(o_positions, h1_positions, h2_positions,
                                    surface_normal_z=1.0, z_range_angstrom=(0.0, 4.5),
                                    bins=36):
    o = np.asarray(o_positions, dtype=float)
    h1 = np.asarray(h1_positions, dtype=float)
    h2 = np.asarray(h2_positions, dtype=float)

    z = o[..., 2]
    mask = (z >= z_range_angstrom[0]) & (z <= z_range_angstrom[1])

    bisector = (h1 - o) + (h2 - o)
    norm = np.linalg.norm(bisector, axis=-1, keepdims=True)
    bisector_unit = bisector / np.clip(norm, 1e-12, None)

    normal = np.array([0.0, 0.0, surface_normal_z])
    cos_theta = np.clip(bisector_unit @ normal, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    selected = theta_deg[mask]
    hist, edges = np.histogram(selected, bins=bins, range=(0.0, 180.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
