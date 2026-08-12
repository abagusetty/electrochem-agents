"""
Lightweight, dependency-free random packing for explicit solvent + ions
above a slab. This is a Packmol-free fallback: it uses rejection sampling
with a minimum pairwise distance and periodic boundary conditions in x/y.

For production-scale, high-density packing, prefer Packmol (as used in the
original OC25 pipeline, Methods 5.1.2) if it is available in the target
environment; this module keeps the scaffold runnable without that
dependency and is adequate for moderate solvent depths (5-10 A) and
system sizes typical of the Cu(100)/Cu(310) case study.
"""

import numpy as np


def pack_points(cell_xy, z_range, n_points, min_dist=2.8, existing_points=None,
                 max_attempts_per_point=4000, rng=None):
    """Randomly place n_points in [0,Lx]x[0,Ly]x[zlo,zhi] with periodic
    boundaries in x/y, rejecting candidates closer than `min_dist` to any
    already-accepted point (including points passed in via
    `existing_points`, e.g. previously placed cations or slab atoms).

    Returns (points, success, total_attempts). `success` is False if the
    target density could not be reached within the attempt budget --
    in that case, reduce `min_dist`, increase `z_range`, or reduce
    `n_points`.
    """
    rng = rng or np.random.default_rng(0)
    accepted = (
        np.zeros((0, 3)) if existing_points is None
        else np.asarray(existing_points, dtype=float).copy()
    )
    Lx, Ly = cell_xy
    zlo, zhi = z_range
    new_points = []
    total_attempts = 0
    for _ in range(n_points):
        placed = False
        for _attempt in range(max_attempts_per_point):
            total_attempts += 1
            cand = np.array([
                rng.uniform(0, Lx), rng.uniform(0, Ly), rng.uniform(zlo, zhi)
            ])
            if accepted.shape[0] > 0:
                d_xy = accepted[:, :2] - cand[:2]
                d_xy -= np.round(d_xy / np.array([Lx, Ly])) * np.array([Lx, Ly])
                d_z = accepted[:, 2] - cand[2]
                dist = np.sqrt((d_xy ** 2).sum(axis=1) + d_z ** 2)
                if dist.min() < min_dist:
                    continue
            new_points.append(cand)
            accepted = np.vstack([accepted, cand])
            placed = True
            break
        if not placed:
            return np.array(new_points), False, total_attempts
    return np.array(new_points), True, total_attempts
