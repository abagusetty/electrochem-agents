"""Tests for analysis.free_energy reweighting and convergence utilities."""

import numpy as np

from analysis.free_energy import (
    reweighted_fes,
    extract_barrier_and_reaction_energy,
    block_free_energy_convergence,
    is_converged,
    water_orientation_distribution,
)


def _synthetic_trajectory(seed=7, n_steps=20000):
    rng = np.random.default_rng(seed)
    cv = np.concatenate([
        rng.normal(1.5, 0.15, n_steps // 2),
        rng.normal(3.5, 0.2, n_steps // 2),
    ])
    bias_ev = 0.05 * np.abs(np.sin(cv))
    return cv, bias_ev


def test_reweighted_fes_min_is_zero():
    cv, bias = _synthetic_trajectory()
    centers, fes = reweighted_fes(cv, bias, bins=80)
    assert np.nanmin(fes) == 0.0
    assert len(centers) == 80


def test_barrier_and_reaction_energy_are_finite():
    cv, bias = _synthetic_trajectory()
    result = extract_barrier_and_reaction_energy(cv, bias, bins=80)
    assert result["barrier_ev"] is not None
    assert result["barrier_ev"] >= 0.0


def test_block_convergence_reports_all_blocks_on_smooth_trajectory():
    cv, bias = _synthetic_trajectory(n_steps=40000)
    stats = block_free_energy_convergence(cv, bias, n_blocks=5, bins=80)
    assert stats["n_blocks_used"] == 5
    assert stats["barrier_ev_std"] >= 0.0


def test_is_converged_true_for_stationary_long_trajectory():
    cv, bias = _synthetic_trajectory(n_steps=40000)
    assert is_converged(cv, bias, tol_ev=0.05, n_blocks=5, bins=80)


def test_water_orientation_distribution_shape():
    n_frames, n_water = 3, 50
    rng = np.random.default_rng(5)
    o = rng.uniform(0, 4.5, size=(n_frames, n_water, 3))
    h1 = o + np.array([0.5, 0.0, 0.3])
    h2 = o + np.array([-0.5, 0.0, 0.3])
    centers, hist = water_orientation_distribution(o, h1, h2, z_range_angstrom=(0.0, 4.5), bins=36)
    assert len(centers) == 36
    assert len(hist) == 36
    assert (hist >= 0).all()
