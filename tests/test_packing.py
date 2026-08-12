"""Tests for the Packmol-free rejection-sampling packer in systems.packing."""

import numpy as np

from systems.packing import pack_points


def test_pack_points_respects_min_distance_and_pbc():
    rng = np.random.default_rng(42)
    points, ok, attempts = pack_points((15.0, 15.0), (12.0, 20.0), n_points=60,
                                        min_dist=2.8, rng=rng)
    assert ok
    assert len(points) == 60

    lx, ly = 15.0, 15.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d_xy = points[i, :2] - points[j, :2]
            d_xy -= np.round(d_xy / np.array([lx, ly])) * np.array([lx, ly])
            d_z = points[i, 2] - points[j, 2]
            dist = np.sqrt((d_xy ** 2).sum() + d_z ** 2)
            assert dist >= 2.8 - 1e-9


def test_pack_points_respects_existing_points():
    rng = np.random.default_rng(1)
    existing = np.array([[7.5, 7.5, 15.0]])
    points, ok, _ = pack_points((15.0, 15.0), (12.0, 20.0), n_points=10, min_dist=3.0,
                                 existing_points=existing, rng=rng)
    assert ok
    for p in points:
        d_xy = p[:2] - existing[0, :2]
        d_xy -= np.round(d_xy / np.array([15.0, 15.0])) * np.array([15.0, 15.0])
        dist = np.sqrt((d_xy ** 2).sum() + (p[2] - existing[0, 2]) ** 2)
        assert dist >= 3.0 - 1e-9


def test_pack_points_reports_failure_when_infeasible():
    rng = np.random.default_rng(0)
    points, ok, attempts = pack_points((5.0, 5.0), (0.0, 2.0), n_points=200, min_dist=3.0,
                                        max_attempts_per_point=200, rng=rng)
    assert not ok
    assert len(points) < 200
