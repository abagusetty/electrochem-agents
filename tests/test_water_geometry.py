"""Pure-numpy tests for systems.water_geometry (no ase/pymatgen required)."""

import numpy as np

from systems.water_geometry import (
    O_H_BOND_LENGTH_ANGSTROM,
    H_O_H_ANGLE_DEGREES,
    make_water_molecule,
    n_molecules_for_density,
)


def test_water_geometry_bond_length_and_angle():
    rng = np.random.default_rng(1)
    o, h1, h2 = make_water_molecule(np.array([5.0, 5.0, 15.0]), rng)
    d1 = np.linalg.norm(h1 - o)
    d2 = np.linalg.norm(h2 - o)
    angle = np.degrees(np.arccos(np.dot(h1 - o, h2 - o) / (d1 * d2)))
    assert abs(d1 - O_H_BOND_LENGTH_ANGSTROM) < 1e-6
    assert abs(d2 - O_H_BOND_LENGTH_ANGSTROM) < 1e-6
    assert abs(angle - H_O_H_ANGLE_DEGREES) < 1e-6


def test_water_geometry_random_orientation_varies():
    rng = np.random.default_rng(2)
    o = np.array([0.0, 0.0, 0.0])
    _, h1_a, _ = make_water_molecule(o, rng)
    _, h1_b, _ = make_water_molecule(o, rng)
    assert not np.allclose(h1_a, h1_b)


def test_n_molecules_for_density_matches_bulk_water():
    n = n_molecules_for_density(15.0, 15.0, 8.0)
    assert 55 <= n <= 65
