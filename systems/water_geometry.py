"""
Rigid TIP3P-like water molecule geometry utilities (pure numpy, no external
dependencies). Used by systems.cu_interface to place explicit solvent
molecules with correct internal geometry and random orientation.
"""

import numpy as np

O_H_BOND_LENGTH_ANGSTROM = 0.9572
H_O_H_ANGLE_DEGREES = 104.52


def make_water_molecule(o_position, rng):
    """Return (O, H1, H2) positions for a single water molecule centered at
    `o_position`, with a randomly sampled orientation and TIP3P-like rigid
    geometry (O-H = 0.9572 A, H-O-H = 104.52 deg).
    """
    o_position = np.asarray(o_position, dtype=float)
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    bisector = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    tmp = np.array([1.0, 0.0, 0.0]) if abs(bisector[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(bisector, tmp)
    e1 /= np.linalg.norm(e1)
    half_angle = np.deg2rad(H_O_H_ANGLE_DEGREES) / 2.0
    h1_dir = np.cos(half_angle) * bisector + np.sin(half_angle) * e1
    h2_dir = np.cos(half_angle) * bisector - np.sin(half_angle) * e1
    h1 = o_position + O_H_BOND_LENGTH_ANGSTROM * h1_dir
    h2 = o_position + O_H_BOND_LENGTH_ANGSTROM * h2_dir
    return o_position, h1, h2


def n_molecules_for_density(lx, ly, depth_angstrom, density_g_cm3=1.0, molar_mass_g_mol=18.015):
    """Number of solvent molecules needed to approximately match a target
    mass density in a box of footprint lx*ly and given depth (Angstrom),
    following the OC25 dataset construction convention (Methods 5.1.2).
    """
    volume_angstrom3 = lx * ly * depth_angstrom
    volume_cm3 = volume_angstrom3 * 1e-24
    mass_g = density_g_cm3 * volume_cm3
    n_molecules = mass_g / molar_mass_g_mol * 6.02214076e23
    return int(round(n_molecules))
