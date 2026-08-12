"""Tests for systems.io_utils extxyz / LAMMPS data writers."""

import numpy as np

from systems.io_utils import write_extxyz, write_lammps_data


def test_write_extxyz_matches_cp_mace_format(tmp_path):
    species = ["Cu", "Cu", "O", "H", "H"]
    positions = np.array([
        [0.0, 0.0, 0.0], [1.8, 1.8, 0.0],
        [1.0, 1.0, 5.0], [1.5, 1.0, 5.8], [0.5, 1.5, 5.8],
    ])
    cell = np.diag([10.0, 10.0, 25.0])
    forces = np.zeros_like(positions)
    out_path = tmp_path / "frame.xyz"
    text = write_extxyz(str(out_path), species, positions, cell, forces=forces, energy=-1.0,
                         extra_fields={"electron": 100.0, "potential": -3.4})
    lines = text.splitlines()
    assert lines[0] == str(len(species))
    assert "electron=100.0" in lines[1]
    assert "potential=-3.4" in lines[1]
    assert "REF_forces:R:3" in lines[1]
    assert out_path.read_text() == text


def test_write_lammps_data_type_mapping_is_alphabetical(tmp_path):
    species = ["O", "Cu", "H", "Cu"]
    positions = np.zeros((4, 3))
    cell = np.diag([10.0, 10.0, 10.0])
    out_path = tmp_path / "system.data"
    text, type_map = write_lammps_data(str(out_path), species, positions, cell,
                                        {"Cu": 63.546, "O": 15.999, "H": 1.008})
    assert type_map == {"Cu": 1, "H": 2, "O": 3}
    assert "4 atoms" in text
    assert "3 atom types" in text
