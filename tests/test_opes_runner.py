"""Tests for md.opes_runner PLUMED input rendering."""

from md.opes_runner import OPESConfig, render_plumed_opes_input, EV_TO_KJ_PER_MOL


def test_render_plumed_opes_input_contains_expected_directives():
    config = OPESConfig(atom1_index=1, atom2_index=2, barrier_ev=5.0, pace=500,
                         upper_wall_angstrom=6.0)
    text = render_plumed_opes_input(config)
    assert "DISTANCE ATOMS=1,2" in text
    assert "OPES_METAD" in text
    assert "SIGMA=ADAPTIVE" in text
    assert "PACE=500" in text
    assert "UPPER_WALLS ARG=cc AT=6.0" in text


def test_barrier_conversion_to_kj_per_mol():
    config = OPESConfig(atom1_index=1, atom2_index=2, barrier_ev=2.0)
    text = render_plumed_opes_input(config)
    expected_kj = 2.0 * EV_TO_KJ_PER_MOL
    assert f"BARRIER={expected_kj:.4f}" in text
