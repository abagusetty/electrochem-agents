"""Tests for cp_dft.jdftx_interface tag-assembly logic (pure logic; no
pymatgen, JDFTx executable, or PYTHONPATH setup required)."""

from cp_dft.jdftx_interface import JDFTxRunConfig, _base_tags


def test_base_tags_neutral_run_has_no_target_mu():
    config = JDFTxRunConfig(target_mu_hartree=None)
    tags = _base_tags(config)
    assert "target-mu" not in tags
    assert tags["elec-cutoff"] == "20 100"
    assert tags["fluid"] == "LinearPCM"


def test_base_tags_constant_potential_sets_target_mu():
    config = JDFTxRunConfig(target_mu_hartree=-0.15)
    tags = _base_tags(config)
    assert tags["target-mu"] == "-0.150000"


def test_base_tags_extra_tags_override_and_merge():
    config = JDFTxRunConfig(extra_tags={"dump": "End ElecDensity", "elec-cutoff": "30"})
    tags = _base_tags(config)
    assert tags["dump"] == "End ElecDensity"
    assert tags["elec-cutoff"] == "30"


def test_base_tags_no_fluid_model_omits_fluid_key():
    config = JDFTxRunConfig(fluid_model=None)
    tags = _base_tags(config)
    assert "fluid" not in tags
