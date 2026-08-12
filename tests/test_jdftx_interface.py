"""Tests for cp_dft.jdftx_interface command assembly (pure logic, no JDFTx
executable or PYTHONPATH setup required)."""

from cp_dft.jdftx_interface import JDFTxConfig, build_jdftx_commands


def test_build_jdftx_commands_neutral_run_has_no_target_mu():
    config = JDFTxConfig(target_mu_hartree=None)
    commands = build_jdftx_commands(config)
    assert "target-mu" not in commands
    assert commands["elec-cutoff"] == "20 100"
    assert commands["fluid"] == "LinearPCM"


def test_build_jdftx_commands_constant_potential_sets_target_mu():
    config = JDFTxConfig(target_mu_hartree=-0.15)
    commands = build_jdftx_commands(config)
    assert commands["target-mu"] == "-0.150000"


def test_build_jdftx_commands_extra_commands_override_and_merge():
    config = JDFTxConfig(extra_commands={"dump": "End ElecDensity", "elec-cutoff": "30"})
    commands = build_jdftx_commands(config)
    assert commands["dump"] == "End ElecDensity"
    assert commands["elec-cutoff"] == "30"


def test_build_jdftx_commands_no_fluid_model_omits_fluid_key():
    config = JDFTxConfig(fluid_model=None)
    commands = build_jdftx_commands(config)
    assert "fluid" not in commands
