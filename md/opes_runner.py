"""
PLUMED OPES enhanced-sampling input generation for the CO-CO dimerization
collective variable, following Methods 5.4 of arXiv:2509.17862.
"""

from dataclasses import dataclass

EV_TO_KJ_PER_MOL = 96.48533212


@dataclass
class OPESConfig:
    atom1_index: int
    atom2_index: int
    barrier_ev: float = 5.0
    pace: int = 500
    upper_wall_angstrom: float = 6.0
    upper_wall_kappa: float = 100.0
    temperature_k: float = 300.0
    print_stride: int = 10


def render_plumed_opes_input(config: OPESConfig) -> str:
    barrier_kj = config.barrier_ev * EV_TO_KJ_PER_MOL
    lines = [
        f"cc: DISTANCE ATOMS={config.atom1_index},{config.atom2_index}",
        "",
        "opes: OPES_METAD ARG=cc "
        f"PACE={config.pace} BARRIER={barrier_kj:.4f} "
        f"TEMP={config.temperature_k} SIGMA=ADAPTIVE",
        "",
        f"uwall: UPPER_WALLS ARG=cc AT={config.upper_wall_angstrom} "
        f"KAPPA={config.upper_wall_kappa}",
        "",
        f"PRINT ARG=cc,opes.bias STRIDE={config.print_stride} FILE=COLVAR",
    ]
    return "\n".join(lines) + "\n"


def write_plumed_input(config: OPESConfig, out_path: str) -> str:
    text = render_plumed_opes_input(config)
    with open(out_path, "w") as fh:
        fh.write(text)
    return text
