"""
Constant-potential MD driver for CP-MACE, wired to import CP-MACE's own
NoseHoover integrator LIVE from a local checkout rather than vendoring a
reconstructed copy of it into this repo.

IMPORTANT (why we do not vendor the integrator class itself): CP-MACE's
`NoseHoover` integrator (simulation/slow_growth/integrator.py and the
textually identical simulation/metadynamics/integrator.py) implements the
actual grand-canonical/constant-mu physics -- an extra electronic degree
of freedom coupled to nuclear motion via `Mne` (fictitious electron
mass), `eta_length` (Nose-Hoover chain length), and `targetmu` (target
electrode potential). Confirmed directly from their source (via GitHub
code search, 2026-08-12):
  - `class NoseHoover(MolecularDynamics)` with constructor signature
    `__init__(self, atoms, timestep, ...)`.
  - `step()` begins with
    `accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)`.
  - Confirmed unit handling in their simulate.py:
    `config["integrator_config"]["timestep"] *= units.fs`; and if
    `integrator in ["NoseHoover", "NoseHooverChain"]`:
    `config["integrator_config"]["temperature"] *= units.kB`.
  - The full internal chain-propagation / electron-coupling algorithm
    could NOT be retrieved in full via the tools available in this
    session (GitHub code search returns short match fragments only, and
    raw.githubusercontent.com fetches failed for this repo) -- so it is
    NOT reproduced here. Reconstructing it from fragments would risk
    silently wrong constant-potential physics, which is worse than not
    vendoring at all.

Architecture (confirmed from their metadynamics/simulate.py, which
imports BOTH `ase.calculators.plumed.Plumed` AND uses `integrator:
NoseHoover` in its own inputs.yml): PLUMED wraps the CALCULATOR to add
CV bias forces; NoseHoover remains the INTEGRATOR that steps positions/
velocities (and the electron degree of freedom) forward. The two are
complementary layers, not alternatives -- do not attempt to replace the
integrator with PLUMED; PLUMED has no time-stepping role.

Setup required before this module works:
  1. Clone CP-MACE: `git clone https://github.com/yuanyue-liu-group/CP-MACE`
  2. Follow their README to install the CP-MACE-patched `mace` package.
  3. Pass `cp_mace_repo_path` below pointing at your CP-MACE checkout, so
     this module can import `NoseHoover` directly from
     `<repo>/simulation/slow_growth/integrator.py` at runtime.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CPMACEIntegratorConfig:
    """Mirrors CP-MACE's own `integrator_config` schema exactly, as
    confirmed from simulation/slow_growth/inputs.yml and
    simulation/metadynamics/inputs.yml.
    """
    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    ttime: float = 40.0
    constraints: List[List[float]] = field(default_factory=list)  # [[i, j, k, dist]] for
                                                                     # slow-growth CV, or [] for free MD
    increm: float = 0.001  # CV increment per step (slow growth only)
    mne: float = 660.74  # fictitious electron mass ("Mne")
    eta_length: int = 2  # Nose-Hoover chain length
    targetmu: float = -3.36  # target electrode potential (grand-canonical)
    shaketol: Optional[float] = None  # metadynamics mode only
    shakemaxiter: Optional[int] = None  # metadynamics mode only

    def to_cp_mace_dict(self) -> Dict[str, Any]:
        """Return a dict matching CP-MACE's `integrator_config` YAML
        format exactly (pre-unit-conversion; CP-MACE's own simulate.py
        multiplies timestep by ase.units.fs and temperature by
        ase.units.kB internally when integrator is NoseHoover/NoseHooverChain).
        """
        d = {
            "timestep": self.timestep_fs, "temperature": self.temperature_k,
            "ttime": self.ttime, "constraints": self.constraints,
            "increm": self.increm, "Mne": self.mne, "eta_length": self.eta_length,
            "targetmu": self.targetmu,
        }
        if self.shaketol is not None:
            d["shaketol"] = self.shaketol
        if self.shakemaxiter is not None:
            d["shakemaxiter"] = self.shakemaxiter
        return d


@dataclass
class CPMACERunConfig:
    cp_mace_repo_path: str  # path to your CP-MACE git checkout
    model_path: str  # path to a trained/compiled FermiMACE .model file
    init_xyz_path: str  # initial structure (electron=/potential= tags optional here)
    integrator_config: CPMACEIntegratorConfig = field(default_factory=CPMACEIntegratorConfig)
    mode: str = "slow_growth"  # "slow_growth" or "metadynamics"
    steps: int = 100
    save_dir: str = "result"
    save_freq: int = 1
    read_velocity: bool = True
    force_threshold: float = 0.15
    fermi_threshold: float = 0.04
    t_init_k: float = 300.0
    plumed_input_lines: Optional[List[str]] = None  # required if mode == "metadynamics"


def _import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_cp_mace_integrator_class(config: CPMACERunConfig):
    """Import CP-MACE's own `NoseHoover` integrator class LIVE from the
    user's CP-MACE checkout, rather than a copy vendored into this repo
    (see module docstring for why). Returns the class object; instantiate
    it directly with `NoseHoover(atoms, timestep, **rest_of_integrator_config)`
    per CP-MACE's own simulate.py usage pattern.
    """
    repo_path = Path(config.cp_mace_repo_path)
    subdir = "slow_growth" if config.mode == "slow_growth" else "metadynamics"
    integrator_file = repo_path / "simulation" / subdir / "integrator.py"
    if not integrator_file.exists():
        raise FileNotFoundError(
            f"Could not find {integrator_file}. Check cp_mace_repo_path points "
            "at a valid clone of https://github.com/yuanyue-liu-group/CP-MACE."
        )
    module = _import_module_from_path(f"cp_mace_integrator_{config.mode}", integrator_file)
    if not hasattr(module, "NoseHoover"):
        raise AttributeError(
            f"{integrator_file} does not define NoseHoover -- CP-MACE's "
            "integrator.py may have been renamed/refactored upstream; "
            "check github.com/yuanyue-liu-group/CP-MACE for the current API."
        )
    return module.NoseHoover


def load_fermi_mace_calculator(config: CPMACERunConfig):
    """Load a trained FermiMACE .model file as an ASE calculator, using
    CP-MACE's own MACECalculator class (mace/calculators/mace.py, an
    ase.calculators.calculator.Calculator subclass -- confirmed from
    source). Requires the CP-MACE-patched `mace` package to be installed
    (see CP-MACE README: clone ACEsuit/mace, replace ./mace with the
    CP-MACE version, `pip install ./mace`).
    """
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "Could not import MACECalculator. Install the CP-MACE-patched "
            "mace package per its README: clone ACEsuit/mace, replace "
            "./mace with the CP-MACE version, then `pip install ./mace`."
        ) from exc
    return MACECalculator(model_paths=[config.model_path], device="cuda")


def build_atoms_and_calculator(config: CPMACERunConfig):
    """Read the initial structure once and attach the appropriate
    calculator: plain FermiMACE for slow-growth mode, or FermiMACE
    wrapped in ase.calculators.plumed.Plumed for metadynamics mode --
    matching CP-MACE's own simulation/metadynamics/simulate.py
    architecture (PLUMED wraps the calculator; NoseHoover remains the
    integrator). Returns the ASE Atoms object with `.calc` already set.
    """
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required for build_atoms_and_calculator.") from exc

    atoms = read(config.init_xyz_path)
    base_calc = load_fermi_mace_calculator(config)

    if config.mode == "slow_growth":
        atoms.calc = base_calc
        return atoms

    if config.mode == "metadynamics":
        if not config.plumed_input_lines:
            raise ValueError("plumed_input_lines is required when mode='metadynamics'.")
        try:
            from ase import units
            from ase.calculators.plumed import Plumed
        except ImportError as exc:
            raise ImportError(
                "ASE (with PLUMED bindings) is required for metadynamics mode. "
                "Install ase and py-plumed (`conda install -c conda-forge py-plumed`)."
            ) from exc
        atoms.calc = Plumed(
            calc=base_calc, input=config.plumed_input_lines,
            timestep=config.integrator_config.timestep_fs * units.fs,
            atoms=atoms, kT=config.integrator_config.temperature_k * units.kB,
        )
        return atoms

    raise ValueError(f"Unknown mode: {config.mode!r} (expected 'slow_growth' or 'metadynamics')")


def run_cp_mace_md(config: CPMACERunConfig):
    """Assemble atoms + calculator + CP-MACE's own NoseHoover integrator
    and run `config.steps` steps, following the exact unit-conversion
    convention confirmed from CP-MACE's simulate.py:
        integrator_config["timestep"] *= ase.units.fs
        integrator_config["temperature"] *= ase.units.kB  # NoseHoover(Chain) only

    Returns the NoseHoover integrator instance after running.
    """
    try:
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    except ImportError as exc:
        raise ImportError("ASE is required for run_cp_mace_md.") from exc

    NoseHoover = load_cp_mace_integrator_class(config)
    atoms = build_atoms_and_calculator(config)

    if not config.read_velocity:
        MaxwellBoltzmannDistribution(atoms, temperature_K=config.t_init_k)

    integrator_kwargs = config.integrator_config.to_cp_mace_dict()
    integrator_kwargs["timestep"] = integrator_kwargs["timestep"] * units.fs
    integrator_kwargs["temperature"] = integrator_kwargs["temperature"] * units.kB

    dyn = NoseHoover(atoms, **integrator_kwargs)
    dyn.run(config.steps)
    return dyn
