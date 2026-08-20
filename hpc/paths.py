"""
Where everything lives on disk.

Every external code (JDFTx, PLUMED, CP-MACE, pseudopotentials, Python
environments) is assumed ALREADY BUILT AND INSTALLED. Nothing in this
module compiles, installs, downloads, or checks out anything -- it only
records paths and validates that they exist when asked.

Defaults follow ALCF Aurora conventions:
  * project space:  /lus/flare/projects/<PROJECT>
  * home:           ~ (small, not for run data)
Run data must live on /lus/flare, never in home.

Override any path via environment variable or by constructing the
dataclass explicitly; `SoftwareStack.from_env()` reads the ELECTROCHEM_*
variables listed against each field.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default


@dataclass
class SoftwareStack:
    """Absolute paths to already-installed external software.

    None means "not configured"; a driver that needs it will raise a
    specific error naming the field and the env var that sets it, rather
    than failing deep inside a subprocess call.
    """

    # --- JDFTx (grand-canonical DFT) -------------------------------------
    # ELECTROCHEM_JDFTX_BIN -- the jdftx executable itself. Do NOT include
    # an MPI launcher prefix here; launching is hpc.aurora's job.
    jdftx_bin: Optional[str] = None
    # ELECTROCHEM_JDFTX_GPU_BIN -- optional separate GPU build (jdftx_gpu).
    jdftx_gpu_bin: Optional[str] = None
    # ELECTROCHEM_JDFTX_PSEUDO_DIR -- pseudopotential root (GBRV / SG15).
    jdftx_pseudo_dir: Optional[str] = None
    # ELECTROCHEM_JDFTX_SCRIPTS -- <jdftx>/scripts, whose ase/ subdir must
    # go on PYTHONPATH for the (secondary) ASE calculator path.
    jdftx_scripts_dir: Optional[str] = None

    # --- PLUMED -----------------------------------------------------------
    # ELECTROCHEM_PLUMED_KERNEL -- libplumedKernel.so; py-plumed dispatches
    # through PLUMED_KERNEL, so this must be exported into the run env.
    plumed_kernel: Optional[str] = None
    plumed_root: Optional[str] = None

    # --- CP-MACE ----------------------------------------------------------
    # ELECTROCHEM_CP_MACE_REPO -- a local git checkout of
    # github.com/yuanyue-liu-group/CP-MACE.
    #
    # LICENSING: that repository ships NO LICENSE file, so it is
    # all-rights-reserved by default. Cloning and running it locally for
    # research is fine. Vendoring its source into this repository, or
    # redistributing its pretrained .model files, is NOT. This is exactly
    # why mlip.cp_mace_simulation imports the NoseHoover integrator live
    # from this path instead of copying it. Keep it that way.
    cp_mace_repo: Optional[str] = None
    # Pin a commit; upstream is unmaintained (last push 2025-09-16).
    cp_mace_commit: Optional[str] = None

    # --- Python environments ---------------------------------------------
    # Aurora practice is one environment per accelerator concern. These are
    # activated by the generated PBS script, not by this process.
    python_env: Optional[str] = None          # ELECTROCHEM_PYTHON_ENV
    fairchem_env: Optional[str] = None        # ELECTROCHEM_FAIRCHEM_ENV
    mace_env: Optional[str] = None            # ELECTROCHEM_MACE_ENV

    # --- Model checkpoints -------------------------------------------------
    # ELECTROCHEM_MODEL_DIR -- local cache for eSEN-OC25 / FermiMACE
    # checkpoints. OC25 checkpoints are gated (Meta FAIR Chemistry License);
    # stage them here once rather than hitting HF from compute nodes, which
    # typically have no outbound network.
    model_dir: Optional[str] = None
    hf_home: Optional[str] = None             # ELECTROCHEM_HF_HOME

    @classmethod
    def from_env(cls) -> "SoftwareStack":
        return cls(
            jdftx_bin=_env("ELECTROCHEM_JDFTX_BIN"),
            jdftx_gpu_bin=_env("ELECTROCHEM_JDFTX_GPU_BIN"),
            jdftx_pseudo_dir=_env("ELECTROCHEM_JDFTX_PSEUDO_DIR"),
            jdftx_scripts_dir=_env("ELECTROCHEM_JDFTX_SCRIPTS"),
            plumed_kernel=_env("ELECTROCHEM_PLUMED_KERNEL"),
            plumed_root=_env("ELECTROCHEM_PLUMED_ROOT"),
            cp_mace_repo=_env("ELECTROCHEM_CP_MACE_REPO"),
            cp_mace_commit=_env("ELECTROCHEM_CP_MACE_COMMIT"),
            python_env=_env("ELECTROCHEM_PYTHON_ENV"),
            fairchem_env=_env("ELECTROCHEM_FAIRCHEM_ENV"),
            mace_env=_env("ELECTROCHEM_MACE_ENV"),
            model_dir=_env("ELECTROCHEM_MODEL_DIR"),
            hf_home=_env("ELECTROCHEM_HF_HOME"),
        )

    def require(self, *fields: str) -> None:
        """Raise a single, actionable error naming every missing field."""
        env_for = {
            "jdftx_bin": "ELECTROCHEM_JDFTX_BIN",
            "jdftx_gpu_bin": "ELECTROCHEM_JDFTX_GPU_BIN",
            "jdftx_pseudo_dir": "ELECTROCHEM_JDFTX_PSEUDO_DIR",
            "jdftx_scripts_dir": "ELECTROCHEM_JDFTX_SCRIPTS",
            "plumed_kernel": "ELECTROCHEM_PLUMED_KERNEL",
            "plumed_root": "ELECTROCHEM_PLUMED_ROOT",
            "cp_mace_repo": "ELECTROCHEM_CP_MACE_REPO",
            "python_env": "ELECTROCHEM_PYTHON_ENV",
            "fairchem_env": "ELECTROCHEM_FAIRCHEM_ENV",
            "mace_env": "ELECTROCHEM_MACE_ENV",
            "model_dir": "ELECTROCHEM_MODEL_DIR",
        }
        missing = []
        for name in fields:
            if getattr(self, name, None) in (None, ""):
                missing.append(f"  {name}  (set {env_for.get(name, '?')})")
        if missing:
            raise RuntimeError(
                "SoftwareStack is missing required paths:\n"
                + "\n".join(missing)
                + "\n\nThese point at software assumed to be already installed. "
                  "Set the environment variables, or construct SoftwareStack(...) "
                  "explicitly."
            )

    def check_exists(self, *fields: str) -> Dict[str, bool]:
        """Report which configured paths actually exist on this filesystem.

        Purely diagnostic -- login nodes and compute nodes can see different
        mounts, so a False here is informative, not necessarily fatal.
        """
        out = {}
        for name in fields:
            value = getattr(self, name, None)
            out[name] = bool(value) and Path(value).exists()
        return out


@dataclass
class ProjectPaths:
    """Run-data layout under Aurora project space.

    One campaign owns one root. Everything the campaign produces is
    addressable from it, so a restart needs only the root.

        <root>/
          inputs/       generated JDFTx inputs, PLUMED files, init structures
          runs/         one directory per calculation, named by state-point id
          labels/       harvested CP-DFT labels (JSONL)
          datasets/     CP-MACE extended-XYZ training sets
          models/       trained committee members
          trajectories/ MD trajectories and COLVAR files
          logs/         PBS stdout/stderr, driver logs
          state/        campaign state, acquisition history, pre-registration
    """

    root: str
    project: Optional[str] = None       # PBS account, e.g. "Catalysis_aesp"

    def __post_init__(self):
        self.root = str(Path(self.root).expanduser())

    @classmethod
    def on_flare(cls, project: str, campaign: str,
                 base: str = "/lus/flare/projects") -> "ProjectPaths":
        """Standard Aurora location: /lus/flare/projects/<project>/<campaign>."""
        return cls(root=str(Path(base) / project / campaign), project=project)

    @property
    def inputs(self) -> Path: return Path(self.root) / "inputs"
    @property
    def runs(self) -> Path: return Path(self.root) / "runs"
    @property
    def labels(self) -> Path: return Path(self.root) / "labels"
    @property
    def datasets(self) -> Path: return Path(self.root) / "datasets"
    @property
    def models(self) -> Path: return Path(self.root) / "models"
    @property
    def trajectories(self) -> Path: return Path(self.root) / "trajectories"
    @property
    def logs(self) -> Path: return Path(self.root) / "logs"
    @property
    def state(self) -> Path: return Path(self.root) / "state"

    def all_dirs(self) -> List[Path]:
        return [self.inputs, self.runs, self.labels, self.datasets,
                self.models, self.trajectories, self.logs, self.state]

    def create(self) -> "ProjectPaths":
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, state_point_id: str) -> Path:
        """Directory for one calculation. Created on demand."""
        directory = self.runs / state_point_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def to_dict(self) -> Dict[str, str]:
        d = asdict(self)
        d.update({name: str(p) for name, p in
                  zip(["inputs", "runs", "labels", "datasets", "models",
                       "trajectories", "logs", "state"], self.all_dirs())})
        return d
