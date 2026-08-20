"""HPC execution layer, targeting ALCF Aurora (Intel Data Center GPU Max /
PVC, oneAPI, PBS Pro).

Three concerns, deliberately separated:

  hpc.paths     -- where software lives. Everything is assumed already
                   installed; nothing here builds or installs anything.
  hpc.aurora    -- Aurora machine model: node topology, module loads,
                   environment, MPI launch command construction.
  hpc.pbs       -- PBS Pro job scripts, submission, and state polling.
  hpc.launcher  -- one interface over "run it here" vs "submit it to PBS",
                   so drivers do not care which they got.
"""

from hpc.paths import SoftwareStack, ProjectPaths
from hpc.aurora import AuroraConfig, AURORA_NODE
from hpc.pbs import PBSJobSpec, PBSJob, submit, poll, wait
from hpc.launcher import Launcher, LocalLauncher, PBSLauncher, JobHandle

__all__ = [
    "SoftwareStack", "ProjectPaths",
    "AuroraConfig", "AURORA_NODE",
    "PBSJobSpec", "PBSJob", "submit", "poll", "wait",
    "Launcher", "LocalLauncher", "PBSLauncher", "JobHandle",
]
