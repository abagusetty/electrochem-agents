"""
ALCF Aurora machine model: node topology, environment, and MPI launch.

Aurora node hardware (per ALCF machine documentation):
  * 2x Intel Xeon CPU Max 9470C ("Sapphire Rapids HBM"), 52 cores each
    -> 104 physical cores per node, 208 hardware threads.
  * 6x Intel Data Center GPU Max 1550 ("Ponte Vecchio"), 2 tiles each
    -> 12 tiles per node.
  * 8x HPE Slingshot-11 NICs.
  * Scheduler: PBS Pro. Launcher: mpiexec (PALS).
  * Project filesystem: /lus/flare.

GPU exposure depends on ZE_FLAT_DEVICE_HIERARCHY:
  FLAT       -> each TILE is a separate device (12 per node). Default here,
                because it is the simpler model for one-rank-per-tile MPI
                codes like JDFTx and for torch.xpu enumeration.
  COMPOSITE  -> each CARD is one device (6 per node) with implicit scaling
                across its two tiles.

>>> VERIFY BEFORE FIRST PRODUCTION RUN <<<
Module names, queue names, and node counts change. The values below are
defaults, not guarantees. Check docs.alcf.anl.gov/aurora against
`AuroraConfig.modules`, `AuroraConfig.queue`, and AURORA_NODE, and override
in your campaign config rather than editing this file. Nothing here is
verified against a live machine -- no job has been submitted from this
codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NodeTopology:
    cpus_per_node: int = 104          # 2 sockets x 52 physical cores
    threads_per_core: int = 2
    sockets_per_node: int = 2
    gpus_per_node: int = 6            # physical cards
    tiles_per_gpu: int = 2
    nics_per_node: int = 8

    @property
    def tiles_per_node(self) -> int:
        return self.gpus_per_node * self.tiles_per_gpu

    def devices_per_node(self, flat_hierarchy: bool = True) -> int:
        return self.tiles_per_node if flat_hierarchy else self.gpus_per_node

    def cores_per_rank(self, ranks_per_node: int) -> int:
        if ranks_per_node <= 0:
            raise ValueError("ranks_per_node must be positive.")
        return max(1, self.cpus_per_node // ranks_per_node)


AURORA_NODE = NodeTopology()


@dataclass
class AuroraConfig:
    """Everything needed to turn a command into an Aurora-ready invocation."""

    project: str                              # PBS account (-A)
    queue: str = "prod"                       # debug | debug-scaling | prod | ...
    nodes: int = 1
    walltime: str = "01:00:00"
    filesystems: str = "flare:home"           # -l filesystems=...

    # Parallel decomposition. ranks_per_node defaults to one rank per tile,
    # which is the standard mapping for GPU-resident codes on Aurora.
    ranks_per_node: Optional[int] = None
    threads_per_rank: Optional[int] = None
    flat_device_hierarchy: bool = True

    # Module environment. `frameworks` provides the ALCF-built PyTorch with
    # XPU support plus oneAPI; a pure-MPI code such as JDFTx typically needs
    # only the compiler/MPI modules.
    modules: List[str] = field(default_factory=lambda: ["frameworks"])

    # Extra environment, applied after the defaults below so it can override.
    env: Dict[str, str] = field(default_factory=dict)

    # ALCF ships CPU/GPU affinity helper scripts; if set, the launcher runs
    # the command through this wrapper instead of computing binding itself.
    affinity_script: Optional[str] = None     # e.g. ".../gpu_tile_compact.sh"

    topology: NodeTopology = field(default_factory=lambda: AURORA_NODE)

    def __post_init__(self):
        if self.ranks_per_node is None:
            self.ranks_per_node = self.topology.devices_per_node(
                self.flat_device_hierarchy)
        if self.threads_per_rank is None:
            self.threads_per_rank = self.topology.cores_per_rank(self.ranks_per_node)

    # -- derived -----------------------------------------------------------

    @property
    def total_ranks(self) -> int:
        return self.nodes * self.ranks_per_node

    # -- environment -------------------------------------------------------

    def base_env(self) -> Dict[str, str]:
        """Environment applied to every Aurora job launched from here.

        Rationale for each non-obvious entry:
          ZE_FLAT_DEVICE_HIERARCHY  tile-vs-card device model (see module docstring)
          OMP_NUM_THREADS           must match the PBS/mpiexec CPU binding or
                                    ranks oversubscribe cores and collapse
          OMP_PLACES/PROC_BIND      pin threads; Xeon Max HBM is sensitive to this
          FI_CXI_DEFAULT_CQ_SIZE    Slingshot completion-queue depth; the default
                                    is small for many-rank collectives
          CCL_*                     oneCCL settings for multi-tile torch.distributed
          TORCH_LLM_ALLREDUCE       unset deliberately -- irrelevant here
        """
        environment = {
            "ZE_FLAT_DEVICE_HIERARCHY": "FLAT" if self.flat_device_hierarchy else "COMPOSITE",
            "OMP_NUM_THREADS": str(self.threads_per_rank),
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "close",
            "FI_CXI_DEFAULT_CQ_SIZE": "131072",
            "CCL_PROCESS_LAUNCHER": "pmix",
            "CCL_ATL_TRANSPORT": "mpi",
            # Compute nodes generally have no outbound network. Force offline
            # so a stray HF call fails fast and loudly rather than hanging on
            # a socket timeout inside a 6-hour allocation.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        environment.update(self.env)
        return environment

    def export_lines(self, extra: Optional[Dict[str, str]] = None) -> List[str]:
        environment = self.base_env()
        if extra:
            environment.update(extra)
        return [f'export {key}="{value}"' for key, value in sorted(environment.items())]

    def module_lines(self) -> List[str]:
        if not self.modules:
            return []
        return ["module load " + " ".join(self.modules)]

    # -- launch ------------------------------------------------------------

    def mpiexec(self, command: List[str],
                ranks: Optional[int] = None,
                ranks_per_node: Optional[int] = None,
                depth: Optional[int] = None) -> List[str]:
        """Build the mpiexec (PALS) invocation for `command`.

        `--cpu-bind depth --depth N` spreads each rank over N cores, which is
        what you want when ranks_per_node < cpus_per_node. If an ALCF affinity
        script is configured, it is inserted between mpiexec and the command
        and handles tile binding itself.
        """
        rpn = ranks_per_node or self.ranks_per_node
        total = ranks if ranks is not None else self.nodes * rpn
        cpu_depth = depth or self.topology.cores_per_rank(rpn)

        launch = [
            "mpiexec",
            "-n", str(total),
            "--ppn", str(rpn),
            "--cpu-bind", "depth",
            "--depth", str(cpu_depth),
        ]
        if self.affinity_script:
            launch.append(self.affinity_script)
        return launch + list(command)

    def describe(self) -> str:
        return (
            f"Aurora: {self.nodes} node(s) x {self.ranks_per_node} rank(s) "
            f"= {self.total_ranks} ranks, {self.threads_per_rank} thread(s)/rank, "
            f"{self.topology.devices_per_node(self.flat_device_hierarchy)} device(s)/node "
            f"({'tiles' if self.flat_device_hierarchy else 'cards'}), "
            f"queue={self.queue}, walltime={self.walltime}"
        )


# ---------------------------------------------------------------------------
# Sizing helpers. These encode intent, not measured performance -- no timing
# data exists for this workload on Aurora yet. Treat them as starting points
# to be replaced by measurements after the first real runs.
# ---------------------------------------------------------------------------

def nodes_for_jdftx(n_atoms: int, ranks_per_node: int = 12,
                    atoms_per_rank: int = 8, gpu: bool = True) -> int:
    """Node count for one grand-canonical JDFTx single point.

    Plane-wave DFT parallelises over bands and k-points. The atoms-per-rank
    heuristic stands in for a real strong-scaling study, which does not exist
    for this workload on Aurora.

    JDFTx is ported to SYCL for Aurora PVC, so the GPU path is the production
    path. One rank drives one tile with ~64 GB of HBM, absorbing far more atoms
    than a CPU rank -- `atoms_per_rank` is scaled up accordingly and an
    800-atom explicit-solvent cell lands on roughly one to two nodes rather
    than nine.

    BOTH the x8 multiplier and `atoms_per_rank` are unmeasured guesses. The
    Phase-0 strong-scaling run exists to replace them; they are the difference
    between a comfortable job and an out-of-memory one.
    """
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    per_rank = atoms_per_rank * (8 if gpu else 1)
    ranks = max(1, -(-n_atoms // per_rank))
    return max(1, -(-ranks // ranks_per_node))


def jdftx_sycl_env(flat_hierarchy: bool = True) -> Dict[str, str]:
    """Environment for the SYCL/oneAPI JDFTx build on Aurora PVC.

    ONEAPI_DEVICE_SELECTOR pins JDFTx to Level Zero GPU devices; without it a
    SYCL runtime may silently select the OpenCL CPU device and run correctly
    but at CPU speed, which shows up as a mysteriously slow job rather than an
    error.

    Immediate-command-list and in-order-queue settings are the standard PVC
    latency reductions; plane-wave DFT issues many small kernels per SCF step
    and is sensitive to launch overhead.

    JDFTx is ported to SYCL for Aurora PVC, so this is the production path for
    grand-canonical label generation, not a fallback. The settings below are
    conventional oneAPI/Level-Zero choices; tune them against measured
    performance from the Phase-0 strong-scaling run rather than treating them
    as final.
    """
    return {
        "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT" if flat_hierarchy else "COMPOSITE",
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
        "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE": "0:2",
        "SYCL_CACHE_PERSISTENT": "1",
        # JDFTx allocates large FFT scratch buffers; without pooling, repeated
        # allocation dominates SCF iteration time on Level Zero.
        "SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR": "0",
    }


def nodes_for_mlip_md(n_atoms: int, atoms_per_tile: int = 4000,
                      tiles_per_node: int = 12) -> int:
    """Node count for MLIP MD. Interface MD is usually latency-bound, not
    memory-bound, so the honest default for an 800-atom cell is ONE node --
    often one tile. Scale out over independent state points, not over one
    small trajectory."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    tiles = max(1, -(-n_atoms // atoms_per_tile))
    return max(1, -(-tiles // tiles_per_node))
