"""
Drive JDFTx grand-canonical calculations on Aurora.

Responsibilities, in order:
  1. materialise a structure for each JDFTxCalcSpec (from disk, or from a
     trajectory frame the acquisition policy chose),
  2. write the JDFTx input,
  3. build the mpiexec command,
  4. hand it to a Launcher (local | one PBS job each | bundled in one
     allocation),
  5. harvest results into CPDFTLabels.

It does not decide WHAT to run -- that is the acquisition policy -- and it
does not decide whether a run succeeded -- that is data.harvest.

Nothing here has been executed against JDFTx.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cp_dft.jdftx_setup import JDFTxCalcSpec, JDFTxProtocol, write_input
from data.harvest import harvest_jdftx_run
from data.schema import CPDFTLabel, RunStatus
from hpc.aurora import AuroraConfig, jdftx_sycl_env, nodes_for_jdftx
from hpc.launcher import JobHandle, Launcher
from hpc.paths import ProjectPaths, SoftwareStack


@dataclass
class JDFTxDriverConfig:
    stack: SoftwareStack
    paths: ProjectPaths
    aurora: AuroraConfig
    launcher: Launcher
    # JDFTx is ported to SYCL and runs GPU-native on Aurora PVC, so the GPU
    # build is the production path and `atoms_per_rank` is interpreted per tile
    # (see hpc.aurora.nodes_for_jdftx).
    use_gpu_build: bool = True
    sycl: bool = True
    # Ranks per node for JDFTx. One rank per tile (12) matches the SYCL build;
    # a CPU build usually wants fewer, fatter ranks with more OMP threads.
    ranks_per_node: Optional[int] = None
    atoms_per_rank: int = 8
    max_nodes_per_calc: int = 32
    input_filename: str = "in"
    output_filename: str = "out"
    # Reuse the converged state of a previous potential as the starting guess
    # for the next. A potential sweep converges far faster stepped than cold,
    # because the density changes little between adjacent potentials.
    chain_initial_state: bool = True

    def executable(self) -> str:
        if self.use_gpu_build and self.stack.jdftx_gpu_bin:
            return self.stack.jdftx_gpu_bin
        if self.use_gpu_build:
            raise RuntimeError(
                "use_gpu_build=True but no jdftx_gpu_bin is configured. Point "
                "ELECTROCHEM_JDFTX_GPU_BIN at the SYCL build of JDFTx (note "
                "upstream JDFTx has no Intel-GPU backend, so this must be the "
                "project's port). Set use_gpu_build=False to fall "
                "back to a CPU build -- but note that the node-count heuristic "
                "in hpc.aurora.nodes_for_jdftx assumes GPU ranks and will "
                "under-allocate badly for a CPU run."
            )
        self.stack.require("jdftx_bin")
        return self.stack.jdftx_bin

    def run_env(self) -> Dict[str, str]:
        """Environment for the JDFTx step, including SYCL device selection."""
        env: Dict[str, str] = {}
        if self.use_gpu_build and self.sycl:
            env.update(jdftx_sycl_env(
                flat_hierarchy=self.aurora.flat_device_hierarchy))
        if self.stack.jdftx_pseudo_dir:
            env["JDFTX_PSEUDO_DIR"] = self.stack.jdftx_pseudo_dir
        return env


def _resolve_structure(spec: JDFTxCalcSpec):
    """Get a pymatgen Structure for this spec.

    Three sources, in priority order: an explicit structure file; a frame of
    a trajectory; or built from scratch via systems.cu_interface. The third
    path exists so a sweep can be defined purely as state points, with no
    pre-staged files.
    """
    if spec.structure_path:
        from pymatgen.core import Structure
        path = Path(spec.structure_path)
        if not path.exists():
            raise FileNotFoundError(f"structure_path does not exist: {path}")
        return Structure.from_file(str(path))

    if spec.parent_trajectory is not None and spec.frame_index is not None:
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = read(spec.parent_trajectory, index=spec.frame_index)
        return AseAtomsAdaptor.get_structure(atoms)

    from pymatgen.io.ase import AseAtomsAdaptor
    from systems.cu_interface import InterfaceSpec, build_cu_water_cation_interface

    sp = spec.state_point
    interface_spec = InterfaceSpec(
        facet=sp.facet, nx=sp.nx, ny=sp.ny, n_layers=sp.n_layers,
        solvent=sp.solvent, solvent_depth_angstrom=sp.solvent_depth_angstrom,
        cation=sp.cation, n_cation=sp.n_cation, seed=sp.seed,
    )
    atoms, _ = build_cu_water_cation_interface(interface_spec)
    return AseAtomsAdaptor.get_structure(atoms)


def _n_atoms_of(structure) -> int:
    return len(structure)


def build_jdftx_command(config: JDFTxDriverConfig, n_atoms: int,
                        run_dir: Path) -> List[str]:
    """The shell lines that run one JDFTx calculation.

    Uses $ELECTROCHEM_MPI_HOSTS when present so a BundledLauncher can pin this
    calculation to its own slice of a shared allocation.
    """
    ranks_per_node = config.ranks_per_node or config.aurora.ranks_per_node
    nodes = min(
        config.max_nodes_per_calc,
        nodes_for_jdftx(n_atoms, ranks_per_node=ranks_per_node,
                        atoms_per_rank=config.atoms_per_rank,
                        gpu=config.use_gpu_build),
    )
    nodes = min(nodes, config.aurora.nodes)
    total_ranks = nodes * ranks_per_node

    exe = config.executable()
    base = config.aurora.mpiexec(
        [exe, "-i", config.input_filename, "-o", config.output_filename],
        ranks=total_ranks, ranks_per_node=ranks_per_node,
    )
    launch = " ".join(base)

    # Same command, but host-pinned, when running inside a bundle.
    pinned = launch.replace(
        "mpiexec", 'mpiexec --hosts "$ELECTROCHEM_MPI_HOSTS"', 1)

    lines = [f'cd {run_dir}']
    for key, value in sorted(config.run_env().items()):
        lines.append(f'export {key}="{value}"')
    lines += [
        'if [ -n "${ELECTROCHEM_MPI_HOSTS:-}" ]; then',
        f'  {pinned}',
        'else',
        f'  {launch}',
        'fi',
    ]
    return lines


@dataclass
class JDFTxSubmission:
    spec: JDFTxCalcSpec
    run_dir: str
    handle: JobHandle
    n_atoms: int


class JDFTxDriver:
    """Submit and harvest grand-canonical JDFTx calculations."""

    def __init__(self, config: JDFTxDriverConfig):
        self.config = config
        self.config.paths.create()

    # -- submit ------------------------------------------------------------

    def submit_one(self, spec: JDFTxCalcSpec,
                   initial_state_from: Optional[str] = None) -> JDFTxSubmission:
        config = self.config
        run_dir = config.paths.run_dir(spec.calc_id)

        if initial_state_from and config.chain_initial_state:
            spec.initial_state_dir = initial_state_from

        structure = _resolve_structure(spec)
        n_atoms = _n_atoms_of(structure)
        write_input(spec, structure, run_dir, filename=config.input_filename)

        # Record the exact protocol next to the run; a label whose protocol
        # cannot be reconstructed is not usable as training data.
        (run_dir / "protocol.txt").write_text(
            f"protocol_fingerprint={spec.protocol.fingerprint()}\n"
            f"target_mu_hartree={spec.target_mu_hartree}\n"
            f"target_mu_ev={spec.target_mu_ev}\n"
            f"n_atoms={n_atoms}\n"
        )

        preamble = []
        if config.stack.python_env:
            preamble.append(f'source "{config.stack.python_env}/bin/activate"')

        handle = config.launcher.run(
            name=f"jdftx_{spec.calc_id}",
            commands=build_jdftx_command(config, n_atoms, run_dir),
            work_dir=str(run_dir),
            log_dir=str(config.paths.logs),
            preamble=preamble,
        )
        return JDFTxSubmission(spec=spec, run_dir=str(run_dir),
                               handle=handle, n_atoms=n_atoms)

    def submit_sweep(self, specs: Sequence[JDFTxCalcSpec]) -> List[JDFTxSubmission]:
        """Submit a potential sweep.

        With `chain_initial_state`, specs are ordered by target-mu and each
        warm-starts from the previous run's converged state. That makes the
        sweep sequential rather than parallel -- worth it, because a cold
        grand-canonical solve at strongly negative potential is slow and can
        fail to converge at all.
        """
        specs = list(specs)
        if not self.config.chain_initial_state:
            return [self.submit_one(s) for s in specs]

        specs.sort(key=lambda s: (s.target_mu_hartree is None,
                                  s.target_mu_hartree or 0.0), reverse=True)
        submissions: List[JDFTxSubmission] = []
        previous_dir: Optional[str] = None
        for spec in specs:
            submission = self.submit_one(spec, initial_state_from=previous_dir)
            submission.handle.wait()
            previous_dir = submission.run_dir
            submissions.append(submission)
        return submissions

    def submit_parallel(self, specs: Sequence[JDFTxCalcSpec]) -> List[JDFTxSubmission]:
        """Submit all specs at once, no warm-start chaining.

        The right mode for auditing many independent MD frames at one
        potential, which is what the acquisition policy produces.
        """
        return [self.submit_one(s) for s in specs]

    # -- harvest -----------------------------------------------------------

    def harvest(self, submissions: Sequence[JDFTxSubmission],
                wait: bool = True, interval_s: float = 60.0) -> List[CPDFTLabel]:
        labels = []
        for submission in submissions:
            if wait:
                submission.handle.wait(interval_s=interval_s)
            labels.append(harvest_jdftx_run(
                submission.run_dir,
                state_point=submission.spec.state_point,
                sp_id=submission.spec.calc_id,
                output_filename=self.config.output_filename,
                target_mu_ev=submission.spec.target_mu_ev,
                frame_index=submission.spec.frame_index,
                parent_trajectory=submission.spec.parent_trajectory,
                pseudo_set=submission.spec.protocol.pseudo_set,
            ))
        return labels

    def run_and_harvest(self, specs: Sequence[JDFTxCalcSpec],
                        parallel: bool = False,
                        interval_s: float = 60.0) -> List[CPDFTLabel]:
        submissions = (self.submit_parallel(specs) if parallel
                       else self.submit_sweep(specs))
        return self.harvest(submissions, interval_s=interval_s)


def summarise_labels(labels: Sequence[CPDFTLabel]) -> Dict[str, Any]:
    """Post-sweep report. Deliberately surfaces failures first -- a sweep that
    silently loses a third of its points to non-convergence looks fine if you
    only print the mean."""
    total = len(labels)
    converged = [l for l in labels if l.converged]
    usable = [l for l in labels if l.is_usable()]
    mu_errors = [l.mu_error_ev for l in labels if l.mu_error_ev is not None]
    by_status: Dict[str, int] = {}
    for label in labels:
        by_status[label.status.value] = by_status.get(label.status.value, 0) + 1
    return {
        "n_total": total,
        "n_converged": len(converged),
        "n_usable": len(usable),
        "by_status": by_status,
        "max_abs_mu_error_ev": max((abs(e) for e in mu_errors), default=None),
        "mean_abs_mu_error_ev": (sum(abs(e) for e in mu_errors) / len(mu_errors)
                                 if mu_errors else None),
        "failed_ids": [l.state_point_id for l in labels if not l.converged][:20],
    }
