"""
One interface over "run it right here" and "submit it to PBS".

Drivers (cp_dft.jdftx_driver, md.cp_md_driver, ...) should never call qsub
or subprocess directly. They build a command and hand it to a Launcher.
That way the same driver works:

  * interactively on a login node during setup       -> LocalLauncher
  * as one PBS job per calculation                   -> PBSLauncher
  * as many calculations packed inside one big
    allocation, each on its own node subset          -> PBSLauncher(bundle=...)
                                                        + BundledLauncher

The third mode is the DEFAULT for this project, not a convenience.

Because JDFTx is ported to SYCL and runs GPU-native on Aurora PVC, constant-
potential MD and the grand-canonical DFT audits it triggers execute on the same
machine. Bundling therefore lets one allocation hold the whole acquisition loop:
MD on some host groups, DFT audits on others, concurrently, with no queue
round-trip between them. Loop latency drops from queue cycles to the wall time
of one MD window plus one audit -- which is the difference between an adaptive
instrument and a batch sweep.

It also suits Aurora queue policy, which rewards a few large jobs over many
small ones, while a sigma_mu round naturally produces dozens of independent
1-2 node JDFTx single points.
"""

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from hpc.aurora import AuroraConfig
from hpc.pbs import PBSJob, PBSJobSpec, nodefile_hosts, submit, wait


@dataclass
class JobHandle:
    """What a Launcher hands back. `wait()` blocks; `ok` reports success."""
    name: str
    work_dir: str
    backend: str                       # "local" | "pbs" | "bundle"
    pbs_job: Optional[PBSJob] = None
    returncode: Optional[int] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        if self.pbs_job is not None:
            return self.pbs_job.succeeded or (
                self.pbs_job.is_terminal and self.pbs_job.exit_status is None
                and self.returncode in (None, 0))
        return self.returncode == 0

    def wait(self, interval_s: float = 60.0,
             timeout_s: Optional[float] = None) -> "JobHandle":
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise TimeoutError(f"{self.name} still running after {timeout_s}s")
        if self.pbs_job is not None:
            wait([self.pbs_job], interval_s=interval_s, timeout_s=timeout_s)
            self.returncode = self.pbs_job.exit_status
        return self


class Launcher:
    """Base interface. `run` returns immediately; call `handle.wait()`."""

    def run(self, name: str, commands: List[str], work_dir: str,
            aurora: Optional[AuroraConfig] = None,
            preamble: Optional[List[str]] = None,
            extra_env: Optional[Dict[str, str]] = None,
            log_dir: Optional[str] = None,
            depends_on: Optional[str] = None) -> JobHandle:
        raise NotImplementedError

    def run_all(self, jobs: List[Dict], interval_s: float = 60.0) -> List[JobHandle]:
        """Launch every job, then wait for all of them.

        Subclasses that own a finite pool of nodes MUST override this to
        throttle -- see BundledLauncher.
        """
        handles = [self.run(**spec) for spec in jobs]
        for handle in handles:
            handle.wait(interval_s=interval_s)
        return handles


class LocalLauncher(Launcher):
    """Run in a background thread on this machine.

    For login-node setup work, single-node debugging, and the CI path where
    no scheduler exists. It ignores `aurora` resource fields except to apply
    the environment -- it cannot allocate nodes it does not have.
    """

    def __init__(self, shell: str = "/bin/bash", dry_run: bool = False):
        self.shell = shell
        self.dry_run = dry_run

    def run(self, name, commands, work_dir, aurora=None, preamble=None,
            extra_env=None, log_dir=None, depends_on=None) -> JobHandle:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        logs = Path(log_dir or work_dir)
        logs.mkdir(parents=True, exist_ok=True)
        out_path, err_path = logs / f"{name}.out", logs / f"{name}.err"

        lines = ["set -euo pipefail", f"cd {work}"]
        if aurora is not None:
            lines += aurora.module_lines() + aurora.export_lines(extra_env or {})
        elif extra_env:
            lines += [f'export {k}="{v}"' for k, v in sorted(extra_env.items())]
        lines += list(preamble or []) + list(commands)
        script = "\n".join(lines) + "\n"

        script_path = work / f"{name}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        handle = JobHandle(name=name, work_dir=str(work), backend="local",
                           stdout_path=str(out_path), stderr_path=str(err_path))
        if self.dry_run:
            handle.returncode = 0
            return handle

        def _target():
            with open(out_path, "w") as out, open(err_path, "w") as err:
                proc = subprocess.run([self.shell, str(script_path)],
                                      stdout=out, stderr=err, cwd=str(work))
            handle.returncode = proc.returncode

        thread = threading.Thread(target=_target, daemon=True)
        handle._thread = thread
        thread.start()
        return handle


class PBSLauncher(Launcher):
    """One PBS job per call.

    `default_aurora` supplies project/queue/walltime; per-call `aurora`
    overrides it wholesale, which is how a 9-node JDFTx job and a 1-node MD
    job coexist in the same campaign.
    """

    def __init__(self, default_aurora: AuroraConfig, dry_run: bool = False):
        self.default_aurora = default_aurora
        self.dry_run = dry_run

    def run(self, name, commands, work_dir, aurora=None, preamble=None,
            extra_env=None, log_dir=None, depends_on=None) -> JobHandle:
        spec = PBSJobSpec(
            name=name,
            commands=list(commands),
            aurora=aurora or self.default_aurora,
            work_dir=work_dir,
            log_dir=log_dir,
            preamble=list(preamble or []),
            extra_env=dict(extra_env or {}),
            depends_on=depends_on,
        )
        job = submit(spec, dry_run=self.dry_run)
        return JobHandle(name=name, work_dir=work_dir, backend="pbs", pbs_job=job,
                         stdout_path=str(Path(log_dir or work_dir) / f"{name}.out"),
                         stderr_path=str(Path(log_dir or work_dir) / f"{name}.err"))


class BundledLauncher(Launcher):
    """Run several calculations CONCURRENTLY inside one existing allocation.

    Use from a driver script that is itself already running inside a PBS job.
    It slices $PBS_NODEFILE into disjoint host groups and pins each command to
    its own group via `mpiexec --hosts`, so N independent DFT single points
    share one allocation without fighting over nodes.

    Raises if not inside PBS -- silently degrading to serial execution inside
    what the user believes is a parallel bundle is the kind of bug that only
    shows up as a burned allocation.
    """

    def __init__(self, nodes_per_task: int, shell: str = "/bin/bash"):
        hosts = nodefile_hosts()
        if not hosts:
            raise RuntimeError(
                "BundledLauncher requires $PBS_NODEFILE -- run it from inside a "
                "PBS job. On a login node use PBSLauncher (submits jobs) or "
                "LocalLauncher (runs here) instead."
            )
        if nodes_per_task <= 0:
            raise ValueError("nodes_per_task must be positive.")
        self.hosts = hosts
        self.nodes_per_task = nodes_per_task
        self.shell = shell
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """How many tasks can run at once in this allocation."""
        return len(self.hosts) // self.nodes_per_task

    def _take_hosts(self) -> List[str]:
        """Claim a disjoint host group. Wraps once the pool is exhausted.

        Wrapping means oversubscription, so callers must stay within
        `capacity` -- `run_all` does. A bare `run()` past capacity is allowed
        deliberately (a caller may know two tasks are tiny and can share), but
        it is the caller's judgement, not an accident.
        """
        with self._lock:
            if self._cursor + self.nodes_per_task > len(self.hosts):
                self._cursor = 0
            group = self.hosts[self._cursor:self._cursor + self.nodes_per_task]
            self._cursor += self.nodes_per_task
            return group

    def run_all(self, jobs: List[Dict], interval_s: float = 5.0) -> List[JobHandle]:
        """Run every job, at most `capacity` at a time, preserving input order.

        This override is REQUIRED for correctness. The base implementation
        launches everything at once; here that would wrap the host cursor and
        place several tasks on the same nodes, so they would fight for the same
        PVC tiles. The symptom is not a crash -- it is silently degraded
        throughput and, for GPU-resident JDFTx, possible out-of-memory on tiles
        that were meant to host one rank set. Throttling is what makes
        co-locating MD and DFT audits in one allocation safe.
        """
        results: List[Optional[JobHandle]] = [None] * len(jobs)
        pending = list(enumerate(jobs))
        running: List[tuple] = []

        while pending or running:
            while pending and len(running) < self.capacity:
                index, spec = pending.pop(0)
                running.append((index, self.run(**spec)))
            still: List[tuple] = []
            for index, handle in running:
                thread = handle._thread
                if thread is not None and thread.is_alive():
                    still.append((index, handle))
                else:
                    results[index] = handle
            if len(still) == len(running) and still:
                still[0][1].wait(interval_s=interval_s)
            running = still
        return [h for h in results if h is not None]

    def run(self, name, commands, work_dir, aurora=None, preamble=None,
            extra_env=None, log_dir=None, depends_on=None) -> JobHandle:
        group = self._take_hosts()
        host_env = dict(extra_env or {})
        # Drivers read ELECTROCHEM_MPI_HOSTS and append `--hosts <list>` to
        # their mpiexec line; see cp_dft.jdftx_driver.
        host_env["ELECTROCHEM_MPI_HOSTS"] = ",".join(group)
        host_env["ELECTROCHEM_MPI_NODES"] = str(len(group))
        local = LocalLauncher(shell=self.shell)
        handle = local.run(name, commands, work_dir, aurora=aurora,
                           preamble=preamble, extra_env=host_env, log_dir=log_dir)
        handle.backend = "bundle"
        return handle


def make_launcher(mode: str, aurora: Optional[AuroraConfig] = None,
                  nodes_per_task: int = 1, dry_run: bool = False) -> Launcher:
    """mode: 'local' | 'pbs' | 'bundle'."""
    if mode == "local":
        return LocalLauncher(dry_run=dry_run)
    if mode == "pbs":
        if aurora is None:
            raise ValueError("mode='pbs' requires an AuroraConfig.")
        return PBSLauncher(aurora, dry_run=dry_run)
    if mode == "bundle":
        return BundledLauncher(nodes_per_task=nodes_per_task)
    raise ValueError(f"Unknown launcher mode {mode!r}; expected local|pbs|bundle.")
