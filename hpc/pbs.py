"""
PBS Pro job scripting, submission, and polling for ALCF Aurora.

Submission uses `qsub`, polling uses `qstat -f -F json`. Both are assumed
to be on PATH (they are, on Aurora login nodes). Nothing here has been run
against a live scheduler.

Job-array note: PBS Pro supports `-J 0-N`, but this module deliberately
does NOT use it. A campaign's calculations differ in node count and
walltime, and one oversized array element wastes an entire allocation.
Submit heterogeneous work as separate jobs, or pack many small calculations
into one job with hpc.launcher's bundling.
"""

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from hpc.aurora import AuroraConfig

# PBS states. R/Q/H are live; F/E are terminal. Anything else is treated as
# live so a transient/unknown state never looks like success.
LIVE_STATES = {"Q", "R", "H", "W", "T", "S", "B", "M"}
TERMINAL_STATES = {"F", "E", "X"}


@dataclass
class PBSJobSpec:
    """One PBS job: what to run, where, and with which resources."""

    name: str
    commands: List[str]                       # bash lines, run in order
    aurora: AuroraConfig
    work_dir: str
    log_dir: Optional[str] = None
    # Lines injected after module loads and env exports, before `commands`.
    # Use for `source <venv>/bin/activate`, PYTHONPATH tweaks, etc.
    preamble: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    # Fail the whole script on the first non-zero exit rather than plowing on
    # and producing half-written outputs that the harvester would then treat
    # as real data.
    strict: bool = True
    depends_on: Optional[str] = None          # PBS job id; afterok dependency

    def script(self) -> str:
        aurora = self.aurora
        log_dir = Path(self.log_dir or self.work_dir)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)

        lines: List[str] = [
            "#!/bin/bash",
            f"#PBS -N {safe_name}",
            f"#PBS -A {aurora.project}",
            f"#PBS -q {aurora.queue}",
            f"#PBS -l select={aurora.nodes}",
            f"#PBS -l walltime={aurora.walltime}",
            f"#PBS -l filesystems={aurora.filesystems}",
            f"#PBS -o {log_dir / (safe_name + '.out')}",
            f"#PBS -e {log_dir / (safe_name + '.err')}",
            "",
        ]
        if self.strict:
            # pipefail matters: `jdftx ... | tee out` would otherwise mask a
            # JDFTx failure behind tee's exit status.
            lines += ["set -euo pipefail", ""]

        lines += [
            f"cd {shlex.quote(str(self.work_dir))}",
            "",
            "echo \"[pbs] job=${PBS_JOBID:-none} host=$(hostname) start=$(date -Is)\"",
            f"echo \"[pbs] {aurora.describe()}\"",
            "",
        ]
        lines += aurora.module_lines()
        lines += aurora.export_lines(self.extra_env)
        lines += [""]
        if self.preamble:
            lines += list(self.preamble) + [""]
        lines += list(self.commands)
        lines += [
            "",
            "echo \"[pbs] end=$(date -Is) rc=$?\"",
        ]
        return "\n".join(lines) + "\n"

    def write(self, path: Optional[str] = None) -> Path:
        target = Path(path) if path else Path(self.work_dir) / f"{self.name}.pbs"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.script())
        target.chmod(0o755)
        return target


@dataclass
class PBSJob:
    job_id: str
    name: str
    script_path: str
    work_dir: str
    state: str = "Q"
    exit_status: Optional[int] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.is_terminal and self.exit_status == 0


def submit(spec: PBSJobSpec, script_path: Optional[str] = None,
           dry_run: bool = False) -> PBSJob:
    """Write the script and `qsub` it. With dry_run=True, write the script,
    print nothing, submit nothing, and return a job with a sentinel id --
    which is how every campaign should be rehearsed before it burns hours."""
    Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
    if spec.log_dir:
        Path(spec.log_dir).mkdir(parents=True, exist_ok=True)
    path = spec.write(script_path)

    if dry_run:
        return PBSJob(job_id=f"dryrun-{spec.name}", name=spec.name,
                      script_path=str(path), work_dir=spec.work_dir, state="Q")

    cmd = ["qsub"]
    if spec.depends_on:
        cmd += ["-W", f"depend=afterok:{spec.depends_on}"]
    cmd.append(str(path))

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=spec.work_dir)
    if result.returncode != 0:
        raise RuntimeError(
            f"qsub failed for {spec.name} (rc={result.returncode}).\n"
            f"  script: {path}\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    job_id = result.stdout.strip().splitlines()[-1].strip()
    if not job_id:
        raise RuntimeError(f"qsub returned no job id for {spec.name}.")
    return PBSJob(job_id=job_id, name=spec.name, script_path=str(path),
                  work_dir=spec.work_dir, state="Q")


def poll(job: PBSJob) -> PBSJob:
    """Refresh `job.state` and `job.exit_status` from qstat, in place.

    A job that has left the queue entirely (qstat non-zero, or absent from
    the JSON) is treated as finished. Exit status is then unknown, so it is
    left as-is rather than guessed -- the harvester decides success by
    inspecting outputs, which is the only trustworthy signal anyway.
    """
    if job.job_id.startswith("dryrun-"):
        job.state, job.exit_status = "F", 0
        return job

    result = subprocess.run(["qstat", "-f", "-F", "json", job.job_id],
                            capture_output=True, text=True)
    if result.returncode != 0:
        job.state = "F"
        return job
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        job.state = "F"
        return job

    entry = (payload.get("Jobs") or {}).get(job.job_id)
    if entry is None:
        jobs = payload.get("Jobs") or {}
        entry = next((v for k, v in jobs.items()
                      if k.split(".")[0] == job.job_id.split(".")[0]), None)
    if entry is None:
        job.state = "F"
        return job

    job.state = entry.get("job_state", job.state)
    if "Exit_status" in entry:
        try:
            job.exit_status = int(entry["Exit_status"])
        except (TypeError, ValueError):
            job.exit_status = None
    return job


def wait(jobs: List[PBSJob], interval_s: float = 60.0,
         timeout_s: Optional[float] = None,
         on_update=None) -> List[PBSJob]:
    """Block until every job reaches a terminal state.

    `on_update(jobs)` is called after each poll round -- the campaign driver
    uses it to stream progress. Raises TimeoutError rather than returning
    partial results silently, so a caller cannot mistake a timeout for
    completion.
    """
    start = time.time()
    pending = [j for j in jobs if not j.is_terminal]
    while pending:
        for job in pending:
            poll(job)
        if on_update:
            on_update(jobs)
        pending = [j for j in jobs if not j.is_terminal]
        if not pending:
            break
        if timeout_s is not None and (time.time() - start) > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s with {len(pending)} job(s) "
                f"still live: {[j.job_id for j in pending]}"
            )
        time.sleep(interval_s)
    return jobs


def current_job_id() -> Optional[str]:
    """PBS job id if this process is itself running inside a PBS job."""
    return os.environ.get("PBS_JOBID")


def inside_pbs() -> bool:
    return current_job_id() is not None


def nodefile_hosts() -> List[str]:
    """Hostnames from $PBS_NODEFILE, for splitting one allocation across
    several concurrent sub-jobs (see hpc.launcher bundling)."""
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile or not Path(nodefile).exists():
        return []
    seen, hosts = set(), []
    for line in Path(nodefile).read_text().splitlines():
        host = line.strip()
        if host and host not in seen:
            seen.add(host)
            hosts.append(host)
    return hosts
