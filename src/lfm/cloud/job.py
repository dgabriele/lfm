"""Cloud job orchestration — ties provider + deployer into a single workflow.

A Job represents the full lifecycle: provision → upload → setup →
launch → monitor → download → terminate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from lfm.cloud.config import CloudConfig
from lfm.cloud.deployer import Deployer
from lfm.cloud.providers.base import CloudProvider, Instance

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A cloud training job."""

    id: str
    instance: Instance
    screen_name: str
    config: CloudConfig
    started_at: float = field(default_factory=time.time)
    status: str = "running"


class JobManager:
    """Manage cloud training jobs end-to-end.

    Orchestrates the provider (instance lifecycle) and deployer
    (SSH-based code upload and job execution).

    Args:
        provider: Cloud provider implementation.
        config: Cloud deployment configuration.
    """

    def __init__(
        self, provider: CloudProvider, config: CloudConfig,
    ) -> None:
        self.provider = provider
        self.config = config
        self.deployer = Deployer(
            ssh_user=config.ssh_user,
            ssh_key_path=config.ssh_key_path,
            remote_workdir=config.remote_workdir,
        )
        self._jobs: dict[str, Job] = {}
        self._state_file = Path(config.local_download_dir) / "jobs.json"

    def launch(
        self,
        config_path: str | None = None,
        upload_data: list[str] | None = None,
        job_name: str = "lfm-train",
    ) -> Job:
        """Full launch workflow: provision → upload → setup → run.

        Args:
            config_path: Local YAML config to upload and use.
            upload_data: Additional local files/dirs to upload.
            job_name: Name for the screen session and logging.

        Returns:
            Job handle for monitoring.
        """
        cfg = self.config

        # 1. Provision instance
        instance = self.provider.launch(
            instance_type=cfg.instance_type,
            region=cfg.region,
            ssh_key_name=cfg.ssh_key_name,
            name=job_name,
        )

        # 2. Upload project code + config + data
        extra = list(upload_data or cfg.upload_data)
        if config_path:
            extra.append(config_path)

        self.deployer.upload_project(
            instance, extra_files=extra,
        )

        # 3. Setup environment
        self.deployer.setup_environment(instance)

        # 4. Launch job
        remote_config = None
        if config_path:
            remote_config = config_path  # same relative path

        screen = self.deployer.launch_job(
            instance,
            command=cfg.command,
            config_path=remote_config,
            job_name=job_name,
        )

        job = Job(
            id=instance.id,
            instance=instance,
            screen_name=screen,
            config=cfg,
        )
        self._jobs[job.id] = job
        self._save_state()
        return job

    def status(self, job: Job) -> str:
        """Check if a job is still running."""
        running = self.deployer.job_running(job.instance, job.screen_name)
        job.status = "running" if running else "completed"
        return job.status

    def logs(self, job: Job, lines: int = 30) -> str:
        """Fetch recent log output."""
        return self.deployer.tail_logs(job.instance, job.screen_name, lines)

    def download(self, job: Job, remote_dir: str | None = None) -> list[str]:
        """Download results from the instance."""
        remote = remote_dir or self.config.remote_output_dir
        return self.deployer.download_results(
            job.instance, remote, self.config.local_download_dir,
        )

    def terminate(self, job: Job) -> None:
        """Terminate the instance."""
        self.provider.terminate(job.instance.id)
        job.status = "terminated"
        self._save_state()
        logger.info("Terminated %s", job.instance.id)

    def wait(
        self,
        job: Job,
        poll_interval: float = 60,
        sync_interval: int = 10,
    ) -> str:
        """Block until the job completes or max_runtime exceeded.

        Downloads checkpoints periodically during training so you
        have a local backup even if the instance dies.

        Args:
            job: Job to monitor.
            poll_interval: Seconds between status checks.
            sync_interval: Download checkpoints every N polls.

        Returns:
            Final status ("completed" or "timeout").
        """
        cfg = self.config
        max_seconds = cfg.max_runtime_hours * 3600
        polls = 0

        while True:
            elapsed = time.time() - job.started_at
            if elapsed > max_seconds:
                logger.warning(
                    "Job %s exceeded max runtime (%.1f hrs), terminating.",
                    job.id, cfg.max_runtime_hours,
                )
                self.download(job)
                if cfg.auto_terminate:
                    self.terminate(job)
                return "timeout"

            s = self.status(job)
            if s != "running":
                logger.info("Job %s finished.", job.id)
                self.download(job)
                if cfg.auto_terminate:
                    self.terminate(job)
                return s

            # Print latest log line
            try:
                tail = self.deployer.tail_logs(
                    job.instance, job.screen_name, 1,
                )
                if tail.strip():
                    logger.info("[%s] %s", job.id[:8], tail.strip())
            except Exception:
                pass

            # Periodic checkpoint sync
            polls += 1
            if polls % sync_interval == 0:
                try:
                    files = self.download(job)
                    if files:
                        logger.info(
                            "Synced %d checkpoint files (%.0f min elapsed)",
                            len(files), elapsed / 60,
                        )
                except Exception as e:
                    logger.debug("Checkpoint sync failed: %s", e)

            time.sleep(poll_interval)

    def _save_state(self) -> None:
        """Persist job state for resume across CLI invocations."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            jid: {
                "instance_id": j.instance.id,
                "ip": j.instance.ip,
                "screen_name": j.screen_name,
                "status": j.status,
                "started_at": j.started_at,
            }
            for jid, j in self._jobs.items()
        }
        self._state_file.write_text(json.dumps(state, indent=2))
