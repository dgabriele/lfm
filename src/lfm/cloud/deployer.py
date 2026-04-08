"""SSH-based deployment: upload code, launch jobs, monitor, download.

Provider-agnostic — works with any CloudProvider that returns an
Instance with an IP address. All remote interaction is over SSH/SFTP.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

from lfm.cloud.providers.base import Instance

logger = logging.getLogger(__name__)

# Directories and patterns to exclude from the source tarball
EXCLUDE_PATTERNS = {
    ".git", "__pycache__", ".venv", "*.pyc", ".mypy_cache",
    ".ruff_cache", "output", "*.egg-info", ".eggs",
}

# Top-level dirs to exclude (not subdirs like src/lfm/data/)
EXCLUDE_TOP_DIRS = {"data", ".git"}

# Remote setup script — installs deps and prepares environment
SETUP_SCRIPT = """\
set -e
cd {workdir}
if [ ! -f pyproject.toml ]; then
    echo "ERROR: pyproject.toml not found in {workdir}"
    exit 1
fi
pip install poetry
poetry install --no-interaction
echo "SETUP_COMPLETE"
"""


class Deployer:
    """Deploy and manage training jobs on cloud instances via SSH.

    Handles the full lifecycle: upload → setup → launch → monitor →
    download. Provider-agnostic — just needs an Instance with an IP.

    Args:
        ssh_user: SSH username (default: ubuntu).
        ssh_key_path: Path to SSH private key.
        remote_workdir: Working directory on the remote instance.
    """

    def __init__(
        self,
        ssh_user: str = "ubuntu",
        ssh_key_path: str = "~/.ssh/id_ed25519",
        remote_workdir: str = "/home/ubuntu/lfm",
    ) -> None:
        self.ssh_user = ssh_user
        self.ssh_key_path = os.path.expanduser(ssh_key_path)
        self.remote_workdir = remote_workdir

    def _connect(self, instance: Instance) -> paramiko.SSHClient:
        """Open SSH connection to instance with retry.

        Handles RunPod-style ``ip:port`` addresses and standard
        ``ip``-only addresses (port 22).
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Parse ip:port (RunPod) vs plain ip (Lambda)
        if ":" in instance.ip:
            hostname, port_str = instance.ip.rsplit(":", 1)
            port = int(port_str)
        else:
            hostname = instance.ip
            port = 22

        for attempt in range(10):
            try:
                ssh.connect(
                    hostname=hostname,
                    port=port,
                    username=self.ssh_user,
                    key_filename=self.ssh_key_path,
                    timeout=30,
                )
                return ssh
            except Exception as e:
                if attempt == 9:
                    raise ConnectionError(
                        f"Cannot SSH to {hostname}:{port} after 10 attempts: {e}"
                    )
                logger.info(
                    "SSH attempt %d/10 failed, retrying in 15s...",
                    attempt + 1,
                )
                time.sleep(15)
        raise RuntimeError("unreachable")

    def _exec(
        self, ssh: paramiko.SSHClient, cmd: str, timeout: int = 900,
    ) -> tuple[int, str, str]:
        """Execute command and return (exit_code, stdout, stderr)."""
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return exit_code, stdout.read().decode(), stderr.read().decode()

    def _create_tarball(self, project_root: Path) -> str:
        """Create a tarball of the project source code."""
        tmp = tempfile.mktemp(suffix=".tar.gz")

        def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            name = info.name
            # Check top-level directory exclusions (e.g., lfm/data/ but not lfm/src/lfm/data/)
            parts = name.split("/")
            if len(parts) >= 2 and parts[1] in EXCLUDE_TOP_DIRS:
                return None
            for pat in EXCLUDE_PATTERNS:
                if pat.startswith("*"):
                    if name.endswith(pat[1:]):
                        return None
                elif f"/{pat}/" in f"/{name}/":
                    return None
            return info

        with tarfile.open(tmp, "w:gz") as tar:
            tar.add(str(project_root), arcname="lfm", filter=_filter)

        size_mb = os.path.getsize(tmp) / (1024 * 1024)
        logger.info("Created tarball: %.1f MB", size_mb)
        return tmp

    def _scp(self, instance: Instance, local: str, remote: str) -> None:
        """Upload a file via scp subprocess (reliable for large files)."""
        if ":" in instance.ip:
            hostname, port = instance.ip.rsplit(":", 1)
        else:
            hostname, port = instance.ip, "22"

        cmd = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-P", port,
            "-i", self.ssh_key_path,
            local,
            f"{self.ssh_user}@{hostname}:{remote}",
        ]
        logger.info("scp %s → %s:%s", local, hostname, remote)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            # Filter out SSH banners from stderr
            err = "\n".join(
                l for l in result.stderr.splitlines()
                if not any(x in l for x in ("Welcome to", "Have fun", "Permanently added"))
            ).strip()
            if err:
                raise RuntimeError(f"scp failed (exit {result.returncode}): {err}")

    def upload_project(
        self,
        instance: Instance,
        project_root: Path | None = None,
        extra_files: list[str] | None = None,
    ) -> None:
        """Upload project source code and optional data files.

        Uses scp for file transfer (reliable for large files).

        Args:
            instance: Target instance.
            project_root: Local project root. Defaults to cwd.
            extra_files: Additional local files to upload (e.g., data).
        """
        project_root = project_root or Path.cwd()
        tarball = self._create_tarball(project_root)

        ssh = self._connect(instance)
        try:
            # Create workdir
            self._exec(ssh, f"mkdir -p {self.remote_workdir}")
        finally:
            ssh.close()

        try:
            # Upload tarball via scp
            remote_tar = "/tmp/lfm_upload.tar.gz"
            self._scp(instance, tarball, remote_tar)

            # Extract
            ssh = self._connect(instance)
            try:
                self._exec(ssh, (
                    f"cd {self.remote_workdir} && "
                    f"tar -xzf {remote_tar} --strip-components=1 && "
                    f"rm {remote_tar}"
                ))
            finally:
                ssh.close()

            # Upload extra files via scp
            for local_path in extra_files or []:
                local = Path(local_path)
                remote = f"{self.remote_workdir}/{local}"
                # Create parent dirs
                ssh = self._connect(instance)
                try:
                    self._exec(ssh, f"mkdir -p $(dirname {remote})")
                finally:
                    ssh.close()
                self._scp(instance, str(local), remote)
        finally:
            os.unlink(tarball)

    def setup_environment(self, instance: Instance) -> None:
        """Install dependencies on the remote instance."""
        ssh = self._connect(instance)
        try:
            logger.info("Setting up environment on %s...", instance.ip)
            script = SETUP_SCRIPT.format(workdir=self.remote_workdir)
            exit_code, stdout, stderr = self._exec(ssh, script, timeout=600)
            if exit_code != 0 or "SETUP_COMPLETE" not in stdout:
                raise RuntimeError(
                    f"Setup failed (exit {exit_code}):\n{stderr}\n{stdout}"
                )
            logger.info("Environment ready.")
        finally:
            ssh.close()

    def launch_job(
        self,
        instance: Instance,
        command: str,
        config_path: str | None = None,
        job_name: str = "lfm",
        env: dict[str, str] | None = None,
    ) -> str:
        """Launch a training job in a detached screen session.

        Args:
            instance: Target instance.
            command: The lfm CLI command (e.g., "lfm translate pretrain").
            config_path: Remote path to YAML config (appended to command).
            job_name: Screen session name.
            env: Extra environment variables.

        Returns:
            The screen session name (used as job handle).
        """
        ssh = self._connect(instance)
        try:
            # Build the full command
            parts = [f"cd {self.remote_workdir}"]

            # Export env vars
            for k, v in (env or {}).items():
                parts.append(f"export {k}={v}")

            run_cmd = f"poetry run {command}"
            if config_path:
                run_cmd += f" {config_path}"

            log_path = f"{self.remote_workdir}/{job_name}.log"
            parts.append(f"{run_cmd} > {log_path} 2>&1")

            full_cmd = " && ".join(parts)
            nohup_cmd = f"nohup bash -c '{full_cmd}' &"

            exit_code, _, stderr = self._exec(ssh, nohup_cmd)
            if exit_code != 0:
                raise RuntimeError(f"Failed to launch: {stderr}")

            logger.info(
                "Job '%s' launched on %s. Logs: %s",
                job_name, instance.ip, log_path,
            )
            return job_name
        finally:
            ssh.close()

    def job_running(self, instance: Instance, job_name: str) -> bool:
        """Check if the training job is still running."""
        ssh = self._connect(instance)
        try:
            _, stdout, _ = self._exec(
                ssh, f"pgrep -f '{job_name}' || true",
            )
            return stdout.strip() != ""
        finally:
            ssh.close()

    def tail_logs(
        self, instance: Instance, job_name: str, lines: int = 20,
    ) -> str:
        """Fetch the last N lines of the job log."""
        ssh = self._connect(instance)
        try:
            log_path = f"{self.remote_workdir}/{job_name}.log"
            _, stdout, _ = self._exec(ssh, f"tail -n {lines} {log_path}")
            return stdout
        finally:
            ssh.close()

    def download_results(
        self,
        instance: Instance,
        remote_dir: str,
        local_dir: str,
    ) -> list[str]:
        """Download output files from the remote instance.

        Args:
            instance: Source instance.
            remote_dir: Remote directory to download from.
            local_dir: Local directory to save to.

        Returns:
            List of downloaded file paths.
        """
        ssh = self._connect(instance)
        downloaded = []
        try:
            sftp = ssh.open_sftp()
            local = Path(local_dir)
            local.mkdir(parents=True, exist_ok=True)

            # List remote files
            for entry in sftp.listdir_attr(remote_dir):
                remote_path = f"{remote_dir}/{entry.filename}"
                local_path = local / entry.filename
                logger.info("Downloading %s...", entry.filename)
                sftp.get(remote_path, str(local_path))
                downloaded.append(str(local_path))

            sftp.close()
        finally:
            ssh.close()

        logger.info("Downloaded %d files to %s", len(downloaded), local_dir)
        return downloaded
