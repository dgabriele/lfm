"""Configuration for cloud GPU deployment."""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class CloudConfig(LFMBaseConfig):
    """Cloud deployment configuration.

    Can be embedded in a training YAML or used standalone::

        cloud:
          provider: lambda_labs
          instance_type: gpu_1x_a100_sxm4
          region: us-west-1
          ssh_key_name: lfm-key
          auto_terminate: true
    """

    provider: str = "lambda_labs"
    instance_type: str = "gpu_1x_a100_sxm4"
    region: str = "us-west-1"
    ssh_key_name: str = "lfm-key"
    ssh_user: str = "ubuntu"
    ssh_key_path: str = "~/.ssh/id_ed25519"

    # Job configuration
    command: str = "lfm translate pretrain"
    config_path: str | None = None
    upload_data: list[str] = []

    # Lifecycle
    auto_terminate: bool = True
    max_runtime_hours: float = 24.0

    # Remote paths
    remote_workdir: str = "/home/ubuntu/lfm"
    remote_output_dir: str = "/home/ubuntu/lfm/output"
    local_download_dir: str = "data/cloud_results"
