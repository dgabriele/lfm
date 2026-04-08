"""Cloud subcommand group for ``lfm cloud {launch,status,logs,terminate,types}``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


def _get_provider(args):
    """Build a cloud provider from CLI args or config."""
    provider_name = getattr(args, "provider", None) or "lambda_labs"
    api_key = getattr(args, "api_key", None)

    if provider_name == "runpod":
        from lfm.cloud.providers.runpod import RunPodProvider
        return RunPodProvider(api_key=api_key)

    if provider_name == "vastai":
        from lfm.cloud.providers.vastai import VastAIProvider
        return VastAIProvider(api_key=api_key)

    from lfm.cloud.providers.lambda_labs import LambdaLabsProvider
    return LambdaLabsProvider(api_key=api_key)


def _get_manager(args):
    """Build a JobManager from CLI args + YAML config."""
    import yaml

    from lfm.cloud.config import CloudConfig
    from lfm.cloud.job import JobManager

    cfg_dict: dict = {}
    config_path = getattr(args, "cloud_config", None)
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        cfg_dict = raw.get("cloud", raw)

    # CLI overrides
    for key in ("instance_type", "region", "ssh_key_name", "command", "provider"):
        val = getattr(args, key, None)
        if val is not None:
            cfg_dict[key] = val

    config = CloudConfig(**cfg_dict)
    # Provider from CLI flag or config
    if not getattr(args, "provider", None):
        args.provider = config.provider
    provider = _get_provider(args)
    return JobManager(provider, config)


class LaunchCommand(CLICommand):
    """Launch a training job on a cloud GPU."""

    @property
    def name(self) -> str:
        return "launch"

    @property
    def help(self) -> str:
        return "Launch a training job on a cloud GPU instance"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file for training")
        parser.add_argument("--cloud-config", default=None,
                            help="Cloud deployment YAML (or embed under 'cloud:' key)")
        parser.add_argument("--provider", default=None,
                            choices=["lambda_labs", "runpod", "vastai"],
                            help="Cloud provider (default: lambda_labs)")
        parser.add_argument("--instance-type", default=None)
        parser.add_argument("--region", default=None)
        parser.add_argument("--ssh-key-name", default=None)
        parser.add_argument("--command", default=None,
                            help="LFM CLI command (e.g., 'lfm translate pretrain')")
        parser.add_argument("--upload", nargs="*", default=[],
                            help="Additional files to upload")
        parser.add_argument("--name", default="lfm-train")
        parser.add_argument("--api-key", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        manager = _get_manager(args)
        job = manager.launch(
            config_path=args.config,
            upload_data=args.upload,
            job_name=args.name,
        )
        print(f"Job launched: {job.id} @ {job.instance.ip}")
        print(f"Monitor: lfm cloud logs {job.id}")
        print(f"Download: lfm cloud download {job.id}")

        # Block and wait
        final = manager.wait(job)
        print(f"Job {job.id}: {final}")
        return 0 if final == "completed" else 1


class StatusCommand(CLICommand):
    """Check status of cloud instances."""

    @property
    def name(self) -> str:
        return "status"

    @property
    def help(self) -> str:
        return "List active cloud instances"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--provider", default=None,
                            choices=["lambda_labs", "runpod", "vastai"])
        parser.add_argument("--api-key", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        provider = _get_provider(args)
        instances = provider.list_instances()
        if not instances:
            print("No active instances.")
            return 0
        for inst in instances:
            print(
                f"  {inst.id[:12]}  {inst.instance_type:30s}  "
                f"{inst.status:10s}  {inst.ip}  {inst.name or ''}"
            )
        return 0


class TypesCommand(CLICommand):
    """List available GPU instance types."""

    @property
    def name(self) -> str:
        return "types"

    @property
    def help(self) -> str:
        return "List available GPU instance types and pricing"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--provider", default=None,
                            choices=["lambda_labs", "runpod", "vastai"])
        parser.add_argument("--api-key", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        provider = _get_provider(args)
        types = provider.list_instance_types()
        if not types:
            print("No instance types available.")
            return 0
        for t in sorted(types, key=lambda x: x["price_cents_per_hour"]):
            price = t["price_cents_per_hour"] / 100
            regions = ", ".join(t["regions"][:3]) or "none"
            print(
                f"  {t['name']:35s}  ${price:.2f}/hr  "
                f"{t['gpu_memory_gb']}GB  [{regions}]"
            )
        return 0


class TerminateCommand(CLICommand):
    """Terminate a cloud instance."""

    @property
    def name(self) -> str:
        return "terminate"

    @property
    def help(self) -> str:
        return "Terminate a cloud GPU instance"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("instance_id", help="Instance ID to terminate")
        parser.add_argument("--api-key", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        provider = _get_provider(args)
        provider.terminate(args.instance_id)
        print(f"Terminated {args.instance_id}")
        return 0


def register_cloud_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``cloud`` subcommand group."""
    cloud_parser = parent_subparsers.add_parser(
        "cloud",
        help="Cloud GPU deployment",
        description="Launch and manage training jobs on cloud GPUs.",
    )
    cloud_subparsers = cloud_parser.add_subparsers(
        title="cloud commands",
        description="Available cloud commands",
        dest="cloud_cmd",
    )

    commands = [
        LaunchCommand(),
        StatusCommand(),
        TypesCommand(),
        TerminateCommand(),
    ]

    for cmd in commands:
        sub = cloud_subparsers.add_parser(
            cmd.name, help=cmd.help, description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    cloud_parser.set_defaults(
        command_handler=type(
            "_CloudHelp", (),
            {"execute": staticmethod(
                lambda _args: cloud_parser.print_help() or 0,
            )},
        )()
    )
