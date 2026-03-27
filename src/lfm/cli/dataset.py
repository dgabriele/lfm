"""Dataset subcommand group for ``lfm dataset {generate,list}``."""

from __future__ import annotations

import argparse
from pathlib import Path

from lfm.cli.base import CLICommand


class GenerateCommand(CLICommand):
    """Generate an HDF5 dataset from a corpus source."""

    @property
    def name(self) -> str:
        return "generate"

    @property
    def help(self) -> str:
        return "Generate HDF5 dataset from corpus source"

    @property
    def description(self) -> str:
        return (
            "Load raw text from a corpus source, sanitize, convert to IPA, "
            "and write a compressed HDF5 dataset for pretraining."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--source", default="leipzig",
            help="Corpus source name (default: leipzig)",
        )
        parser.add_argument(
            "--output", default="",
            help="Output directory (default: data/datasets/<source>)",
        )
        parser.add_argument(
            "--languages", nargs="+", default=[],
            help="ISO 639-3 language codes to include (default: all)",
        )
        parser.add_argument(
            "--max-samples", type=int, default=50000,
            help="Per-language sample cap (default: 50000)",
        )
        parser.add_argument(
            "--min-samples", type=int, default=100,
            help="Minimum samples per language (default: 100)",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed (default: 42)",
        )
        parser.add_argument(
            "--num-workers", type=int, default=None,
            help="Parallel workers (default: auto)",
        )

        # LLM gate
        parser.add_argument(
            "--no-llm-gate", action="store_true",
            help="Disable LLM quality gatekeeper",
        )
        parser.add_argument(
            "--llm-gate-model", default="Qwen/Qwen2.5-0.5B",
            help="LLM gatekeeper model (default: Qwen/Qwen2.5-0.5B)",
        )

        # Constituency extraction
        parser.add_argument(
            "--extract-constituents", action="store_true",
            help="Augment with phrase constituents (NP, VP, PP, etc.)",
        )
        parser.add_argument(
            "--min-constituent-length", type=int, default=10,
            help="Min character length for extracted constituents (default: 10)",
        )

        # Sanitize overrides (--sanitize-* prefix)
        san = parser.add_argument_group("sanitization", "Override sanitization settings")
        san.add_argument("--sanitize-number-policy", default="spell_out",
                         choices=["reject", "strip", "keep", "spell_out"],
                         help="Number handling (default: spell_out)")
        san.add_argument("--sanitize-symbol-policy", default="spell_out",
                         choices=["reject", "strip", "keep", "spell_out"],
                         help="Greek/math symbol handling (default: spell_out)")
        san.add_argument("--sanitize-alpha-ratio-min", type=float, default=0.7,
                         help="Min alphabetic ratio (default: 0.7)")
        san.add_argument("--sanitize-max-digit-ratio", type=float, default=0.0,
                         help="Max digit ratio (default: 0.0)")
        san.add_argument("--sanitize-max-foreign-script-ratio", type=float, default=0.3,
                         help="Max foreign script ratio (default: 0.3)")
        san.add_argument("--sanitize-require-terminal-punctuation",
                         action="store_true", default=True,
                         help="Require terminal punctuation (default: true)")
        san.add_argument("--sanitize-no-terminal-punctuation",
                         action="store_true",
                         help="Do not require terminal punctuation")
        san.add_argument("--sanitize-min-line-length", type=int, default=20,
                         help="Min line length in characters (default: 20)")
        san.add_argument("--sanitize-max-line-length", type=int, default=500,
                         help="Max line length in characters (default: 500)")

        # Source-specific config
        parser.add_argument(
            "--data-dir", default=None,
            help="Corpus data directory (forwarded to source loader)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.data.dataset.config import DatasetGenerateConfig, LLMGateConfig
        from lfm.data.dataset.generator import DatasetGenerator
        from lfm.data.sanitize import SanitizeConfig

        # Build sanitize config
        require_punct = args.sanitize_require_terminal_punctuation
        if args.sanitize_no_terminal_punctuation:
            require_punct = False

        sanitize_cfg = SanitizeConfig(
            number_policy=args.sanitize_number_policy,
            symbol_policy=args.sanitize_symbol_policy,
            alpha_ratio_min=args.sanitize_alpha_ratio_min,
            max_digit_ratio=args.sanitize_max_digit_ratio,
            max_foreign_script_ratio=args.sanitize_max_foreign_script_ratio,
            require_terminal_punctuation=require_punct,
            min_line_length=args.sanitize_min_line_length,
            max_line_length=args.sanitize_max_line_length,
        )

        # Build LLM gate config
        llm_gate_cfg = LLMGateConfig(
            enabled=not args.no_llm_gate,
            model_name=args.llm_gate_model,
        )

        # Source config
        source_config: dict = {}
        if args.data_dir:
            source_config["data_dir"] = args.data_dir

        config = DatasetGenerateConfig(
            source=args.source,
            source_config=source_config,
            output=args.output,
            languages=args.languages,
            max_samples=args.max_samples,
            min_samples=args.min_samples,
            sanitize=sanitize_cfg,
            llm_gate=llm_gate_cfg,
            extract_constituents=args.extract_constituents,
            min_constituent_length=args.min_constituent_length,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        generator = DatasetGenerator(config)
        output_path = generator.generate()
        print(f"Dataset generated: {output_path}")
        return 0


class ListCommand(CLICommand):
    """List installed datasets."""

    @property
    def name(self) -> str:
        return "list"

    @property
    def help(self) -> str:
        return "List installed datasets"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--datasets-dir", default="data/datasets",
            help="Root directory for datasets (default: data/datasets)",
        )
        parser.add_argument(
            "--detail", action="store_true",
            help="Show detailed per-language statistics",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.data.dataset.manifest import DatasetManifest

        datasets_dir = Path(args.datasets_dir)
        if not datasets_dir.is_dir():
            print("No datasets found.")
            return 0

        found = False
        for subdir in sorted(datasets_dir.iterdir()):
            manifest_path = subdir / "manifest.yaml"
            if not manifest_path.is_file():
                continue

            found = True
            manifest = DatasetManifest.load(manifest_path)
            print(f"\n{manifest.name}")
            print(f"  Path:     {subdir}")
            print(f"  Created:  {manifest.created_at}")
            print(f"  Samples:  {manifest.total_samples:,}")
            print(f"  Rejected: {manifest.rejected_samples:,}")
            print(f"  Languages: {len(manifest.languages)}")

            if args.detail and manifest.languages:
                print("  Per-language:")
                for lang, count in sorted(
                    manifest.languages.items(), key=lambda x: -x[1]
                ):
                    print(f"    {lang}: {count:,}")

        if not found:
            print("No datasets found.")

        return 0


def register_dataset_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``dataset`` subcommand group."""
    dataset_parser = parent_subparsers.add_parser(
        "dataset",
        help="Dataset generation and management",
        description="Generate, inspect, and manage HDF5 datasets for pretraining.",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        title="dataset commands",
        description="Available dataset commands",
        dest="dataset_cmd",
    )

    commands = [
        GenerateCommand(),
        ListCommand(),
    ]

    for cmd in commands:
        sub = dataset_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    # Default: print help if no subcommand given
    dataset_parser.set_defaults(
        command_handler=type(
            "_DatasetHelp", (), {"execute": lambda self, args: dataset_parser.print_help() or 0}
        )()
    )
