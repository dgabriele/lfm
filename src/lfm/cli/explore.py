"""Explore subcommand group for ``lfm explore {dim-sweep}``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lfm.cli.base import CLICommand


class DimSweepCommand(CLICommand):
    """Sweep individual latent dimensions and decode to IPA.

    Samples a random z vector from the VAE's tracked distribution, then
    for selected dimensions sweeps each from -Nσ to +Nσ (holding all
    others constant) and decodes to IPA text.  Output goes to stdout.
    """

    @property
    def name(self) -> str:
        return "dim-sweep"

    @property
    def help(self) -> str:
        return "Sweep individual latent dimensions and decode to IPA (stdout)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint", default="data/vae_resume.pt",
            help="Path to VAE checkpoint (default: data/vae_resume.pt)",
        )
        parser.add_argument(
            "--spm-model", default="data/spm.model",
            help="Path to sentencepiece model (default: data/spm.model)",
        )
        parser.add_argument(
            "--dims", type=int, nargs="+", default=None,
            help="Dimension indices to sweep (default: auto-select top-variance)",
        )
        parser.add_argument(
            "--num-dims", type=int, default=8,
            help="Auto-select this many dims when --dims not given (default: 8)",
        )
        parser.add_argument(
            "--steps", type=int, default=7,
            help="Sweep steps per dimension (default: 7)",
        )
        parser.add_argument(
            "--sigma-range", type=float, default=1.0,
            help="Sweep range in std devs (default: 1.0 = ±1σ)",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for base z sampling (default: 42)",
        )
        parser.add_argument(
            "--temperature", type=float, default=0.8,
            help="Decoding temperature (default: 0.8)",
        )
        parser.add_argument(
            "--top-p", type=float, default=0.9,
            help="Nucleus sampling threshold (default: 0.9)",
        )
        parser.add_argument(
            "--device", default="cuda",
            help="Compute device (default: cuda)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        import torch

        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import decode_z, load_checkpoint

        # Load model
        viz_cfg = VisualizeConfig(
            checkpoint=args.checkpoint,
            spm_model=args.spm_model,
            device=args.device,
        )
        model_data = load_checkpoint(viz_cfg)
        cfg = model_data["cfg"]
        device = model_data["device"]

        # Load SPM
        import sentencepiece as spm_lib

        sp = spm_lib.SentencePieceProcessor(model_file=args.spm_model)
        vocab_size = sp.vocab_size()

        # Get z distribution stats — try model_data first, then the checkpoint
        # file, then sibling vae_decoder.pt (resume checkpoints don't store
        # z_mean/z_std but the decoder checkpoint does).
        z_mean = model_data.get("z_mean")
        z_std = model_data.get("z_std")

        if z_mean is None or z_std is None:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            z_mean = ckpt.get("z_mean")
            z_std = ckpt.get("z_std")

        if z_mean is None or z_std is None:
            from pathlib import Path

            decoder_path = Path(args.checkpoint).parent / "vae_decoder.pt"
            if decoder_path.is_file():
                dec_ckpt = torch.load(decoder_path, map_location="cpu", weights_only=False)
                z_mean = dec_ckpt.get("z_mean")
                z_std = dec_ckpt.get("z_std")
                if z_mean is not None:
                    print(f"# Loaded z stats from {decoder_path}", file=sys.stderr)

        latent_dim = cfg.latent_dim

        if z_mean is not None:
            z_mean = z_mean.float().to(device)
            z_std = z_std.float().to(device)
        else:
            print("WARNING: No z_mean/z_std in checkpoint — using N(0,1)", file=sys.stderr)
            z_mean = torch.zeros(latent_dim, device=device)
            z_std = torch.ones(latent_dim, device=device)

        # Select dimensions
        if args.dims is not None:
            dims = [d for d in args.dims if 0 <= d < latent_dim]
        else:
            std_np = z_std.cpu().numpy()
            top_idx = np.argsort(-std_np)[: args.num_dims]
            dims = sorted(top_idx.tolist())
            print(f"# Auto-selected top-{args.num_dims} variance dims: {dims}", file=sys.stderr)

        # Sample base z
        rng = torch.Generator(device=device).manual_seed(args.seed)
        z_base = torch.randn(latent_dim, generator=rng, device=device) * z_std + z_mean

        # Decode base
        base_tokens = decode_z(
            z_base.unsqueeze(0), model_data, viz_cfg,
            temperature=args.temperature, top_p=args.top_p,
        )[0]
        base_text = sp.decode([t for t in base_tokens if t < vocab_size])

        # Header
        sigma = args.sigma_range
        steps = args.steps
        offsets = np.linspace(-sigma, sigma, steps)

        print("# Latent Dimension Sweep")
        print(f"# seed={args.seed}, range=±{sigma}σ, steps={steps}, "
              f"T={args.temperature}, top_p={args.top_p}")
        print(f"# latent_dim={latent_dim}, dims={dims}")
        print("#")
        print("# Base z (random sample from encoder distribution):")
        print(f"#   {base_text}")
        print()

        # Sweep each dim
        for dim in dims:
            dim_std = z_std[dim].item()
            dim_mean = z_mean[dim].item()
            dim_base = z_base[dim].item()

            print(f"## dim={dim}  (μ={dim_mean:.3f}, σ={dim_std:.3f}, base={dim_base:.3f})")
            print()

            z_batch = z_base.unsqueeze(0).expand(steps, -1).clone()
            for i, offset in enumerate(offsets):
                z_batch[i, dim] = z_mean[dim] + offset * z_std[dim]

            token_lists = decode_z(
                z_batch, model_data, viz_cfg,
                temperature=args.temperature, top_p=args.top_p,
            )

            for offset, tokens in zip(offsets, token_lists):
                text = sp.decode([t for t in tokens if t < vocab_size])
                print(f"  {offset:+.2f}σ  {text}")

            print()

        return 0


def register_explore_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``explore`` subcommand group."""
    explore_parser = parent_subparsers.add_parser(
        "explore",
        help="Latent space exploration tools",
        description="Interactive exploration of the VAE latent space (text output).",
    )
    explore_subparsers = explore_parser.add_subparsers(
        title="explore commands",
        description="Available exploration commands",
        dest="explore_cmd",
    )

    commands = [
        DimSweepCommand(),
    ]

    for cmd in commands:
        sub = explore_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    explore_parser.set_defaults(
        command_handler=type(
            "_ExploreHelp", (), {"execute": lambda self, args: explore_parser.print_help() or 0}
        )()
    )
