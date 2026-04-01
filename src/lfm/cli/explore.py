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


class ExpressionSampleCommand(CLICommand):
    """Sample expressions from a trained expression game checkpoint.

    Loads an expression game checkpoint + embedding store, runs random
    embeddings through the GRU z-sequence generator, decodes through
    the frozen decoder, and prints English → IPA expression pairs.
    """

    @property
    def name(self) -> str:
        return "expression-sample"

    @property
    def help(self) -> str:
        return "Sample decoded IPA expressions from a trained expression game"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint", default="data/expression_game/best.pt",
            help="Expression game checkpoint (default: data/expression_game/best.pt)",
        )
        parser.add_argument(
            "--decoder-path", default="data/vae_decoder.pt",
            help="Pretrained decoder checkpoint (default: data/vae_decoder.pt)",
        )
        parser.add_argument(
            "--spm-path", default="data/spm.model",
            help="Sentencepiece model (default: data/spm.model)",
        )
        parser.add_argument(
            "--embedding-store", default="data/embeddings",
            help="Embedding store directory (default: data/embeddings)",
        )
        parser.add_argument(
            "--num-samples", type=int, default=10,
            help="Number of samples to decode (default: 10)",
        )
        parser.add_argument(
            "--max-segments", type=int, default=16,
            help="Max segments (must match training, default: 16)",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed (default: 42)",
        )
        parser.add_argument(
            "--device", default="cuda",
            help="Compute device (default: cuda)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        import random

        import torch

        from lfm.agents.games.expression import ExpressionGame, ExpressionGameConfig
        from lfm.embeddings.store import EmbeddingStore
        from lfm.faculty import FacultyConfig, LanguageFaculty
        from lfm.generator.config import GeneratorConfig

        import sentencepiece as spm_lib

        device = args.device
        sp = spm_lib.SentencePieceProcessor(model_file=args.spm_path)
        vocab_size = sp.vocab_size()
        eos_id = vocab_size + 1

        # Build expression game
        cfg = ExpressionGameConfig(
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            max_segments=args.max_segments,
            device=device,
        )
        faculty = LanguageFaculty(cfg.build_faculty_config()).to(device)
        game = ExpressionGame(cfg, faculty).to(device)

        # Load checkpoint
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        game.load_checkpoint_state(ckpt)
        game.eval()

        # Load embeddings
        store = EmbeddingStore(args.embedding_store)
        store.load()

        # Load passage text if available
        passages_path = store.store_dir / "passages.jsonl"
        passages: list[str] | None = None
        if passages_path.is_file():
            import json as _json
            with passages_path.open("r", encoding="utf-8") as fh:
                passages = [_json.loads(line).get("text", "") for line in fh]

        # Sample
        rng = random.Random(args.seed)
        indices = rng.sample(range(store.num_passages), min(args.num_samples, store.num_passages))

        with torch.no_grad():
            for i, idx in enumerate(indices):
                emb = torch.from_numpy(store.get_embeddings([idx])).to(device)

                # GRU z-sequence
                z_seq, halt_probs, z_weights, num_segs = game.z_gen(emb)

                # Multi-segment decode
                tokens, gen_mask, seg_bounds = game._multiseg_decode(z_seq, z_weights)

                # Decode to IPA
                token_ids = tokens[0].tolist()
                mask = gen_mask[0].tolist()
                valid_ids = [t for t, m in zip(token_ids, mask) if m and t != eos_id and t < vocab_size]
                ipa = sp.decode(valid_ids)

                n_segs = int(num_segs[0].item())
                if passages and idx < len(passages):
                    print(f'[{i + 1}] ENG: "{passages[idx][:100]}"')
                else:
                    print(f'[{i + 1}] embedding #{idx}')
                print(f'    IPA: {ipa}')
                print(f'    ({len(valid_ids)} tokens, {n_segs} segments)')
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
        ExpressionSampleCommand(),
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
