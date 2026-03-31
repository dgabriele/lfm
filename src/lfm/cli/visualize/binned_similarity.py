"""CLI command for binned similarity visualization."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class BinnedSimilarityCommand(CLICommand):
    """Binned similarity heatmap for agent game linguistic encodings."""

    @property
    def name(self) -> str:
        return "binned-similarity"

    @property
    def help(self) -> str:
        return "Dendrogram + heatmap of binned IPA token similarity from agent checkpoint"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint", required=True,
            help="Path to trained agent game checkpoint",
        )
        parser.add_argument(
            "--embedding-store", default="data/embeddings",
            help="Path to embedding store directory",
        )
        parser.add_argument(
            "--decoder-path", default="data/vae_decoder.pt",
        )
        parser.add_argument("--spm-path", default="data/spm.model")
        parser.add_argument("--num-memory-tokens", type=int, default=8)
        parser.add_argument("--output-dir", default="output/viz")
        parser.add_argument("--n-bins", type=int, default=10000)
        parser.add_argument(
            "--metrics", nargs="+", default=["jaccard", "cosine", "edit"],
            choices=["jaccard", "cosine", "edit"],
            help="Similarity metrics to compute (default: all)",
        )
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--device", default="cuda")

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.binned_similarity import generate_binned_similarity

        paths = generate_binned_similarity(
            checkpoint_path=args.checkpoint,
            embedding_store_dir=args.embedding_store,
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            num_memory_tokens=args.num_memory_tokens,
            output_dir=args.output_dir,
            n_bins=args.n_bins,
            metrics=args.metrics,
            batch_size=args.batch_size,
            device=args.device,
        )

        for p in paths:
            print(f"  Saved: {p}")
        return 0
