"""Translate subcommand group for ``lfm translate {generate-pairs,train,eval}``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class GeneratePairsCommand(CLICommand):
    """Generate (IPA, English) parallel pairs via the trained expression game."""

    @property
    def name(self) -> str:
        return "generate-pairs"

    @property
    def help(self) -> str:
        return "Generate cached (IPA, English) pairs from faculty pipeline"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--embedding-store", default="data/embeddings",
            help="Embedding store directory (default: data/embeddings)",
        )
        parser.add_argument(
            "--decoder-path", default="data/vae_decoder.pt",
            help="Pretrained VAE decoder checkpoint",
        )
        parser.add_argument(
            "--spm-path", default="data/spm.model",
            help="Sentencepiece model",
        )
        parser.add_argument(
            "--expression-checkpoint", default="data/expression_game/best.pt",
            help="Trained expression game checkpoint",
        )
        parser.add_argument(
            "--max-segments", type=int, default=16,
            help="Max segments (must match training, default: 16)",
        )
        parser.add_argument(
            "--batch-size", type=int, default=64,
            help="Generation batch size (default: 64)",
        )
        parser.add_argument(
            "--output", default="data/translator/pairs.jsonl",
            help="Output JSONL path",
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=42)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.translator.config import PairGenerationConfig
        from lfm.translator.pairs import PairGenerator

        config = PairGenerationConfig(
            embedding_store_dir=args.embedding_store,
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            expression_checkpoint=args.expression_checkpoint,
            max_segments=args.max_segments,
            batch_size=args.batch_size,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
        )

        generator = PairGenerator(config)
        pairs = generator.generate()
        print(f"Generated {len(pairs)} pairs -> {args.output}")
        return 0


class TrainCommand(CLICommand):
    """Fine-tune a causal LM on IPA -> English translation."""

    @property
    def name(self) -> str:
        return "train"

    @property
    def help(self) -> str:
        return "Train IPA -> English translator"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            default="Qwen/Qwen2.5-0.5B",
            help="HuggingFace model to fine-tune (default: Qwen/Qwen2.5-0.5B)",
        )
        parser.add_argument(
            "--pairs",
            default="data/models/v1/translator/pairs.jsonl",
            help="Path to JSONL pairs file",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of training epochs (default: 3)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help="Learning rate (default: 2e-5)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Training batch size (default: 8)",
        )
        parser.add_argument(
            "--max-len",
            type=int,
            default=256,
            help="Max sequence length (default: 256)",
        )
        parser.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=4,
            help="Gradient accumulation steps (default: 4)",
        )
        parser.add_argument(
            "--use-lora",
            action="store_true",
            help="Enable LoRA parameter-efficient fine-tuning",
        )
        parser.add_argument(
            "--output-dir",
            default="data/models/v1/translator",
            help="Output directory (default: data/models/v1/translator)",
        )
        parser.add_argument(
            "--device",
            default="cuda",
            help="Compute device (default: cuda)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.translator.config import TranslatorConfig
        from lfm.translator.trainer import TranslatorTrainer

        p = self.validate_file_exists(args.pairs, "Pairs file")
        if p is None:
            return 1

        config = TranslatorConfig(
            model_name=args.model_name,
            use_lora=args.use_lora,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            pairs_path=args.pairs,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
        )

        trainer = TranslatorTrainer(config)
        results = trainer.train()
        print(f"Training complete. Val loss: {results.get('final_val_loss', 'N/A')}")
        return 0


class EvalCommand(CLICommand):
    """Evaluate a trained IPA -> English translator."""

    @property
    def name(self) -> str:
        return "eval"

    @property
    def help(self) -> str:
        return "Evaluate trained translator (BLEU + semantic similarity)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-dir",
            default="data/models/v1/translator",
            help="Directory containing trained model",
        )
        parser.add_argument(
            "--pairs",
            default=None,
            help="Override pairs path (default: read from model config)",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=200,
            help="Max samples to evaluate (default: 200)",
        )
        parser.add_argument(
            "--max-new-tokens",
            type=int,
            default=64,
            help="Max new tokens during generation (default: 64)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Generation temperature (default: 0.7)",
        )
        parser.add_argument(
            "--device",
            default="cuda",
            help="Compute device (default: cuda)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.translator.evaluator import TranslatorEvaluator

        model_dir = self.validate_file_exists(args.model_dir, "Model directory")
        if model_dir is None:
            return 1

        evaluator = TranslatorEvaluator(
            model_dir=str(model_dir),
            device=args.device,
        )
        results = evaluator.evaluate(
            pairs_path=args.pairs,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"\nResults saved to {model_dir / 'results.json'}")
        for k, v in sorted(results.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        return 0


class EvalPhonologyCommand(CLICommand):
    """Run phonology benchmark on the trained translator."""

    @property
    def name(self) -> str:
        return "eval-phonology"

    @property
    def help(self) -> str:
        return "Evaluate translator's phonological competence in the emergent language"

    @property
    def description(self) -> str:
        return (
            "Run PhonologyBench-inspired tasks: syllable counting, rhyme "
            "detection, and minimal pair discrimination on the emergent "
            "language. Tests whether the translator LLM has acquired "
            "genuine phonological understanding."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-dir",
            default="data/models/v1/translator",
            help="Translator model directory",
        )
        parser.add_argument(
            "--decoder-path",
            default="data/models/v1/vae_decoder.pt",
            help="Path to frozen decoder checkpoint",
        )
        parser.add_argument(
            "--spm-path",
            default="data/models/v1/spm.model",
            help="Path to sentencepiece model",
        )
        parser.add_argument(
            "--num-samples", type=int, default=200,
            help="Samples per task (default: 200)",
        )
        parser.add_argument(
            "--device", default="cuda",
            help="Device for inference",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Save results JSON to this path (default: model-dir/phonology_bench.json)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.translator.phonology_bench import PhonologyBench

        bench = PhonologyBench(
            translator_model_dir=args.model_dir,
            faculty_decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            num_samples=args.num_samples,
            device=args.device,
        )
        results = bench.run_all()
        bench.print_report(results)

        output = args.output or str(Path(args.model_dir) / "phonology_bench.json")
        bench.save_results(results, output)
        return 0


class GenerateCorpusCommand(CLICommand):
    """Generate romanized IPA corpus for self-supervised LLM pretraining."""

    @property
    def name(self) -> str:
        return "generate-corpus"

    @property
    def help(self) -> str:
        return "Generate romanized IPA corpus for self-supervised pretraining"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--expression-checkpoint", default="data/expression_game/best.pt")
        parser.add_argument("--decoder-path", default="data/vae_decoder.pt")
        parser.add_argument("--spm-path", default="data/spm.model")
        parser.add_argument("--embedding-store", default="data/embeddings")
        parser.add_argument("--passes", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--output", default="data/translator/corpus.txt")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=42)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.translator.config import CorpusConfig
        from lfm.translator.corpus import CorpusGenerator

        config = CorpusConfig(
            expression_checkpoint=args.expression_checkpoint,
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            embedding_store_dir=args.embedding_store,
            num_passes=args.passes,
            batch_size=args.batch_size,
            output_path=args.output,
            device=args.device,
            seed=args.seed,
        )

        gen = CorpusGenerator(config)
        stats = gen.generate()
        print(f"Generated {stats['num_lines']} lines, {stats['num_tokens']} tokens, "
              f"{stats['unique_lines']} unique → {args.output}")
        return 0


def register_translate_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``translate`` subcommand group with sub-subparsers."""
    translate_parser = parent_subparsers.add_parser(
        "translate",
        help="IPA -> English translation tools",
        description="Generate pairs, train, and evaluate IPA -> English translators.",
    )
    translate_subparsers = translate_parser.add_subparsers(
        title="translation commands",
        description="Available translation commands",
        dest="translate_cmd",
    )

    commands = [
        GeneratePairsCommand(),
        GenerateCorpusCommand(),
        TrainCommand(),
        EvalCommand(),
        EvalPhonologyCommand(),
    ]

    for cmd in commands:
        sub = translate_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    # Default: print help if no subcommand given
    translate_parser.set_defaults(
        command_handler=type(
            "_TranslateHelp",
            (),
            {"execute": staticmethod(lambda _args: translate_parser.print_help() or 0)},
        )()
    )
