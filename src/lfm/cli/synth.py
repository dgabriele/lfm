"""Synth subcommand group: lfm synth {build-vocab,train-phase1,train-phase2,generate-corpus}."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


def _setup_file_logging(log_path: "Path") -> None:
    """Route all logging exclusively to log_path, removing any StreamHandlers."""
    import logging
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    root.addHandler(handler)


def _build_model(cfg, alien_vocab_size: int):
    """Construct CausalDecoderBackend + SynthLM from config."""
    import logging
    from lfm.synth.backend import CausalDecoderBackend
    from lfm.synth.model import SynthLM

    logger = logging.getLogger(__name__)
    logger.info("loading backend model: %s", cfg.base_model_name)
    backend = CausalDecoderBackend(
        cfg.base_model_name,
        alien_vocab_size=alien_vocab_size,
        with_reference_body=(cfg.phase1_hidden_mse_weight > 0),
    )
    return SynthLM(backend, cfg)


class BuildVocabCommand(CLICommand):
    @property
    def name(self) -> str:
        return "build-vocab"

    @property
    def help(self) -> str:
        return "Build and save the alien syllable vocabulary + tokenizer"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")

    def execute(self, args: argparse.Namespace) -> int:
        import json
        import yaml
        from pathlib import Path
        from lfm.synth.cipher import WordCipher
        from lfm.synth.config import SynthConfig
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        vocab = AlienVocab(vocab_size=cfg.vocab_size, seed=cfg.vocab_seed)
        vocab.save(out_dir)

        # Encode corpus with the cipher to produce BPE training data.
        print(f"Encoding {cfg.phase1_dataset_dir} for BPE training...")
        cipher = WordCipher(vocab)
        dataset_path = Path(cfg.phase1_dataset_dir)
        if dataset_path.suffix == ".jsonl":
            texts = [json.loads(l)["text"] for l in dataset_path.read_text().splitlines() if l.strip()]
        else:
            texts = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
        encoded = cipher.encode_batch(texts)

        print(f"Training BPE tokenizer (vocab_size={cfg.vocab_size}) on {len(encoded)} sentences...")
        tokenizer = vocab.build_tokenizer(encoded, vocab_size=cfg.vocab_size)
        tokenizer.save_pretrained(str(out_dir / "alien_tokenizer"))
        print(f"Alien vocab: {len(vocab.syllables)} syllables, BPE vocab={cfg.vocab_size} → {out_dir}")
        return 0


class TrainPhase1Command(CLICommand):
    @property
    def name(self) -> str:
        return "train-phase1"

    @property
    def help(self) -> str:
        return "Phase 1: cipher fine-tuning — English text -> alien tokens"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")
        parser.add_argument("--resume", default=None, help="Phase 1 trainer checkpoint to resume from")

    def execute(self, args: argparse.Namespace) -> int:
        import yaml
        from pathlib import Path
        from transformers import PreTrainedTokenizerFast
        from lfm.synth.cipher import WordCipher
        from lfm.synth.config import SynthConfig
        from lfm.synth.trainer import AlienLMTrainer
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        _setup_file_logging(out_dir / "train_phase1.log")
        vocab = AlienVocab.load(out_dir)
        alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = _build_model(cfg, alien_vocab_size=len(alien_tok))
        trainer = AlienLMTrainer(model, cfg, WordCipher(vocab), alien_tok)
        start_step = 0
        if args.resume:
            start_step = trainer.load_checkpoint(args.resume)
            print(f"Resumed from {args.resume} at step {start_step}")
        trainer.train(start_step=start_step)
        return 0


class TrainPhase2Command(CLICommand):
    @property
    def name(self) -> str:
        return "train-phase2"

    @property
    def help(self) -> str:
        return "Phase 2: embedding conditioning — source embedding -> alien tokens"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")
        parser.add_argument("--phase1-checkpoint", required=True,
                            help="Path to phase1_final.pt (alien emb + head weights)")
        parser.add_argument("--resume", default=None, help="Phase 2 trainer checkpoint to resume from")

    def execute(self, args: argparse.Namespace) -> int:
        import yaml
        from pathlib import Path
        from transformers import PreTrainedTokenizerFast
        from lfm.synth.cipher import WordCipher
        from lfm.synth.config import SynthConfig
        from lfm.synth.trainer import ConditioningTrainer
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        _setup_file_logging(out_dir / "train_phase2.log")
        vocab = AlienVocab.load(out_dir)
        alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = _build_model(cfg, alien_vocab_size=len(alien_tok))
        model.load_phase1(args.phase1_checkpoint)
        trainer = ConditioningTrainer(model, cfg, WordCipher(vocab), alien_tok)
        start_step = 0
        if args.resume:
            start_step = trainer.load_checkpoint(args.resume)
            print(f"Resumed from {args.resume} at step {start_step}")
        trainer.train(start_step=start_step)
        return 0


class GenerateCorpusCommand(CLICommand):
    @property
    def name(self) -> str:
        return "generate-corpus"

    @property
    def help(self) -> str:
        return "Generate one alien sentence per embedding (corpus generation)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")
        parser.add_argument("--phase1-checkpoint", required=True)
        parser.add_argument("--phase2-checkpoint", required=True)
        parser.add_argument("--store-dir", required=True, help="Embedding store directory")
        parser.add_argument("--output", required=True, help="Output corpus file")
        parser.add_argument("--batch-size", type=int, default=64)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml
        from pathlib import Path
        from transformers import PreTrainedTokenizerFast
        from lfm.synth.config import SynthConfig
        from lfm.synth.generator import CorpusGenerator
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = _build_model(cfg, alien_vocab_size=len(alien_tok))
        model.load_phase1(args.phase1_checkpoint)
        model.load_phase2(args.phase2_checkpoint)
        n = CorpusGenerator(model, alien_tok, cfg).generate_corpus(
            args.store_dir, args.output, batch_size=args.batch_size,
        )
        print(f"Wrote {n} alien sentences to {args.output}")
        return 0


def register_synth_group(parent_subparsers: argparse._SubParsersAction) -> None:
    synth_parser = parent_subparsers.add_parser(
        "synth",
        help="Decoder-only alien language pipeline",
        description="Build alien vocab, train (phase1+phase2), and generate UNMT corpus.",
    )
    synth_subparsers = synth_parser.add_subparsers(
        title="synth commands", dest="synth_cmd",
    )
    commands = [
        BuildVocabCommand(),
        TrainPhase1Command(),
        TrainPhase2Command(),
        GenerateCorpusCommand(),
    ]
    for cmd in commands:
        sub = synth_subparsers.add_parser(cmd.name, help=cmd.help, description=cmd.description)
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)
    synth_parser.set_defaults(
        command_handler=type(
            "_SynthHelp", (),
            {"execute": staticmethod(lambda _: synth_parser.print_help() or 0)},
        )()
    )
