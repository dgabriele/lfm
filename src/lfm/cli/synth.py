"""Synth subcommand group: lfm synth {build-vocab,train-phase1,train-phase2,generate-corpus}."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


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
        import yaml
        from pathlib import Path
        from lfm.synth.config import SynthConfig
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        vocab = AlienVocab(vocab_size=cfg.vocab_size, seed=cfg.vocab_seed)
        vocab.save(out_dir)
        tokenizer = vocab.build_tokenizer()
        tokenizer.save_pretrained(str(out_dir / "alien_tokenizer"))
        print(f"Alien vocabulary ({len(vocab.syllables)} syllables) saved to {out_dir}")
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
        parser.add_argument("--resume", default=None, help="Phase1 checkpoint to resume from")

    def execute(self, args: argparse.Namespace) -> int:
        import yaml
        from pathlib import Path
        from transformers import PreTrainedTokenizerFast
        from lfm.synth.cipher import WordCipher
        from lfm.synth.config import SynthConfig
        from lfm.synth.model import SynthLM
        from lfm.synth.trainer import CipherTrainer
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        vocab = AlienVocab.load(out_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = SynthLM(cfg, alien_vocab_size=len(tokenizer))
        if args.resume:
            model.load_phase1(args.resume)
            print(f"Resumed from {args.resume}")
        cipher = WordCipher(vocab)
        CipherTrainer(model, cfg, cipher, tokenizer).train()
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
        parser.add_argument("--phase1-checkpoint", required=True, help="Path to phase1_final.pt")
        parser.add_argument("--resume", default=None, help="Phase2 checkpoint to resume from")

    def execute(self, args: argparse.Namespace) -> int:
        import yaml
        from pathlib import Path
        from transformers import PreTrainedTokenizerFast
        from lfm.synth.cipher import WordCipher
        from lfm.synth.config import SynthConfig
        from lfm.synth.model import SynthLM
        from lfm.synth.trainer import ConditioningTrainer
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        vocab = AlienVocab.load(out_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = SynthLM(cfg, alien_vocab_size=len(tokenizer))
        model.load_phase1(args.phase1_checkpoint)
        if args.resume:
            model.load_phase2(args.resume)
        cipher = WordCipher(vocab)
        ConditioningTrainer(model, cfg, cipher, tokenizer).train()
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
        from lfm.synth.model import SynthLM
        from lfm.synth.vocab import AlienVocab

        cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
        out_dir = Path(cfg.output_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
        model = SynthLM(cfg, alien_vocab_size=len(tokenizer))
        model.load_phase1(args.phase1_checkpoint)
        model.load_phase2(args.phase2_checkpoint)
        n = CorpusGenerator(model, tokenizer, cfg).generate_corpus(
            args.store_dir, args.output, batch_size=args.batch_size,
        )
        print(f"Wrote {n} alien sentences to {args.output}")
        return 0


def register_synth_group(parent_subparsers: argparse._SubParsersAction) -> None:
    synth_parser = parent_subparsers.add_parser(
        "synth",
        help="Pretrained-decoder alien language pipeline",
        description="Build alien vocab, train, and generate UNMT corpus.",
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
