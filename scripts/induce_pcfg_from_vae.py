"""Induce a PCFG from a phrase VAE by sampling its latent manifold.

Generates a synthetic corpus by drawing z vectors via scrambled
Sobol → Gaussian CDF (low-discrepancy coverage of N(z_mean, diag(z_std²))
where z_mean/z_std are the stored training-time moments), decoding each
through the frozen VAE, and optionally rendering through p2g for an
English-surface variant.  The resulting corpus is then fed to the same
distributional-clustering + rule-counting pipeline as
``scripts/induce_pcfg.py``.

Why scrambled Sobol over i.i.d. Gaussian: 256-dim latent space is vast,
and naive i.i.d. Gaussian sampling leaves big gaps in the manifold for
any reasonable N.  Scrambled Sobol gives dramatically better coverage of
the relevant hypervolume for the same budget while still producing
marginally Gaussian draws after the inverse-CDF transform.

Usage:

    poetry run python scripts/induce_pcfg_from_vae.py \\
        --decoder-path data/models/v7_english/vae_decoder.pt \\
        --spm-path    data/datasets/english-constituents-v7/spm.model \\
        --num-samples 10000 \\
        --rendering   ipa \\
        --num-categories 24

    # or, for English surface via p2g (recommended for PCFG legibility):
    poetry run python scripts/induce_pcfg_from_vae.py \\
        --decoder-path data/models/v7_english/vae_decoder.pt \\
        --spm-path data/datasets/english-constituents-v7/spm.model \\
        --num-samples 10000 \\
        --rendering p2g_english \\
        --p2g-checkpoint data/models/p2g_v11/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from induce_pcfg import build_word_vectors, cluster_words, induce_rules  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def sobol_gaussian(
    n: int, d: int, z_mean: np.ndarray, z_std: np.ndarray, seed: int = 42,
) -> np.ndarray:
    """Scrambled Sobol → Gaussian CDF inverse, scaled to N(z_mean, z_std²).

    Returns (n, d) array of samples that are marginally Gaussian per dim
    but with low-discrepancy *set* coverage (much better latent-space
    exploration than i.i.d. Gaussian at equal N).
    """
    from scipy.stats import norm, qmc
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = engine.random(n)
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    z = norm.ppf(u)
    return z * z_std[None, :] + z_mean[None, :]


def load_vae(decoder_path: str, spm_path: str, device: torch.device):
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.generator.config import GeneratorConfig

    # Peek at stored config scalars so the in-memory model matches checkpoint.
    ckpt = torch.load(decoder_path, map_location="cpu", weights_only=False)
    ckpt_vocab = int(ckpt.get("vocab_size", 8000))

    # Generator backend: multilingual_vae for SPM-based (v7), phoneme_vae
    # for JSON-alphabet (v8/v9/v9.5).  Auto-detect from spm_path extension.
    name = "phoneme_vae" if spm_path.endswith(".json") else "multilingual_vae"

    gcfg_kwargs = dict(
        name=name,
        pretrained_decoder_path=decoder_path,
        spm_model_path=spm_path,
        freeze_decoder=True,
        vocab_size=ckpt_vocab,
    )
    # Forward architectural config from the checkpoint where the field
    # actually exists on GeneratorConfig (encoder-side params don't).
    from lfm.generator.config import GeneratorConfig as _GC
    _valid = set(_GC.model_fields.keys())
    for k in (
        "latent_dim", "num_memory_tokens", "decoder_hidden_dim",
        "decoder_num_layers", "decoder_num_heads",
        "attention_head_windows", "attention_global_every",
        "use_rope", "share_decoder_layers",
    ):
        if k in ckpt and k in _valid:
            gcfg_kwargs[k] = ckpt[k]
    cfg = FacultyConfig(
        dim=384,  # dummy; _input_proj isn't used for sampling
        generator=GeneratorConfig(**gcfg_kwargs),
    )
    faculty = LanguageFaculty(cfg).to(device)
    faculty.generator.eval()
    return faculty.generator


def sample_corpus(
    gen, num_samples: int, device: torch.device, batch_size: int = 32,
    seed: int = 42,
) -> list[str]:
    """Sample num_samples phrases from the VAE via Sobol-Gaussian z."""
    z_mean = gen._z_mean.detach().cpu().numpy().astype(np.float64)  # noqa: SLF001
    z_std = gen._z_std.detach().cpu().numpy().astype(np.float64)    # noqa: SLF001
    d = z_mean.shape[0]
    logger.info(
        "z stats: dim=%d mean_norm=%.3f std_mean=%.4f",
        d, np.linalg.norm(z_mean), z_std.mean(),
    )

    z_all = sobol_gaussian(num_samples, d, z_mean, z_std, seed=seed)
    logger.info(
        "Sobol-Gaussian samples: shape=%s mean=%.3f std=%.4f",
        z_all.shape, z_all.mean(), z_all.std(),
    )

    rendered: list[str] = []
    with torch.no_grad():
        for lo in range(0, num_samples, batch_size):
            z = torch.tensor(
                z_all[lo: lo + batch_size], dtype=torch.float32, device=device,
            )
            tokens, _, _, _, mask = gen._decode(z)  # noqa: SLF001
            texts = gen.render_surface(tokens, mask=mask, eos_id=gen.eos_id)
            rendered.extend(texts)
            if (lo // batch_size + 1) % 20 == 0:
                logger.info("  decoded %d/%d", lo + batch_size, num_samples)
    return rendered


def maybe_apply_p2g(
    ipa_phrases: list[str], p2g_ckpt: str, device: torch.device,
    batch_size: int = 512,
) -> list[str]:
    """Render each IPA phrase through the word-level p2g seq2seq.

    Batches all words from all phrases through one model.generate() call
    at a time, then stitches back into per-phrase strings.  ~100x faster
    than per-word calls for 50K-phrase inputs.
    """
    from test_qwen_p2g_spelling import load_p2g
    import test_qwen_p2g_spelling as tm
    from lfm.p2g.vocab import PAD_ID
    tm.CKPT_PATH = p2g_ckpt
    logger.info("Loading p2g: %s", p2g_ckpt)
    p2g, ipa_vocab, sp_vocab, cfg = load_p2g(device)

    # Flatten: all (phrase_idx, word_ipa) pairs
    all_words: list[str] = []
    phrase_lengths: list[int] = []
    for phrase in ipa_phrases:
        ws = phrase.split()
        phrase_lengths.append(len(ws))
        all_words.extend(ws)
    logger.info("p2g: %d phrases, %d total words", len(ipa_phrases), len(all_words))

    # Batched decode
    max_ipa = cfg["max_ipa_len"]
    rendered_words: list[str] = []
    for lo in range(0, len(all_words), batch_size):
        batch = all_words[lo: lo + batch_size]
        ids = torch.full((len(batch), max_ipa), PAD_ID, dtype=torch.long, device=device)
        for i, w in enumerate(batch):
            enc = ipa_vocab.encode(w)[:max_ipa]
            if enc:
                ids[i, : len(enc)] = torch.tensor(enc, device=device)
        with torch.no_grad():
            preds = p2g.generate(ids)
        rendered_words.extend(sp_vocab.decode(p) for p in preds)
        if (lo // batch_size + 1) % 10 == 0:
            logger.info("  p2g %d/%d words", lo + batch_size, len(all_words))

    # Reassemble into phrase strings
    out: list[str] = []
    offset = 0
    for L in phrase_lengths:
        out.append(" ".join(rendered_words[offset: offset + L]))
        offset += L
    return out


def tokenize_corpus(phrases: list[str], min_words: int = 3) -> list[list[str]]:
    sentences: list[list[str]] = []
    for p in phrases:
        words = p.strip().split()
        if len(words) >= min_words:
            sentences.append(words)
    return sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-path", required=True,
                    help="VAE decoder .pt (v7 multilingual, v7-english, v8 phoneme, etc.)")
    ap.add_argument("--spm-path", required=True,
                    help="SPM .model (v7/v7-english) OR phoneme alphabet .json (v8/v9/v9.5)")
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rendering", choices=["ipa", "p2g_english"], default="ipa")
    ap.add_argument("--p2g-checkpoint", default="data/models/p2g_v11/latest.pt")
    ap.add_argument("--num-categories", type=int, default=24)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--save-corpus",
                    help="Optional path to dump the sampled corpus (one phrase per line)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load VAE
    logger.info("Loading VAE: %s", args.decoder_path)
    gen = load_vae(args.decoder_path, args.spm_path, device)

    # 2. Sample synthetic corpus
    logger.info("Sampling %d phrases from VAE", args.num_samples)
    phrases = sample_corpus(
        gen, args.num_samples, device, args.batch_size, args.seed,
    )
    del gen
    torch.cuda.empty_cache()

    # 3. Optionally render through p2g for English surface
    if args.rendering == "p2g_english":
        phrases = maybe_apply_p2g(phrases, args.p2g_checkpoint, device)

    # 4. Optionally dump corpus for later inspection
    if args.save_corpus:
        Path(args.save_corpus).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_corpus, "w") as f:
            f.write("\n".join(phrases))
        logger.info("Wrote corpus: %s", args.save_corpus)

    # 5. PCFG induction pipeline (reuses induce_pcfg.py functions)
    sentences = tokenize_corpus(phrases)
    logger.info("Tokenized corpus: %d sentences (min 3 words)", len(sentences))

    vocab, vectors = build_word_vectors(sentences, min_freq=args.min_freq)
    if len(vocab) < args.num_categories:
        logger.warning(
            "Vocabulary (%d) smaller than requested categories (%d); shrinking",
            len(vocab), args.num_categories,
        )
        args.num_categories = max(2, len(vocab) // 2)

    labels, _clusters = cluster_words(vocab, vectors, args.num_categories)
    word2cat = {w: int(l) for w, l in zip(vocab, labels)}

    induce_rules(sentences, word2cat, args.num_categories)


if __name__ == "__main__":
    main()
