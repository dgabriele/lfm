#!/usr/bin/env python
"""Decoder-only interpretation-topology eval.

Tests the pair ``(VAE decoder, Qwen interpretation)`` in isolation — no
dialogue game, no z-generator.  For a batch of held-out source sentences:

  1. SBERT-embed → source topology (ground-truth semantic space).
  2. VAE-encode → z, then autoregressive decode → the decoder's own
     lexicalization of that z.
  3. Qwen-encode the decoder surface (mean-pool last hidden layer) →
     interpretation topology.
  4. Spearman ρ between the pairwise cosine-distance matrices of the
     two spaces.  A high ρ means similar source embeddings produce
     decoder outputs that Qwen interprets similarly — the topology-
     preservation property that makes downstream dialogue + interpretation
     work.

This is not a reconstruction eval.  We do not compare decoder surface to
source string; we compare Qwen's interpretation of the surface against
the source's SBERT embedding, pairwise.

Usage::

    poetry run python scripts/decoder_topology_eval.py \\
        --resume-checkpoint data/models/v12_english_ortho/vae_resume.pt \\
        --spm-path data/datasets/english-constituents-v12/spm.model \\
        --sentences-h5 data/datasets/english-constituents-v12/samples.h5 \\
        --num-samples 300 \\
        --output data/models/v12_english_ortho/topology_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise cosine distance, flattened."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    return (1.0 - sim[iu]).astype(np.float32)


def knn_preservation(D_a: np.ndarray, D_b: np.ndarray, k: int = 10) -> float:
    """Fraction of each sample's top-k neighbors in A that are also
    top-k neighbors in B.  Averaged over samples.  Both ``D_*`` are
    full (N, N) distance matrices; diagonal ignored.
    """
    n = D_a.shape[0]
    neighbors_a = np.argpartition(D_a + np.eye(n) * 1e9, k, axis=1)[:, :k]
    neighbors_b = np.argpartition(D_b + np.eye(n) * 1e9, k, axis=1)[:, :k]
    overlap = [
        len(set(neighbors_a[i]) & set(neighbors_b[i])) / k
        for i in range(n)
    ]
    return float(np.mean(overlap))


def full_cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (Xn @ Xn.T)


def load_sentences(path: Path, n: int, seed: int) -> list[str]:
    """Sample ``n`` sentences from an HDF5 dataset (`samples/ipa` field)
    or a newline-delimited text file.
    """
    rng = random.Random(seed)
    if path.suffix == ".h5":
        import h5py
        with h5py.File(path, "r") as f:
            g = f["samples"]
            total = g["ipa"].shape[0]
            idxs = rng.sample(range(total), min(n, total))
            idxs.sort()  # h5py prefers sorted indexing
            items = [g["ipa"][i].decode() for i in idxs]
    else:
        items = path.read_text().splitlines()
        items = rng.sample(items, min(n, len(items)))
    # Drop very short or very long — keeps eval focused on normal content
    items = [s for s in items if 20 <= len(s) <= 250]
    return items[:n]


def load_vae(
    resume_path: Path, spm_path: Path, device: torch.device,
) -> tuple[dict, object, int, int, int, int, int]:
    """Load the full VAE (encoder + decoder) state from a resume ckpt.

    Returns ``(modules, sp, vocab_size, bos_id, eos_id, max_seq_len,
    num_memory_tokens)``.
    """
    import sentencepiece as spm_lib
    from lfm.generator.pretrain.config import VAEPretrainConfig
    from lfm.generator.pretrain.model import build_model

    logger.info("Loading resume checkpoint: %s", resume_path)
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    # Build a minimal config from the checkpoint's architecture metadata.
    sp = spm_lib.SentencePieceProcessor(model_file=str(spm_path))
    vocab_size = sp.vocab_size()
    full_vocab = vocab_size + 2
    bos_id = vocab_size
    eos_id = vocab_size + 1

    cfg_fields = {
        "latent_dim": ckpt.get("latent_dim", 256),
        "decoder_hidden_dim": ckpt.get("decoder_hidden_dim", 512),
        "decoder_num_layers": 4,
        "decoder_num_heads": 8,
        "decoder_dropout": 0.0,
        "max_seq_len": ckpt.get("max_seq_len", 286),
        "num_memory_tokens": ckpt.get("num_memory_tokens", 8),
        "encoder_num_layers": ckpt.get("encoder_num_layers", 2),
        "attention_head_windows": ckpt.get(
            "attention_head_windows", [3, 3, 7, 7, 15, 15, 0, 0],
        ),
        "attention_global_every": ckpt.get("attention_global_every", 7),
        "use_rope": ckpt.get("use_rope", True),
        "share_decoder_layers": ckpt.get("share_decoder_layers", True),
        "encoder_pooling": ckpt.get("encoder_pooling", "mean"),
        "dataset_path": "",  # not used
        "spm_model_path": str(spm_path),
        "output_path": "",
    }
    # Keep only fields the config actually declares
    valid = set(VAEPretrainConfig.model_fields.keys())
    cfg = VAEPretrainConfig(**{k: v for k, v in cfg_fields.items() if k in valid})

    modules = build_model(cfg, cfg.decoder_hidden_dim, full_vocab, device)
    for k, m in modules.items():
        if isinstance(m, torch.nn.Module) and k in ckpt["modules"]:
            m.load_state_dict(ckpt["modules"][k])
        if isinstance(m, torch.nn.Module):
            m.eval()

    logger.info(
        "Loaded VAE: latent=%d, max_seq_len=%d, num_memory_tokens=%d",
        cfg.latent_dim, cfg.max_seq_len, cfg.num_memory_tokens,
    )
    return (
        modules, sp, vocab_size, bos_id, eos_id,
        cfg.max_seq_len, cfg.num_memory_tokens,
    )


@torch.no_grad()
def encode_to_z(
    sentences: list[str], sp, modules: dict, bos_id: int, eos_id: int,
    device: torch.device, max_seq_len: int, latent_dim: int,
) -> torch.Tensor:
    """Batch-encode → proper mask + mean-pooling (mask-aware) → mu.

    Mirrors the training ``_vae_forward`` encoder path exactly: passes
    ``src_key_padding_mask`` so padded positions don't contribute to the
    pooled representation.  Uses the deterministic ``mu`` from the
    Gaussian head instead of sampling.
    """
    enc_tok = modules["enc_token_embedding"]
    enc_pos = modules["enc_pos_embedding"]
    encoder = modules["encoder"]
    enc_to_latent = modules["enc_to_latent"]

    # Tokenize + pad to max length in this batch (not max_seq_len — saves mem)
    token_lists: list[list[int]] = []
    for text in sentences:
        ids = sp.encode(text, out_type=int)
        ids = [x for x in ids if x not in {0, 1, 2, 3}]
        ids = ids[: max_seq_len - 1] + [eos_id]
        token_lists.append(ids)
    lengths = torch.tensor([len(t) for t in token_lists], device=device)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(sentences), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(token_lists):
        batch[i, :len(ids)] = torch.tensor(ids, device=device)

    src_mask = (
        torch.arange(max_len, device=device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )  # True = valid position
    pos_ids = torch.arange(max_len, device=device).unsqueeze(0)
    h = enc_tok(batch)
    if not isinstance(enc_pos, torch.nn.Identity):
        h = h + enc_pos(pos_ids)
    enc_out = encoder(h, src_key_padding_mask=~src_mask)

    # Mask-aware mean pool (mirrors training: sum over valid, divide by length)
    enc_masked = enc_out * src_mask.unsqueeze(-1).float()
    pooled = enc_masked.sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)

    h_lat = enc_to_latent(pooled)
    if h_lat.size(-1) == 2 * latent_dim:
        mu = h_lat.chunk(2, dim=-1)[0]
    else:
        mu = h_lat
    return mu


@torch.no_grad()
def decode_z(
    z: torch.Tensor, modules: dict, bos_id: int, eos_id: int,
    vocab_size: int, max_seq_len: int, sp, num_memory_tokens: int,
    device: torch.device, latent_dim: int,
) -> list[str]:
    """Top-p nucleus sampling decode — same path the training
    diagnostics use (greedy argmax produces repetitive garbage on this
    architecture; nucleus sampling is the tested inference mode).
    """
    from lfm.generator.pretrain.config import VAEPretrainConfig
    from lfm.generator.pretrain.diagnostics import sample_decode

    cfg = VAEPretrainConfig(
        latent_dim=latent_dim,
        decoder_hidden_dim=512,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_dropout=0.0,
        max_seq_len=max_seq_len,
        num_memory_tokens=num_memory_tokens,
        attention_head_windows=[3, 3, 7, 7, 15, 15, 0, 0],
        attention_global_every=7,
        use_rope=True,
        share_decoder_layers=True,
        dataset_path="",
        spm_model_path="",
        output_path="",
    )
    return sample_decode(
        z, modules=modules, cfg=cfg, vocab_size=vocab_size,
        bos_id=bos_id, eos_id=eos_id, device=device, sp=sp,
        top_p=0.9, temperature=0.8,
    )


@torch.no_grad()
def qwen_interpret(
    texts: list[str], model_name: str, device: torch.device,
    max_new_tokens: int = 48,
    prompt_template: str = (
        "Read the following sentence and, in one short sentence, "
        "describe what it is about.\n\n"
        "Sentence: {text}\n\nSummary:"
    ),
) -> list[str]:
    """Have Qwen generate an English interpretation for each input.

    Unlike :func:`qwen_embed`, this bypasses the residual-stream
    geometry entirely.  We sample an English summary and then let SBERT
    embed *that* — which is what the downstream LFM pipeline actually
    does (LLM is the interpretation layer, not the embedding layer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading Qwen (interpret): %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    interpretations: list[str] = []
    for i, text in enumerate(texts):
        prompt = prompt_template.format(text=text)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        gen = out_ids[0, inputs["input_ids"].size(1):]
        summary = tok.decode(gen, skip_special_tokens=True).strip()
        # First line only — avoid runaway continuations
        summary = summary.split("\n")[0].strip()
        interpretations.append(summary or text)
        if (i + 1) % 50 == 0:
            logger.info("  interpreted %d/%d", i + 1, len(texts))
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return interpretations


@torch.no_grad()
def qwen_embed(
    texts: list[str], model_name: str, device: torch.device,
    pooling: str = "last_token",
) -> np.ndarray:
    """Extract a single vector per text from Qwen's final hidden layer.

    ``pooling``:
      - ``"mean"``: average over all valid positions (weak for CLMs —
        the per-token residuals are specialized for next-token prediction,
        not for semantic summarization).
      - ``"last_token"``: use the final valid position's hidden state,
        which has attended to the entire input and is the closest thing
        a CLM has to a whole-sequence summary.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading Qwen: %s (pooling=%s)", model_name, pooling)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    embs: list[np.ndarray] = []
    for i, text in enumerate(texts):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0]
        mask = inputs["attention_mask"][0].bool()
        if pooling == "last_token":
            last_idx = int(mask.nonzero(as_tuple=True)[0][-1].item())
            pooled = h[last_idx].float().cpu().numpy()
        else:
            pooled = h[mask].mean(dim=0).float().cpu().numpy()
        embs.append(pooled)
        if (i + 1) % 50 == 0:
            logger.info("  Qwen-encoded %d/%d", i + 1, len(texts))
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(embs, axis=0)


def sbert_embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    logger.info("Loading SBERT: %s", model_name)
    m = SentenceTransformer(model_name)
    return m.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-checkpoint", type=Path, required=True)
    ap.add_argument("--spm-path", type=Path, required=True)
    ap.add_argument("--sentences-h5", type=Path, default=None,
                    help="HDF5 with samples/ipa field")
    ap.add_argument("--sentences-txt", type=Path, default=None,
                    help="Newline-delimited sentences (alternative to h5)")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--qwen-pooling", choices=["mean", "last_token"],
                    default="last_token")
    ap.add_argument("--qwen-mode", choices=["embed", "interpret"],
                    default="embed",
                    help="'embed': pool Qwen's hidden state directly. "
                         "'interpret': generate English summary, then SBERT-embed it.")
    ap.add_argument("--sbert-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── 1. Load sentences ─────────────────────────────────────────────
    src = args.sentences_h5 or args.sentences_txt
    if src is None:
        raise SystemExit("Pass --sentences-h5 or --sentences-txt")
    sentences = load_sentences(src, args.num_samples, args.seed)
    logger.info("Loaded %d sentences", len(sentences))

    # ── 2. Load VAE, encode + decode ──────────────────────────────────
    (
        modules, sp, vocab_size, bos_id, eos_id,
        max_seq_len, num_memory_tokens,
    ) = load_vae(args.resume_checkpoint, args.spm_path, device)

    logger.info("Encoding %d sentences → z", len(sentences))
    latent_dim = modules["latent_to_decoder"].in_features
    z = encode_to_z(
        sentences, sp, modules, bos_id, eos_id, device, max_seq_len, latent_dim,
    )

    logger.info("Decoding z → surface text")
    surfaces = decode_z(
        z, modules, bos_id, eos_id, vocab_size, max_seq_len, sp,
        num_memory_tokens, device, latent_dim,
    )
    # Release VAE memory before loading Qwen
    del modules
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. SBERT source topology ──────────────────────────────────────
    src_emb = sbert_embed(sentences, args.sbert_model)
    logger.info("SBERT source: shape=%s", src_emb.shape)

    # ── 4. Qwen interpretation of decoder surfaces ────────────────────
    interpretations: list[str] | None = None
    if args.qwen_mode == "interpret":
        interpretations = qwen_interpret(surfaces, args.qwen_model, device)
        logger.info("Sample interpretations (first 3):")
        for i in range(min(3, len(interpretations))):
            logger.info("  [%d] %s", i, interpretations[i][:180])
        qwen_emb = sbert_embed(interpretations, args.sbert_model)
    else:
        qwen_emb = qwen_embed(
            surfaces, args.qwen_model, device, pooling=args.qwen_pooling,
        )
    logger.info("Qwen interpretation emb: shape=%s", qwen_emb.shape)

    # ── 5. Optional: SBERT of surfaces too (baseline-direct) ──────────
    sbert_surface_emb = sbert_embed(surfaces, args.sbert_model)

    # ── 6. Topology metrics ──────────────────────────────────────────
    D_src = pairwise_cosine_distance(src_emb)
    D_qwen = pairwise_cosine_distance(qwen_emb)
    D_sbert_surf = pairwise_cosine_distance(sbert_surface_emb)

    sp_qwen = stats.spearmanr(D_src, D_qwen)
    sp_sbert = stats.spearmanr(D_src, D_sbert_surf)

    # k-NN preservation
    F_src = full_cosine_distance_matrix(src_emb)
    F_qwen = full_cosine_distance_matrix(qwen_emb)
    knn10 = knn_preservation(F_src, F_qwen, k=10)
    knn20 = knn_preservation(F_src, F_qwen, k=20)

    logger.info("")
    logger.info("=" * 72)
    logger.info("Decoder + Qwen interpretation topology")
    logger.info("=" * 72)
    logger.info("  N samples                : %d", len(sentences))
    logger.info(
        "  source ↔ qwen(surface)   ρ = %+.4f  (p=%.2e)",
        sp_qwen.correlation, sp_qwen.pvalue,
    )
    logger.info(
        "  source ↔ sbert(surface)  ρ = %+.4f  (p=%.2e)  [baseline]",
        sp_sbert.correlation, sp_sbert.pvalue,
    )
    logger.info("  kNN preservation (k=10)  : %.3f", knn10)
    logger.info("  kNN preservation (k=20)  : %.3f", knn20)
    logger.info("")
    logger.info(
        "  mean surface len (chars) : %.1f",
        float(np.mean([len(s) for s in surfaces])),
    )
    logger.info("")
    logger.info("Sample pairs (first 3):")
    for i in range(min(3, len(sentences))):
        logger.info("  [%d] src: %s", i, sentences[i][:140])
        logger.info("      dec: %s", surfaces[i][:140])

    if args.output:
        result = {
            "num_samples": len(sentences),
            "spearman_source_to_qwen_r": float(sp_qwen.correlation),
            "spearman_source_to_qwen_p": float(sp_qwen.pvalue),
            "spearman_source_to_sbert_surface_r": float(sp_sbert.correlation),
            "spearman_source_to_sbert_surface_p": float(sp_sbert.pvalue),
            "knn_preservation_k10": knn10,
            "knn_preservation_k20": knn20,
            "mean_surface_len_chars": float(np.mean([len(s) for s in surfaces])),
            "qwen_model": args.qwen_model,
            "sbert_model": args.sbert_model,
            "checkpoint": str(args.resume_checkpoint),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
