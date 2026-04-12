"""Epoch-end diagnostics: decode, encode, metrics, and orchestration."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .config import VAEPretrainConfig, _IPA_VOWELS

logger = logging.getLogger(__name__)


def _is_vowel(char: str) -> bool:
    """Check if a single IPA character is a vowel."""
    return char in _IPA_VOWELS


def sample_decode(
    z: Tensor,
    *,
    modules: dict[str, nn.Module],
    cfg: VAEPretrainConfig,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    top_p: float = 0.9,
    temperature: float = 0.8,
    sp: object,
) -> list[str]:
    """Decode z via nucleus (top-p) sampling.

    Replaces greedy argmax + repetition penalties with
    sampling from the top-p probability mass.  Naturally
    produces diverse, non-repetitive output.
    """
    from lfm.generator.layers import (
        PhraseDecoder,
        multiscale_causal_mask,
    )

    _dec = modules["decoder"]
    _is_ling = isinstance(_dec, PhraseDecoder)
    n = z.size(0)
    _n_mem = getattr(cfg, "num_memory_tokens", 1)
    mem = modules["latent_to_decoder"](z).reshape(n, _n_mem, -1)
    ids = torch.full(
        (n, 1), bos_id, dtype=torch.long, device=device
    )
    for _t in range(cfg.max_seq_len - 1):
        if _is_ling:
            tgt = modules["dec_token_embedding"](ids)
            if not isinstance(
                modules["dec_pos_embedding"], nn.Identity
            ):
                p = torch.arange(
                    ids.size(1), device=device
                ).unsqueeze(0)
                tgt = tgt + modules["dec_pos_embedding"](p)
            cm = multiscale_causal_mask(
                ids.size(1),
                num_heads=cfg.decoder_num_heads,
                head_windows=cfg.attention_head_windows,
                global_every=cfg.attention_global_every,
                device=device,
            )
            out = _dec(
                tgt, mem, tgt_mask=cm,
                rope_freqs=modules.get("_rope_freqs"),
            )
        else:
            p = torch.arange(
                ids.size(1), device=device
            ).unsqueeze(0)
            tok = modules["dec_token_embedding"](ids)
            tgt = tok + modules["dec_pos_embedding"](p)
            cm = nn.Transformer.generate_square_subsequent_mask(
                ids.size(1), device=device
            )
            out = _dec(tgt=tgt, memory=mem, tgt_mask=cm)

        logits = modules["output_head"](out[:, -1])
        # Suppress special tokens
        logits[:, 0:4] = float("-inf")
        logits[:, bos_id] = float("-inf")
        # Temperature
        logits = logits / max(temperature, 1e-8)
        # Nucleus (top-p) filtering
        sorted_logits, sorted_idx = torch.sort(
            logits, descending=True
        )
        cumprobs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        remove = cumprobs - F.softmax(
            sorted_logits, dim=-1
        ) >= top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(
            1, sorted_idx, sorted_logits
        )
        # Sample
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    # Decode single-sentence output: truncate at first EOS.
    # Specials differ by backend: SPM reserves ids 0-3 for UNK/BOS/EOS/PAD,
    # while the phoneme alphabet uses 0..vocab_size-1 all as valid phonemes.
    texts = []
    _spm_internal = {0, 1, 2, 3} if hasattr(sp, "id_to_piece") else set()
    _specials = _spm_internal | {bos_id, eos_id}
    for j in range(n):
        toks = ids[j, 1:].cpu().tolist()
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        toks = [x for x in toks if x < vocab_size and x not in _specials]
        text = _backend_decode(sp, toks)
        # Strip trailing orphan consonants (IPA-specific heuristic; harmless
        # on phoneme output since phonemes are multi-char).
        words = text.split()
        while words and len(words[-1]) == 1 and not _is_vowel(words[-1]):
            words.pop()
        texts.append(" ".join(words))
    return texts


def _backend_decode(backend: object, ids: list[int]) -> str:
    """Decode token ids via whichever backend is present (SPM or phoneme)."""
    if backend is None:
        return ""
    # SentencePieceProcessor exposes `.decode(list[int]) -> str` directly.
    if hasattr(backend, "id_to_piece"):
        return backend.decode(ids)  # type: ignore[attr-defined]
    # PhonemeTokenizer exposes `.batch_decode([list[int]]) -> list[str]`.
    if hasattr(backend, "batch_decode"):
        return backend.batch_decode([ids])[0]  # type: ignore[attr-defined]
    return ""


def encode_text(
    tokens: Tensor,
    lengths: Tensor,
    *,
    modules: dict[str, nn.Module],
    cfg: VAEPretrainConfig,
    device: torch.device,
) -> Tensor:
    """Encode token batch -> z via the VAE encoder."""
    sl = tokens.size(1)
    src_mask = (
        torch.arange(sl, device=device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    pos = torch.arange(sl, device=device).unsqueeze(0)
    enc_tok = modules["enc_token_embedding"](tokens)
    enc_in = enc_tok + modules["enc_pos_embedding"](pos)
    enc_out = modules["encoder"](enc_in, src_key_padding_mask=~src_mask)
    masked = enc_out * src_mask.unsqueeze(-1).float()
    denom = lengths.unsqueeze(-1).float().clamp(min=1)
    pooled = masked.sum(dim=1) / denom
    h = modules["enc_to_latent"](pooled)
    if cfg.use_vq and modules.get("_residual_vq") is not None:
        # VQ mode: h is the continuous z, quantize it
        mu, _, _ = modules["_residual_vq"](h)
    else:
        mu, _ = h.chunk(2, dim=-1)
    return mu  # deterministic at eval


def word_edit_distance(a: str, b: str) -> tuple[int, int]:
    """Word-level Levenshtein distance. Returns (distance, max_len)."""
    wa, wb = a.split(), b.split()
    n, m = len(wa), len(wb)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if wa[i - 1] == wb[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m], max(n, m)


def structural_metrics(texts: list[str]) -> dict[str, float]:
    """Compute TTR, repetition rate, and mean word length."""
    ttrs, rep_rates, word_lens = [], [], []
    for t in texts:
        words = t.split()
        if not words:
            continue
        # Type-token ratio
        ttrs.append(len(set(words)) / len(words))
        # Bigram repetition rate
        bigrams = [
            (words[i], words[i + 1])
            for i in range(len(words) - 1)
        ]
        if bigrams:
            rep_rates.append(
                1.0 - len(set(bigrams)) / len(bigrams)
            )
        # Mean word length (IPA chars)
        word_lens.extend(len(w) for w in words)
    return {
        "ttr": sum(ttrs) / max(len(ttrs), 1),
        "rep_rate": sum(rep_rates) / max(len(rep_rates), 1),
        "mean_word_len": (
            sum(word_lens) / max(len(word_lens), 1)
        ),
    }


def surface_diversity(
    *,
    modules: dict[str, nn.Module],
    cfg: VAEPretrainConfig,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    sp: object,
    z_running_mean: Tensor,
    z_running_std: Tensor,
    n_samples: int = 200,
) -> dict[str, float]:
    """Measure how many unique token sequences the decoder produces.

    Samples random z vectors from the pretrained distribution, decodes
    each, and computes diversity metrics.  This is the key diagnostic
    for whether the decoder maps different z regions to different
    surface forms.

    Returns:
        Dict with ``unique``, ``total``, ``diversity_ratio``,
        ``mean_pairwise_edit``, and ``mean_length``.
    """
    # Sample from the learned z distribution
    z = torch.randn(n_samples, cfg.latent_dim, device=device) * z_running_std + z_running_mean

    texts = sample_decode(
        z,
        modules=modules,
        cfg=cfg,
        vocab_size=vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        sp=sp,
    )

    # Count unique sequences
    unique = len(set(texts))
    total = len(texts)
    diversity_ratio = unique / max(total, 1)

    # Mean pairwise edit distance (subsample for speed)
    import random as _rng
    n_pairs = min(500, total * (total - 1) // 2)
    indices = list(range(total))
    edit_dists = []
    for _ in range(n_pairs):
        i, j = _rng.sample(indices, 2)
        ed, ml = word_edit_distance(texts[i], texts[j])
        if ml > 0:
            edit_dists.append(ed / ml)
    mean_edit = sum(edit_dists) / max(len(edit_dists), 1)

    # Mean length
    lengths = [len(t.split()) for t in texts]
    mean_length = sum(lengths) / max(len(lengths), 1)

    return {
        "unique": unique,
        "total": total,
        "diversity_ratio": diversity_ratio,
        "mean_pairwise_edit": mean_edit,
        "mean_length": mean_length,
    }


def run_epoch_diagnostics(
    *,
    epoch: int,
    cfg: VAEPretrainConfig,
    modules: dict[str, nn.Module],
    device: torch.device,
    vocab_size: int,
    full_vocab: int | None = None,
    bos_id: int,
    eos_id: int,
    sp: object,
    val_dataset: object,
    val_loader: object,
    dataset: object,
    languages_list: list[str],
    z_running_mean: Tensor,
    z_running_std: Tensor,
    label: str = "",
    constituent_dataset: object | None = None,
) -> None:
    """Run reconstruction, interpolation, perturbation, random, length, and structural diagnostics.

    Args:
        label: Optional label for mid-epoch diagnostics (e.g., "step 5000").
        constituent_dataset: If provided, sample recon examples from
            constituent data instead of full sentences.
    """
    # Common kwargs for sample_decode
    _decode_kw = dict(
        modules=modules,
        cfg=cfg,
        vocab_size=vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        sp=sp,
    )

    # Fixed seed per epoch for reproducible sampling
    torch.manual_seed(cfg.seed + epoch)

    # Select recon samples: from constituent data if available,
    # otherwise from the full-sentence validation set.
    if constituent_dataset is not None:
        # Sample a few constituents for reconstruction diagnostic
        import random as _rng
        _rng.seed(cfg.seed + epoch)
        _n_const = len(constituent_dataset)
        _const_indices = _rng.sample(range(_n_const), min(2, _n_const))
        _toks = []
        _lens = []
        _lang_labels = ["const", "const"]
        for _ci in _const_indices:
            _item = constituent_dataset[_ci]
            # ConstituentDataset returns (enc_tokens, enc_len, dec_tokens, dec_len)
            _toks.append(_item[2])  # decoder target (constituent)
            _lens.append(_item[3])
        val_batch_tokens = torch.stack(_toks).to(device)
        val_batch_lengths = torch.tensor(_lens, device=device)
    else:
        # Find language-specific samples from the validation set.
        _val_indices = list(val_dataset.indices)
        _eng_idx = _non_eng_idx = None
        _lang_labels = ["?", "?"]
        for _vi in _val_indices:
            _lang = languages_list[_vi] if _vi < len(languages_list) else "?"
            if _lang == "eng" and _eng_idx is None:
                _eng_idx = _vi
            elif _lang != "eng" and _non_eng_idx is None:
                _non_eng_idx = _lang
                _non_eng_dataset_idx = _vi
            if _eng_idx is not None and _non_eng_idx is not None:
                break

        _sample_indices = []
        if _eng_idx is not None:
            _sample_indices.append(_eng_idx)
            _lang_labels[0] = "eng"
        if _non_eng_idx is not None:
            _sample_indices.append(_non_eng_dataset_idx)
            _lang_labels[1] = _non_eng_idx
        if len(_sample_indices) < 2:
            val_batch_tokens, val_batch_lengths = next(iter(val_loader))
            val_batch_tokens = val_batch_tokens[:2].to(device)
            val_batch_lengths = torch.as_tensor(
                val_batch_lengths[:2], device=device
            )
        else:
            _toks = []
            _lens = []
            for _si in _sample_indices:
                _t, _l = dataset[_si]
                _toks.append(_t)
                _lens.append(_l)
            val_batch_tokens = torch.stack(_toks).to(device)
            val_batch_lengths = torch.tensor(_lens, device=device)

    # --- 1. Reconstruction ---
    z_real = encode_text(
        val_batch_tokens, val_batch_lengths,
        modules=modules, cfg=cfg, device=device,
    )
    recon_texts = sample_decode(z_real, **_decode_kw)

    for j in range(min(2, len(_lang_labels))):
        orig_ids = val_batch_tokens[j].cpu().tolist()
        orig_ids = [x for x in orig_ids[:val_batch_lengths[j]] if x < vocab_size]
        orig = _backend_decode(sp, orig_ids)
        dec = recon_texts[j]
        ed, ml = word_edit_distance(orig, dec)
        wer = ed / max(ml, 1)
        logger.info(
            "  recon[%s] orig: %s", _lang_labels[j], orig[:120]
        )
        logger.info(
            "  recon[%s]  dec: %s  [WED=%d/%d WER=%.0f%%]",
            _lang_labels[j], dec[:120], ed, ml, wer * 100,
        )

    # --- 2. Interpolation (English <-> non-English) ---
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    z_interp = torch.stack(
        [z_real[0] * (1 - a) + z_real[1] * a for a in alphas]
    )
    if cfg.use_vq and modules.get("_residual_vq") is not None:
        z_interp, _, _ = modules["_residual_vq"](z_interp)
    interp_texts = sample_decode(z_interp, **_decode_kw)
    logger.info(
        "  interp: %s -> %s", _lang_labels[0], _lang_labels[1]
    )
    for k, (a, txt) in enumerate(zip(alphas, interp_texts)):
        logger.info("  interp[%.2f]: %s", a, txt[:120])

    # --- 3. Perturbation (around English sentence) ---
    # Use wider sigma range (0-2) to show meaningful variation
    # as z_std compresses during training.
    noise_scales = [0.0, 0.5, 1.0, 2.0]
    # Generate noise with distinct seeds per scale to guarantee diversity
    perturb_noise = []
    for si, s in enumerate(noise_scales):
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed + epoch * 100 + si)
        noise = torch.randn(cfg.latent_dim, device=device, generator=gen)
        perturb_noise.append(z_real[0] + s * z_running_std * noise)
    z_perturbed = torch.stack(perturb_noise)
    if cfg.use_vq and modules.get("_residual_vq") is not None:
        z_perturbed, _, _ = modules["_residual_vq"](z_perturbed)
    perturb_texts = sample_decode(z_perturbed, **_decode_kw)
    logger.info("  perturb: around %s sentence", _lang_labels[0])
    for k, (s, txt) in enumerate(zip(noise_scales, perturb_texts)):
        logger.info("  perturb[σ=%.1f]: %s", s, txt[:120])

    # --- 4. Random z near English cluster ---
    # Sample from the encoder distribution, then also sample
    # near the English z for interpretable comparison.
    z_random = (
        torch.randn(3, cfg.latent_dim, device=device)
        * z_running_std + z_running_mean
    )
    # 2 more near the English z — use distinct seeds to guarantee different output
    gen_a = torch.Generator(device=device)
    gen_a.manual_seed(cfg.seed + epoch * 100 + 10)
    gen_b = torch.Generator(device=device)
    gen_b.manual_seed(cfg.seed + epoch * 100 + 11)
    z_near_eng = torch.stack([
        z_real[0] + 0.5 * z_running_std * torch.randn(cfg.latent_dim, device=device, generator=gen_a),
        z_real[0] + 0.5 * z_running_std * torch.randn(cfg.latent_dim, device=device, generator=gen_b),
    ])
    z_all_random = torch.cat([z_random, z_near_eng], dim=0)
    if cfg.use_vq and modules.get("_residual_vq") is not None:
        z_all_random, _, _ = modules["_residual_vq"](z_all_random)
    random_texts = sample_decode(z_all_random, **_decode_kw)
    for j in range(3):
        logger.info("  random[%d]: %s", j, random_texts[j][:120])
    for j in range(3, 5):
        logger.info(
            "  near_eng[%d]: %s", j - 3, random_texts[j][:120]
        )

    # --- 5. Length distribution (autoregressive decode) ---
    # Decode a batch of z sampled from the encoder distribution
    # to check EOS behavior and length variation.
    z_length_test = (
        torch.randn(32, cfg.latent_dim, device=device)
        * z_running_std + z_running_mean
    )
    if cfg.use_vq and modules.get("_residual_vq") is not None:
        z_length_test, _, _ = modules["_residual_vq"](z_length_test)
    length_texts = sample_decode(z_length_test, **_decode_kw)
    word_counts = [len(t.split()) for t in length_texts]
    char_counts = [len(t) for t in length_texts]
    logger.info(
        "  length_dist (32 random z): "
        "words: min=%d max=%d mean=%.1f std=%.1f | "
        "chars: min=%d max=%d mean=%.1f std=%.1f",
        min(word_counts), max(word_counts),
        sum(word_counts) / len(word_counts),
        (sum((w - sum(word_counts) / len(word_counts)) ** 2
             for w in word_counts) / len(word_counts)) ** 0.5,
        min(char_counts), max(char_counts),
        sum(char_counts) / len(char_counts),
        (sum((c - sum(char_counts) / len(char_counts)) ** 2
             for c in char_counts) / len(char_counts)) ** 0.5,
    )

    # --- 6. Structural health metrics ---
    all_eval_texts = (
        recon_texts + interp_texts + perturb_texts
        + random_texts + length_texts
    )

    m = structural_metrics(all_eval_texts)
    # EOS rate: fraction of length-test decodes that emitted
    # EOS before max_seq_len (measures stopping ability)
    eos_rate = sum(
        1 for t in length_texts if len(t.split()) < cfg.max_seq_len // 2
    ) / max(len(length_texts), 1)
    logger.info(
        "  struct: TTR=%.3f rep_rate=%.3f "
        "mean_word_len=%.1f eos_rate=%.2f",
        m["ttr"],
        m["rep_rate"],
        m["mean_word_len"],
        eos_rate,
    )

    # --- 7. Surface diversity ---
    # Free cached tensors before decoding to avoid OOM
    torch.cuda.empty_cache()
    sd = surface_diversity(
        modules=modules,
        cfg=cfg,
        vocab_size=vocab_size,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        sp=sp,
        z_running_mean=z_running_mean,
        z_running_std=z_running_std,
        n_samples=64,
    )
    logger.info(
        "  surface_div: %d/%d unique (%.1f%%) "
        "mean_edit=%.3f mean_len=%.1f",
        sd["unique"], sd["total"],
        sd["diversity_ratio"] * 100,
        sd["mean_pairwise_edit"],
        sd["mean_length"],
    )
