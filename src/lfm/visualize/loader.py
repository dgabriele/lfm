"""Model loading and corpus encoding for visualizations.

Reconstructs the VAE from a checkpoint and encodes corpus subsets
into the latent space for downstream analysis.
"""

from __future__ import annotations

import logging
import random

import torch
from torch import Tensor, nn

from lfm.visualize.config import VisualizeConfig

logger = logging.getLogger(__name__)


def load_checkpoint(config: VisualizeConfig) -> dict:
    """Load VAE checkpoint and reconstruct model components.

    The ``vae_resume.pt`` checkpoint contains full encoder + decoder state
    dicts keyed by the module names used in ``VAEPretrainer._build_model()``.

    Returns:
        Dict with keys: ``modules`` (dict of nn.Module), ``config`` (arch params),
        ``device`` (torch.device).
    """
    device = torch.device(config.device)
    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=False)

    # Resume checkpoint stores modules dict and arch config
    if "modules" in ckpt:
        return _load_resume_checkpoint(ckpt, config, device)
    # Decoder-only checkpoint (vae_decoder.pt)
    return _load_decoder_checkpoint(ckpt, config, device)


def _load_resume_checkpoint(
    ckpt: dict, config: VisualizeConfig, device: torch.device
) -> dict:
    """Load from full training resume checkpoint (vae_resume.pt)."""
    from lfm.generator.layers import LinguisticDecoder, precompute_rope_freqs
    from lfm.generator.pretrain import VAEPretrainConfig

    cfg = VAEPretrainConfig()  # defaults match the checkpoint
    hidden = cfg.decoder_hidden_dim
    full_vocab = cfg.spm_vocab_size + 2

    # Reconstruct all modules
    enc_token_embedding = nn.Embedding(full_vocab, hidden).to(device)
    enc_pos_embedding = nn.Embedding(cfg.max_seq_len, hidden).to(device)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden,
        nhead=cfg.decoder_num_heads,
        dim_feedforward=hidden * 4,
        dropout=cfg.decoder_dropout,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=cfg.encoder_num_layers,
        enable_nested_tensor=False,
    ).to(device)

    enc_to_latent = nn.Linear(hidden, cfg.latent_dim * 2).to(device)
    latent_to_decoder = nn.Linear(cfg.latent_dim, hidden).to(device)
    dec_token_embedding = nn.Embedding(full_vocab, hidden).to(device)

    if cfg.use_rope:
        dec_pos_embedding: nn.Module = nn.Identity()
    else:
        dec_pos_embedding = nn.Embedding(cfg.max_seq_len, hidden).to(device)

    decoder = LinguisticDecoder(
        d_model=hidden,
        nhead=cfg.decoder_num_heads,
        num_layers=cfg.decoder_num_layers,
        dim_feedforward=hidden * 4,
        dropout=cfg.decoder_dropout,
        share_layers=cfg.share_decoder_layers,
    ).to(device)

    output_head = nn.Linear(hidden, full_vocab).to(device)

    modules = {
        "enc_token_embedding": enc_token_embedding,
        "enc_pos_embedding": enc_pos_embedding,
        "encoder": encoder,
        "enc_to_latent": enc_to_latent,
        "latent_to_decoder": latent_to_decoder,
        "dec_token_embedding": dec_token_embedding,
        "dec_pos_embedding": dec_pos_embedding,
        "decoder": decoder,
        "output_head": output_head,
    }

    # Load state dicts
    saved_modules = ckpt["modules"]
    for name, module in modules.items():
        if name in saved_modules and isinstance(module, nn.Module):
            module.load_state_dict(saved_modules[name])

    # Precompute RoPE and mask
    if cfg.use_rope:
        from lfm.generator.layers import multiscale_causal_mask

        head_dim = hidden // cfg.decoder_num_heads
        rope_freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len + 1, device=device)
        cached_mask = multiscale_causal_mask(
            cfg.max_seq_len + 1,
            num_heads=cfg.decoder_num_heads,
            head_windows=cfg.attention_head_windows,
            global_every=cfg.attention_global_every,
            device=device,
        )
    else:
        rope_freqs = None
        cached_mask = None

    result = {
        "modules": modules,
        "rope_freqs": rope_freqs,
        "cached_mask": cached_mask,
        "cfg": cfg,
        "device": device,
        "full_vocab": full_vocab,
        "bos_id": cfg.spm_vocab_size,
        "eos_id": cfg.spm_vocab_size + 1,
    }
    if "z_mean" in ckpt:
        result["z_mean"] = ckpt["z_mean"]
        result["z_std"] = ckpt["z_std"]
    return result


def _load_decoder_checkpoint(
    ckpt: dict, config: VisualizeConfig, device: torch.device
) -> dict:
    """Load from decoder-only checkpoint (vae_decoder.pt)."""
    from lfm.generator.layers import (
        LinguisticDecoder,
        multiscale_causal_mask,
        precompute_rope_freqs,
    )
    from lfm.generator.pretrain import VAEPretrainConfig

    cfg = VAEPretrainConfig()
    latent_dim = ckpt.get("latent_dim", cfg.latent_dim)
    hidden = ckpt.get("decoder_hidden_dim", cfg.decoder_hidden_dim)
    num_layers = ckpt.get("decoder_num_layers", cfg.decoder_num_layers)
    num_heads = ckpt.get("decoder_num_heads", cfg.decoder_num_heads)
    vocab_size = ckpt.get("vocab_size", cfg.spm_vocab_size)
    max_seq_len = ckpt.get("max_seq_len", cfg.max_seq_len)
    full_vocab = vocab_size + 2

    latent_to_decoder = nn.Linear(latent_dim, hidden).to(device)
    latent_to_decoder.load_state_dict(ckpt["latent_to_decoder"])

    dec_token_embedding = nn.Embedding(full_vocab, hidden).to(device)
    dec_token_embedding.load_state_dict(ckpt["token_embedding"])

    decoder = LinguisticDecoder(
        d_model=hidden,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=hidden * 4,
        share_layers=cfg.share_decoder_layers,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])

    output_head = nn.Linear(hidden, full_vocab).to(device)
    output_head.load_state_dict(ckpt["output_head"])

    head_dim = hidden // num_heads
    rope_freqs = precompute_rope_freqs(head_dim, max_seq_len + 1, device=device)
    cached_mask = multiscale_causal_mask(
        max_seq_len + 1,
        num_heads=num_heads,
        head_windows=cfg.attention_head_windows,
        global_every=cfg.attention_global_every,
        device=device,
    )

    modules = {
        "latent_to_decoder": latent_to_decoder,
        "dec_token_embedding": dec_token_embedding,
        "decoder": decoder,
        "output_head": output_head,
    }

    return {
        "modules": modules,
        "rope_freqs": rope_freqs,
        "cached_mask": cached_mask,
        "cfg": cfg,
        "device": device,
        "full_vocab": full_vocab,
        "bos_id": vocab_size,
        "eos_id": vocab_size + 1,
        "z_mean": ckpt.get("z_mean"),
        "z_std": ckpt.get("z_std"),
    }


def _load_cache(config: VisualizeConfig) -> dict:
    """Load the preprocessed corpus cache.

    Handles both v1 (no languages) and v2 (with languages) cache formats.
    """
    cache = torch.load(config.corpus_cache, map_location="cpu", weights_only=False)
    result = {
        "token_ids_list": cache["token_ids_list"],
        "vocab_size": cache["vocab_size"],
    }
    if "languages" in cache:
        result["languages"] = cache["languages"]
    else:
        logger.warning(
            "Cache at %s uses v1 format (no language labels). "
            "Regenerate with the latest pretrain pipeline for labeled data. "
            "Falling back to unlabeled mode.",
            config.corpus_cache,
        )
    return result


@torch.no_grad()
def _encode_token_ids(
    token_ids_list: list[list[int]],
    vocab_size: int,
    model_data: dict,
    config: VisualizeConfig,
) -> Tensor:
    """Encode token ID sequences through the VAE encoder → z vectors.

    Args:
        token_ids_list: Pre-tokenized sequences.
        vocab_size: SPM vocabulary size.
        model_data: From ``load_checkpoint()``.
        config: Visualization config.

    Returns:
        Tensor of shape ``(N, latent_dim)`` — the encoder mean (mu).
    """
    device = model_data["device"]
    modules = model_data["modules"]
    cfg = model_data["cfg"]

    from torch.utils.data import DataLoader

    from lfm.data.corpus import MultilingualCorpusDataset

    eos_id = vocab_size + 1
    dataset = MultilingualCorpusDataset(token_ids_list, cfg.max_seq_len, eos_id)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    enc_tok = modules["enc_token_embedding"]
    enc_pos = modules["enc_pos_embedding"]
    encoder = modules["encoder"]
    enc_to_latent = modules["enc_to_latent"]

    for m in modules.values():
        if isinstance(m, nn.Module):
            m.eval()

    all_mu: list[Tensor] = []

    for batch_tokens, batch_lengths in loader:
        batch_tokens = batch_tokens.to(device)
        batch_lengths = torch.as_tensor(batch_lengths, device=device)
        b, seq_len = batch_tokens.shape

        src_mask = (
            torch.arange(seq_len, device=device).unsqueeze(0)
            < batch_lengths.unsqueeze(1)
        )
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        enc_input = enc_tok(batch_tokens) + enc_pos(pos_ids)
        enc_out = encoder(enc_input, src_key_padding_mask=~src_mask)

        enc_masked = enc_out * src_mask.unsqueeze(-1).float()
        denom = batch_lengths.unsqueeze(-1).float().clamp(min=1)
        pooled = enc_masked.sum(dim=1) / denom

        h = enc_to_latent(pooled)
        mu, _ = h.chunk(2, dim=-1)
        all_mu.append(mu.cpu())

    return torch.cat(all_mu, dim=0)


@torch.no_grad()
def encode_corpus(
    model_data: dict,
    config: VisualizeConfig,
) -> dict:
    """Encode the cached corpus through the encoder to get z vectors.

    Requires a resume checkpoint (with encoder). Falls back to the corpus
    cache for token IDs.

    Returns:
        Dict with ``z`` (N, latent_dim), ``token_ids_list``, ``vocab_size``,
        and ``languages`` (if available in cache).
    """
    modules = model_data["modules"]
    cache = _load_cache(config)
    token_ids_list = cache["token_ids_list"]
    vocab_size = cache["vocab_size"]
    languages = cache.get("languages")

    # Subsample if needed (keep language alignment)
    if len(token_ids_list) > config.max_samples:
        rng = random.Random(config.seed)
        indices = rng.sample(range(len(token_ids_list)), config.max_samples)
        token_ids_list = [token_ids_list[i] for i in indices]
        if languages is not None:
            languages = [languages[i] for i in indices]

    logger.info("Encoding %d sequences...", len(token_ids_list))

    if "encoder" not in modules:
        logger.warning("No encoder in checkpoint — cannot produce z vectors")
        result: dict = {"token_ids_list": token_ids_list, "vocab_size": vocab_size}
        if languages is not None:
            result["languages"] = languages
        return result

    z = _encode_token_ids(token_ids_list, vocab_size, model_data, config)
    logger.info("Encoded %d sequences → z shape %s", z.size(0), tuple(z.shape))

    result = {
        "z": z,
        "mu": z,
        "token_ids_list": token_ids_list,
        "vocab_size": vocab_size,
    }
    if languages is not None:
        result["languages"] = languages
    return result


@torch.no_grad()
def encode_labeled_corpus(
    model_data: dict,
    config: VisualizeConfig,
) -> dict:
    """Encode corpus with per-sentence language labels.

    Uses the v2 cache (with language labels) when available.  Falls back to
    re-loading from Leipzig corpus if the cache is v1 format.

    Returns:
        Dict with ``z`` (N, latent_dim), ``languages`` (list[str]),
        ``token_ids_list``, ``vocab_size``.
    """
    modules = model_data["modules"]

    if "encoder" not in modules:
        raise RuntimeError("Labeled encoding requires a resume checkpoint with encoder")

    cache = _load_cache(config)
    token_ids_list = cache["token_ids_list"]
    vocab_size = cache["vocab_size"]
    languages = cache.get("languages")

    if languages is None:
        # v1 cache — fall back to re-loading with labels
        return _encode_labeled_fallback(model_data, config)

    # Per-language balanced subsampling to avoid visual/statistical bias
    if len(token_ids_list) > config.max_samples:
        from collections import defaultdict

        rng = random.Random(config.seed)
        by_lang: dict[str, list[int]] = defaultdict(list)
        for idx, lang in enumerate(languages):
            by_lang[lang].append(idx)

        n_langs = len(by_lang)
        per_lang = config.max_samples // max(n_langs, 1)
        indices: list[int] = []
        for lang in sorted(by_lang.keys()):
            lang_indices = by_lang[lang]
            if len(lang_indices) > per_lang:
                lang_indices = rng.sample(lang_indices, per_lang)
            indices.extend(lang_indices)
        rng.shuffle(indices)

        token_ids_list = [token_ids_list[i] for i in indices]
        languages = [languages[i] for i in indices]

    logger.info("Encoding %d labeled sequences...", len(token_ids_list))
    z = _encode_token_ids(token_ids_list, vocab_size, model_data, config)
    logger.info("Encoded %d labeled sequences → z shape %s", z.size(0), tuple(z.shape))

    return {
        "z": z,
        "languages": languages,
        "token_ids_list": token_ids_list,
        "vocab_size": vocab_size,
    }


def _encode_labeled_fallback(
    model_data: dict,
    config: VisualizeConfig,
    max_per_language: int = 5000,
) -> dict:
    """Fallback for v1 caches: re-load Leipzig corpus with language labels.

    This is the slow path — re-runs IPA conversion.
    """
    cfg = model_data["cfg"]

    from lfm.data.loaders.ipa import IPAConverter
    from lfm.data.loaders.leipzig import LeipzigCorpusConfig, LeipzigCorpusLoader
    from lfm.generator.pretrain import _sanitize_samples

    loader_cfg = LeipzigCorpusConfig(
        data_dir=str(cfg.corpus_loader_config.get("data_dir", "data/leipzig")),
        max_samples_per_language=max_per_language,
    )
    corpus_loader = LeipzigCorpusLoader(loader_cfg)
    samples = corpus_loader.load()

    if not samples:
        raise RuntimeError("No corpus samples loaded. Check data/leipzig directory.")

    rng = random.Random(config.seed)
    if len(samples) > config.max_samples:
        samples = rng.sample(samples, config.max_samples)

    logger.info("Fallback: loaded %d labeled samples, converting to IPA...", len(samples))
    samples = _sanitize_samples(samples)

    converter = IPAConverter()
    labeled_ipa: list[tuple[str, str]] = []
    for lang, text in samples:
        ipa = converter.convert_line(lang, text)
        if ipa and len(ipa) >= 10:
            labeled_ipa.append((lang, ipa))

    if not labeled_ipa:
        raise RuntimeError("No samples survived IPA conversion")

    import sentencepiece as spm_lib

    sp = spm_lib.SentencePieceProcessor(model_file=config.spm_model)
    vocab_size = sp.vocab_size()
    _spm_specials = {0, 1, 2, 3}

    languages: list[str] = []
    token_ids_list: list[list[int]] = []

    for lang, ipa_text in labeled_ipa:
        ids = sp.encode(ipa_text, out_type=int)
        ids = [x for x in ids if x not in _spm_specials]
        if len(ids) >= 5:
            languages.append(lang)
            token_ids_list.append(ids)

    logger.info("Tokenized %d labeled sequences (fallback)", len(token_ids_list))

    z = _encode_token_ids(token_ids_list, vocab_size, model_data, config)
    logger.info("Encoded %d labeled sequences → z shape %s", z.size(0), tuple(z.shape))

    return {
        "z": z,
        "languages": languages,
        "token_ids_list": token_ids_list,
        "vocab_size": vocab_size,
    }


@torch.no_grad()
def decode_z(
    z: Tensor,
    model_data: dict,
    config: VisualizeConfig,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> list[list[int]]:
    """Decode latent z vectors through the decoder using nucleus sampling.

    Uses KV caching when the decoder is a ``LinguisticDecoder`` — each
    step only computes Q/K/V for the new token, reusing cached K/V from
    previous steps.  This is O(n) per step instead of O(n²).

    Args:
        z: Latent codes ``(N, latent_dim)``.
        model_data: From ``load_checkpoint()``.
        config: Visualization config.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        List of decoded token ID lists.
    """
    from lfm.generator.layers import LinguisticDecoder

    device = model_data["device"]
    modules = model_data["modules"]
    cfg = model_data["cfg"]
    rope_freqs = model_data.get("rope_freqs")
    cached_mask = model_data.get("cached_mask")
    bos_id = model_data["bos_id"]
    eos_id = model_data["eos_id"]

    latent_to_decoder = modules["latent_to_decoder"]
    dec_tok = modules["dec_token_embedding"]
    decoder = modules["decoder"]
    output_head = modules["output_head"]

    for m in modules.values():
        if isinstance(m, nn.Module):
            m.eval()

    use_kv_cache = isinstance(decoder, LinguisticDecoder) and cached_mask is not None
    all_tokens: list[list[int]] = []
    max_len = cfg.max_seq_len

    for start in range(0, z.size(0), config.batch_size):
        z_batch = z[start : start + config.batch_size].to(device)
        b = z_batch.size(0)

        memory = latent_to_decoder(z_batch).unsqueeze(1)
        cur_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=device)

        if use_kv_cache:
            kv_cache = decoder.make_kv_cache(
                b, max_len + 1, device, dtype=memory.dtype,
            )

            # Process BOS token to prime the cache
            bos_embed = dec_tok(cur_ids)  # (B, 1, H)
            mask_row = cached_mask[:, 0:1, 0:1]  # (H, 1, 1)
            out = decoder.forward_cached(
                bos_embed, memory, kv_cache,
                rope_freqs=rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()

            for t in range(max_len):
                logits = output_head(out[:, -1]) / temperature

                # Nucleus sampling
                probs = _nucleus_sample(logits, top_p)
                next_token = torch.multinomial(probs, num_samples=1)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

                # Cached forward for new token only
                new_embed = dec_tok(next_token)  # (B, 1, H)
                seq_so_far = kv_cache.seq_len + 1
                mask_row = cached_mask[:, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
        else:
            # Fallback: no KV cache (non-LinguisticDecoder)
            for t in range(max_len):
                seq_len = cur_ids.size(1)
                tgt = dec_tok(cur_ids)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=device
                )
                out = decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                logits = output_head(out[:, -1]) / temperature

                probs = _nucleus_sample(logits, top_p)
                next_token = torch.multinomial(probs, num_samples=1)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

        # Convert to lists, truncate at EOS
        for row in cur_ids[:, 1:]:  # skip BOS
            tokens = row.tolist()
            if eos_id in tokens:
                tokens = tokens[: tokens.index(eos_id)]
            all_tokens.append(tokens)

    return all_tokens


def _nucleus_sample(logits: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) sampling and return probabilities."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_mask] = float("-inf")
    logits = logits.scatter(1, sorted_idx, sorted_logits)
    return torch.softmax(logits, dim=-1)
