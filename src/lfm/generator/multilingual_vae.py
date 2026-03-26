"""Multilingual VAE generator implementation.

Generates structurally well-formed subword token sequences from agent
embeddings by projecting into a pretrained VAE latent space and decoding
through a frozen autoregressive transformer.  The decoder's training on
typologically diverse natural language data ensures outputs are
phonotactically valid, morphologically structured, and compositionally
organized — without any stage-by-stage constraint specification.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.generator.base import GeneratorModule
from lfm.generator.config import GeneratorConfig
from lfm.generator.tokenizer import SubwordTokenizer
from lfm.utils.sampling import gumbel_softmax
from lfm.utils.tensor import create_padding_mask, masked_mean

logger = logging.getLogger(__name__)


@register("generator", "multilingual_vae")
class MultilingualVAEGenerator(GeneratorModule):
    """Multilingual VAE generator with frozen autoregressive decoder.

    Projects input embeddings into a VAE latent space and decodes through
    an autoregressive transformer to produce variable-length subword token
    sequences.  The decoder is frozen after pretraining on multilingual
    data; only the input projection is learned during agent training.

    Args:
        config: Generator configuration specifying architecture, latent
            dimensions, and pretrained checkpoint paths.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config)

        self._latent_dim = config.latent_dim
        self._vocab_size = config.vocab_size
        self._full_vocab = config.vocab_size + 2  # +BOS +EOS
        self._max_output_len = config.max_output_len
        self._kl_weight = config.kl_weight
        self._kl_free_bits = config.kl_free_bits
        self._hard_sample = config.hard_sample
        self._pooling = config.pooling

        self.bos_id = config.vocab_size
        self.eos_id = config.vocab_size + 1

        # -- Input projection (lazy-init: input dim unknown until first call) --
        # Residual architecture: linear path provides direct gradient signal
        # for REINFORCE from step 0, while the MLP refines the mapping.
        #   z = linear(x) + mlp(x)
        self._input_proj: nn.Linear | None = None  # linear path
        self._input_refine: nn.Sequential | None = None  # MLP residual path
        self._attention_pool_query: nn.Parameter | None = None

        # -- Decoder (linguistic architecture) --
        from lfm.generator.layers import (
            LinguisticDecoder,
            precompute_rope_freqs,
        )

        self.latent_to_decoder = nn.Linear(config.latent_dim, config.decoder_hidden_dim)
        self.token_embedding = nn.Embedding(self._full_vocab, config.decoder_hidden_dim)

        # Positional embedding: only used when RoPE is disabled
        if config.use_rope:
            self.pos_embedding: nn.Module = nn.Identity()
        else:
            self.pos_embedding = nn.Embedding(
                config.max_output_len, config.decoder_hidden_dim
            )

        self.decoder = LinguisticDecoder(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            num_layers=config.decoder_num_layers,
            dim_feedforward=config.decoder_hidden_dim * 4,
            dropout=config.decoder_dropout,
            share_layers=config.share_decoder_layers,
        )
        self.output_head = nn.Linear(config.decoder_hidden_dim, self._full_vocab)

        # Precompute RoPE frequencies as a registered buffer so they
        # follow the module to the correct device via .to(device)
        if config.use_rope:
            head_dim = config.decoder_hidden_dim // config.decoder_num_heads
            self.register_buffer(
                "_rope_freqs",
                precompute_rope_freqs(head_dim, config.max_output_len + 1),
                persistent=False,
            )
        else:
            self._rope_freqs: Tensor | None = None

        # -- Precomputed causal mask (lazy, created on first decode) --
        self._full_causal_mask: Tensor | None = None

        # -- Post-hoc VQ codebook for hybrid agent interface --
        self._vq_codebook: nn.Module | None = None
        if config.vq_codebook_path is not None:
            self._load_vq_codebook(config.vq_codebook_path)

        # -- Tokenizer (lazy, loaded from spm_model_path) --
        self._tokenizer: SubwordTokenizer | None = None
        if config.spm_model_path is not None:
            self._tokenizer = SubwordTokenizer(config.spm_model_path, config.vocab_size)

        # -- Latent calibration statistics --
        # Registered as buffers so they follow .to(device) and are
        # saved/loaded with state_dict.  Populated during pretraining
        # to record the empirical z distribution.  At agent time,
        # calibrate_z() uses these to transform projected z values
        # into the distribution the decoder was trained on, ensuring
        # proper EOS behavior and variable-length output.
        self.register_buffer(
            "_z_mean", torch.zeros(config.latent_dim), persistent=True,
        )
        self.register_buffer(
            "_z_std", torch.ones(config.latent_dim), persistent=True,
        )
        self._z_stats_initialized = False

        # -- Agent z running statistics (for calibration) --
        self._agent_z_mean: Tensor | None = None
        self._agent_z_std: Tensor | None = None

        # -- Forward cache for extra_losses --
        self._cached_mu: Tensor | None = None
        self._cached_logvar: Tensor | None = None

        # -- Mutable temperature --
        self._current_temperature = config.temperature

        # -- Load pretrained decoder if configured --
        if config.pretrained_decoder_path is not None:
            self._load_pretrained_decoder(config.pretrained_decoder_path)

        # -- Freeze decoder if configured --
        if config.freeze_decoder:
            self._freeze_decoder_params()

    # ------------------------------------------------------------------
    # Latent space calibration
    # ------------------------------------------------------------------

    def update_z_stats(self, z: Tensor, momentum: float = 0.01) -> None:
        """Update running z statistics from a batch during pretraining.

        Called by the pretraining loop to track the empirical distribution
        of z values produced by the VAE encoder.  The statistics are saved
        with the decoder checkpoint and used at agent time by
        :meth:`calibrate_z` to keep projected z values in-distribution.

        Args:
            z: Latent codes ``(batch, latent_dim)`` from the encoder.
            momentum: Exponential moving average weight for updates.
        """
        with torch.no_grad():
            batch_mean = z.mean(dim=0)
            batch_std = z.std(dim=0).clamp(min=1e-6)

            if not self._z_stats_initialized:
                self._z_mean.copy_(batch_mean)
                self._z_std.copy_(batch_std)
                self._z_stats_initialized = True
            else:
                self._z_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self._z_std.mul_(1 - momentum).add_(batch_std, alpha=momentum)

    def calibrate_z(self, z: Tensor) -> Tensor:
        """Transform z to match the pretrained encoder's distribution.

        Rescales each z vector so its L2 norm matches the typical norm
        from pretraining, then shifts to the encoder's mean.  This is a
        stable, per-sample operation that doesn't depend on batch
        statistics or running averages — no moving target for REINFORCE.

        Preserves the *direction* of z (semantic content) while ensuring
        the decoder sees values in the magnitude range it was trained on.

        Args:
            z: Latent codes ``(batch, latent_dim)``.

        Returns:
            Calibrated z with the pretrained distribution's statistics.
        """
        # Target norm: typical L2 norm of encoder z during pretraining
        target_norm = (self._z_mean.norm() ** 2 + (self._z_std ** 2).sum()).sqrt()

        # Center z around origin, rescale to target norm, shift to encoder mean
        z_centered = z - z.mean(dim=0, keepdim=True)
        z_norm = z_centered.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_scaled = z_centered * (target_norm / z_norm)
        return z_scaled + self._z_mean

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_input_proj(self, input_dim: int) -> None:
        """Initialize residual input projection on first forward call.

        Creates a linear path and an MLP refinement path::

            h = linear(x) + mlp(x)
            mu, logvar = h.chunk(2, dim=-1)

        The linear path gives clean gradient signal for REINFORCE from
        step 0 (no vanishing through nonlinearities).  The MLP path
        learns nonlinear corrections as training progresses.  Together
        they provide more expressive capacity than either alone.
        """
        if self._input_proj is not None:
            return

        device = next(self.decoder.parameters()).device
        out_dim = self._latent_dim * 2
        self._input_proj = nn.Linear(input_dim, out_dim).to(device)
        self._input_refine = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        ).to(device)
        # Initialize refine path near zero so it starts as a no-op,
        # letting the linear path dominate early training.
        with torch.no_grad():
            self._input_refine[-1].weight.mul_(0.01)
            self._input_refine[-1].bias.zero_()

        if self._pooling == "attention":
            self._attention_pool_query = nn.Parameter(
                torch.randn(1, 1, input_dim, device=device) * 0.02
            )

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def _load_vq_codebook(self, path: str) -> None:
        """Load a post-hoc VQ codebook fitted to the encoder z distribution.

        The codebook provides discrete anchor points over the continuous
        latent manifold.  At agent time, z is decomposed into the nearest
        VQ code (in-distribution anchor) plus a continuous residual.
        """
        from pathlib import Path as _Path

        from lfm.generator.quantize import GroupedVQ

        if not _Path(path).exists():
            logger.warning("VQ codebook not found: %s — skipping", path)
            return

        cb = torch.load(path, map_location="cpu", weights_only=False)
        self._vq_codebook = GroupedVQ(
            num_groups=cb["num_groups"],
            codebook_size=cb["codebook_size"],
            embedding_dim=cb["embedding_dim"],
        )
        self._vq_codebook.load_state_dict(cb["vq_state_dict"])
        self._vq_codebook.eval()
        logger.info(
            "Loaded VQ codebook: %d groups × %d codes (qe=%.6f, util=%.0f%%)",
            cb["num_groups"], cb["codebook_size"],
            cb.get("final_qe", 0), cb.get("final_util", 0) * 100,
        )

    def _quantize_z(self, z: Tensor) -> Tensor:
        """Decompose z into VQ anchor + scaled residual.

        Returns z_quantized + alpha * (z - z_quantized), where alpha
        controls how much continuous detail passes through.  At alpha=0,
        output is pure codebook entries (fully discrete).  At alpha=1,
        output equals the original z (codebook has no effect).

        The VQ anchor ensures the decoder always sees a point that's
        in-distribution (a codebook entry ± small residual), solving
        the calibration problem for the agent game.
        """
        if self._vq_codebook is None:
            return z

        vq = self._vq_codebook.to(z.device)
        with torch.no_grad():
            z_q, _, _ = vq(z)

        alpha = self.config.vq_residual_alpha
        # Straight-through for the VQ part, continuous gradient for residual
        return z_q + alpha * (z - z_q)

    def _load_pretrained_decoder(self, path: str) -> None:
        """Load pretrained VAE decoder weights from a checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If checkpoint dimensions don't match config.
        """
        from pathlib import Path as _Path

        if not _Path(path).exists():
            raise FileNotFoundError(f"Pretrained decoder checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if checkpoint["latent_dim"] != self._latent_dim:
            raise ValueError(
                f"Checkpoint latent_dim={checkpoint['latent_dim']} does not match "
                f"config latent_dim={self._latent_dim}"
            )
        if checkpoint["vocab_size"] != self._vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={checkpoint['vocab_size']} does not match "
                f"config vocab_size={self._vocab_size}"
            )

        self.latent_to_decoder.load_state_dict(checkpoint["latent_to_decoder"])
        self.token_embedding.load_state_dict(checkpoint["token_embedding"])
        if "pos_embedding" in checkpoint and not isinstance(self.pos_embedding, nn.Identity):
            self.pos_embedding.load_state_dict(checkpoint["pos_embedding"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.output_head.load_state_dict(checkpoint["output_head"])

        # Load latent calibration statistics if present.
        # calibrate_z() rescales agent z to match the encoder's distribution,
        # preserving direction (semantics) while matching magnitude (EOS
        # behavior).  Only enabled when z_std is reasonably large; with
        # KL=0 pretraining (σ≈0.04), calibration destabilizes REINFORCE
        # training — the uncalibrated path produces fixed-length messages
        # but achieves higher accuracy.
        if "z_mean" in checkpoint and "z_std" in checkpoint:
            z_std_mean = checkpoint["z_std"].mean().item()
            if z_std_mean > 0.1:
                self._z_mean.copy_(checkpoint["z_mean"])
                self._z_std.copy_(checkpoint["z_std"])
                self._z_stats_initialized = True
                logger.info(
                    "Loaded z calibration stats: mean_norm=%.2f, mean_std=%.4f",
                    self._z_mean.norm().item(),
                    z_std_mean,
                )
            else:
                # Store stats for exploration tools (dim-sweep etc.)
                # but don't use them for calibration during agent training.
                self._z_mean.copy_(checkpoint["z_mean"])
                self._z_std.copy_(checkpoint["z_std"])
                logger.info(
                    "z calibration disabled (σ=%.4f, KL=0 pretraining) — "
                    "stats loaded for exploration only",
                    z_std_mean,
                )

        logger.info("Loaded pretrained VAE decoder from %s", path)

    def _freeze_decoder_params(self) -> None:
        """Freeze all decoder parameters (preserves linguistic prior)."""
        frozen_modules: list[nn.Module] = [
            self.latent_to_decoder,
            self.token_embedding,
            self.pos_embedding,
            self.decoder,
            self.output_head,
        ]
        if self._vq_codebook is not None:
            frozen_modules.append(self._vq_codebook)
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        """Pool variable-length input embeddings to a single vector per sample.

        Args:
            embeddings: ``(batch, seq_len, dim)``.
            mask: ``(batch, seq_len)`` boolean.

        Returns:
            ``(batch, dim)``.
        """
        if self._pooling == "attention" and self._attention_pool_query is not None:
            query = self._attention_pool_query.expand(embeddings.size(0), -1, -1)
            attn_weights = torch.bmm(query, embeddings.transpose(1, 2))  # (B, 1, S)
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.bmm(attn_weights, embeddings).squeeze(1)  # (B, dim)

        # Default: masked mean pooling
        return masked_mean(embeddings, mask, dim=1)

    # ------------------------------------------------------------------
    # Reparameterization
    # ------------------------------------------------------------------

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor, training: bool) -> Tensor:
        """Sample z via reparameterization trick.

        Deterministic (returns mu) at eval time.
        """
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    # ------------------------------------------------------------------
    # Autoregressive decoding
    # ------------------------------------------------------------------

    def _decode(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Autoregressive decoding from latent z with KV caching.

        Uses incremental decoding: each step only computes Q/K/V for the
        new token and reuses cached K/V from all previous steps.

        Args:
            z: ``(batch, latent_dim)``.

        Returns:
            Tuple of ``(token_ids, token_probs, decoder_states, lengths, mask)``.
        """
        from lfm.generator.layers import LinguisticDecoder, multiscale_causal_mask

        batch = z.size(0)
        device = z.device
        max_len = self._max_output_len

        # Hybrid VQ: decompose into discrete anchor + continuous residual.
        # This replaces calibrate_z when a codebook is available — the VQ
        # anchor is always in-distribution by construction.
        if self._vq_codebook is not None:
            z = self._quantize_z(z)
        elif self._z_stats_initialized:
            z = self.calibrate_z(z)

        memory = self.latent_to_decoder(z).unsqueeze(1)

        # Precompute the full causal mask once (reuse rows per step)
        if self._full_causal_mask is None or self._full_causal_mask.size(1) < max_len + 1:
            self._full_causal_mask = multiscale_causal_mask(
                max_len + 1,
                num_heads=self.config.decoder_num_heads,
                head_windows=self.config.attention_head_windows,
                global_every=self.config.attention_global_every,
                device=device,
            )

        use_cache = isinstance(self.decoder, LinguisticDecoder)

        bos_ids = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        bos_embed = self.token_embedding(bos_ids)

        all_probs: list[Tensor] = []
        all_states: list[Tensor] = []

        if use_cache:
            kv_cache = self.decoder.make_kv_cache(
                batch, max_len + 1, device, dtype=memory.dtype,
            )

            # Prime cache with BOS
            mask_row = self._full_causal_mask[:, 0:1, 0:1]
            out = self.decoder.forward_cached(
                bos_embed, memory, kv_cache,
                rope_freqs=self._rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()

            for t in range(max_len):
                last_hidden = out  # (B, 1, H)
                logits_t = self.output_head(last_hidden).squeeze(1)

                all_states.append(last_hidden)

                eos_boost = 2.0 * (t / max(max_len - 1, 1))
                logits_t[:, self.eos_id] += eos_boost

                probs_t = gumbel_softmax(
                    logits_t,
                    tau=self._current_temperature,
                    hard=self._hard_sample,
                )
                all_probs.append(probs_t.unsqueeze(1))

                next_embed = (probs_t @ self.token_embedding.weight).unsqueeze(1)
                seq_so_far = kv_cache.seq_len + 1
                mask_row = self._full_causal_mask[
                    :, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far
                ]
                out = self.decoder.forward_cached(
                    next_embed, memory, kv_cache,
                    rope_freqs=self._rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
        else:
            # Fallback: no KV cache
            generated_embeds = bos_embed

            for t in range(max_len):
                seq_len = generated_embeds.size(1)
                decoder_input = generated_embeds
                if not isinstance(self.pos_embedding, nn.Identity):
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    decoder_input = decoder_input + self.pos_embedding(pos_ids)

                tgt_mask = self._full_causal_mask[:, :seq_len, :seq_len]
                decoder_out = self.decoder(
                    decoder_input, memory,
                    tgt_mask=tgt_mask, rope_freqs=self._rope_freqs,
                )

                last_hidden = decoder_out[:, -1:, :]
                logits_t = self.output_head(last_hidden).squeeze(1)

                all_states.append(last_hidden)

                eos_boost = 2.0 * (t / max(max_len - 1, 1))
                logits_t[:, self.eos_id] += eos_boost

                probs_t = gumbel_softmax(
                    logits_t,
                    tau=self._current_temperature,
                    hard=self._hard_sample,
                )
                all_probs.append(probs_t.unsqueeze(1))

                next_embed = probs_t @ self.token_embedding.weight
                generated_embeds = torch.cat(
                    [generated_embeds, next_embed.unsqueeze(1)], dim=1
                )

        token_probs = torch.cat(all_probs, dim=1)
        token_ids = token_probs.argmax(dim=-1)
        decoder_states = torch.cat(all_states, dim=1)

        eos_mask = token_ids == self.eos_id
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.float().argmax(dim=1)

        lengths = torch.where(
            has_eos,
            first_eos + 1,
            torch.full_like(first_eos, max_len),
        )
        output_mask = create_padding_mask(lengths, max_len)

        return token_ids, token_probs, decoder_states, lengths, output_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
    ) -> dict[str, Tensor]:
        """Generate subword token sequences from input embeddings.

        1. Pool input embeddings to a single vector per sample.
        2. Project to VAE latent distribution (μ, σ).
        3. Sample z via reparameterization.
        4. Decode z autoregressively through the (frozen) transformer.

        Args:
            embeddings: Dense input embeddings ``(batch, seq_len, dim)``.
            mask: Boolean mask ``(batch, seq_len)``.

        Returns:
            Dictionary with ``tokens``, ``token_probs``, ``embeddings``,
            ``lengths``, ``mask``, ``mu``, and ``logvar``.
        """
        self._ensure_input_proj(embeddings.size(-1))
        assert self._input_proj is not None

        # 1. Pool
        pooled = self._pool(embeddings, mask)  # (B, input_dim)

        # 2. Residual projection to mu, logvar
        h = self._input_proj(pooled) + self._input_refine(pooled)
        mu, logvar = h.chunk(2, dim=-1)  # each (B, latent_dim)

        # 3. Reparameterize
        z = self._reparameterize(mu, logvar, self.training)

        # 4. Cache for extra_losses
        self._cached_mu = mu
        self._cached_logvar = logvar

        # 5. Decode or use latent directly
        # When the decoder is frozen and we're training, skip the expensive
        # AR decode loop — use the latent-to-decoder projection directly.
        # This gives clean gradient flow to _input_proj without vanishing
        # through 64 frozen AR steps.  Full AR decode is used at eval time.
        if self.config.freeze_decoder and self.training:
            # Latent bottleneck path: use z directly as the message
            # representation. The decoder is for generating readable text
            # at eval time — the information bottleneck IS the latent
            # space itself (256-dim z), not the decoded token sequence.
            # The game's MLP decoder reconstructs directly from z.
            batch = z.size(0)
            return {
                "tokens": torch.zeros(
                    batch, 1, dtype=torch.long, device=z.device
                ),
                "token_probs": torch.zeros(
                    batch, 1, self._full_vocab, device=z.device
                ),
                "embeddings": z.unsqueeze(1),  # (B, 1, latent_dim)
                "lengths": torch.ones(batch, device=z.device),
                "mask": torch.ones(
                    batch, 1, dtype=torch.bool, device=z.device
                ),
                "mu": mu,
                "logvar": logvar,
            }

        # Full AR decode (eval time or unfrozen training)
        token_ids, token_probs, decoder_states, lengths, output_mask = self._decode(z)

        return {
            "tokens": token_ids,
            "token_probs": token_probs,
            "embeddings": decoder_states,
            "lengths": lengths,
            "mask": output_mask,
            "mu": mu,
            "logvar": logvar,
        }

    # ------------------------------------------------------------------
    # Text decoding
    # ------------------------------------------------------------------

    def decode_to_text(self, token_ids: Tensor) -> list[str]:
        """Decode integer token ids to human-readable strings.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            List of decoded strings, one per batch element.

        Raises:
            RuntimeError: If no sentencepiece model was configured.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Cannot decode to text: no spm_model_path configured in GeneratorConfig."
            )
        return self._tokenizer.batch_decode(token_ids)

    # ------------------------------------------------------------------
    # Intrinsic losses
    # ------------------------------------------------------------------

    def extra_losses(self) -> dict[str, Tensor]:
        """Return KL divergence loss toward standard normal prior.

        The KL term regularizes the input projection to produce latent
        distributions close to N(0, I), preserving the structure of the
        pretrained VAE's latent space.  Free bits clamping prevents
        posterior collapse with powerful autoregressive decoders.
        """
        losses: dict[str, Tensor] = {}

        if self._cached_mu is None or self._cached_logvar is None:
            return losses

        mu = self._cached_mu
        logvar = self._cached_logvar

        # Per-dimension KL: 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (B, latent_dim)

        # Free bits: clamp per-dimension KL to a minimum
        if self._kl_free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self._kl_free_bits)

        kl_loss = kl_per_dim.sum(dim=-1).mean()  # scalar
        losses["kl_divergence"] = kl_loss * self._kl_weight

        return losses

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def anneal_temperature(self, step: int) -> None:
        """Update Gumbel-Softmax temperature based on training step.

        Linearly anneals from ``config.temperature`` to
        ``config.temperature_min`` over ``config.temperature_anneal_steps``.

        Args:
            step: Current training step.
        """
        cfg = self.config
        if cfg.temperature_anneal_steps <= 0:
            return
        frac = min(step / cfg.temperature_anneal_steps, 1.0)
        self._current_temperature = (
            cfg.temperature + frac * (cfg.temperature_min - cfg.temperature)
        )
