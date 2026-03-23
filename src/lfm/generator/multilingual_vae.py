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
        self._input_proj: nn.Linear | None = None
        self._attention_pool_query: nn.Parameter | None = None

        # -- Decoder (linguistic architecture) --
        from lfm.generator.layers import (
            LinguisticDecoder,
            precompute_rope_freqs,
        )

        self.latent_norm = nn.LayerNorm(config.latent_dim)
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
                precompute_rope_freqs(head_dim, config.max_output_len),
                persistent=False,
            )
        else:
            self._rope_freqs: Tensor | None = None

        # -- Tokenizer (lazy, loaded from spm_model_path) --
        self._tokenizer: SubwordTokenizer | None = None
        if config.spm_model_path is not None:
            self._tokenizer = SubwordTokenizer(config.spm_model_path, config.vocab_size)

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
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_input_proj(self, input_dim: int) -> None:
        """Initialize input projection and pooling on first forward call."""
        if self._input_proj is not None:
            return

        device = next(self.decoder.parameters()).device
        self._input_proj = nn.Linear(input_dim, self._latent_dim * 2).to(device)

        if self._pooling == "attention":
            self._attention_pool_query = nn.Parameter(
                torch.randn(1, 1, input_dim, device=device) * 0.02
            )

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

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
        if "latent_norm" in checkpoint:
            self.latent_norm.load_state_dict(checkpoint["latent_norm"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.output_head.load_state_dict(checkpoint["output_head"])
        logger.info("Loaded pretrained VAE decoder from %s", path)

    def _freeze_decoder_params(self) -> None:
        """Freeze all decoder parameters (preserves linguistic prior)."""
        frozen_modules: list[nn.Module] = [
            self.latent_norm,
            self.latent_to_decoder,
            self.token_embedding,
            self.pos_embedding,
            self.decoder,
            self.output_head,
        ]
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
        """Autoregressive decoding from latent z.

        Args:
            z: ``(batch, latent_dim)``.

        Returns:
            Tuple of ``(token_ids, token_probs, decoder_states, lengths, mask)``.
        """
        batch = z.size(0)
        device = z.device
        max_len = self._max_output_len

        # Normalize z to distribution-invariant representation before
        # decoding.  This ensures the decoder sees the same scale/range
        # regardless of whether z came from the pretrained encoder or from
        # the agent's learned input projection.
        z = self.latent_norm(z)

        # z -> cross-attention memory: (batch, 1, decoder_hidden_dim)
        memory = self.latent_to_decoder(z).unsqueeze(1)

        # Start with BOS token
        bos_ids = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        generated_embeds = self.token_embedding(bos_ids)  # (B, 1, H)

        all_probs: list[Tensor] = []
        all_states: list[Tensor] = []

        for t in range(max_len):
            seq_len = generated_embeds.size(1)
            decoder_input = generated_embeds
            if not isinstance(self.pos_embedding, nn.Identity):
                pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                decoder_input = decoder_input + self.pos_embedding(pos_ids)

            # Multi-scale causal mask
            from lfm.generator.layers import multiscale_causal_mask

            tgt_mask = multiscale_causal_mask(
                seq_len,
                num_heads=self.config.decoder_num_heads,
                head_windows=self.config.attention_head_windows,
                global_every=self.config.attention_global_every,
                device=device,
            )

            decoder_out = self.decoder(
                decoder_input,
                memory,
                tgt_mask=tgt_mask,
                rope_freqs=self._rope_freqs,
            )  # (B, seq_len, H)

            last_hidden = decoder_out[:, -1:, :]  # (B, 1, H)
            logits_t = self.output_head(last_hidden).squeeze(1)  # (B, V)

            all_states.append(last_hidden)

            # Boost EOS probability as sequence gets longer — encourages
            # variable-length output.  Linear ramp: no boost at t=0,
            # +2.0 at max_len.
            eos_boost = 2.0 * (t / max(max_len - 1, 1))
            logits_t[:, self.eos_id] += eos_boost

            # Differentiable sampling via Gumbel-Softmax
            probs_t = gumbel_softmax(
                logits_t,
                tau=self._current_temperature,
                hard=self._hard_sample,
            )  # (B, V)
            all_probs.append(probs_t.unsqueeze(1))

            # Next input: soft probs @ embedding weight -> differentiable lookup
            next_embed = probs_t @ self.token_embedding.weight  # (B, H)
            generated_embeds = torch.cat(
                [generated_embeds, next_embed.unsqueeze(1)], dim=1
            )

        token_probs = torch.cat(all_probs, dim=1)  # (B, max_len, V)
        token_ids = token_probs.argmax(dim=-1)  # (B, max_len)
        decoder_states = torch.cat(all_states, dim=1)  # (B, max_len, H)

        # Compute lengths: position of first EOS token, or full
        # sequence length if no EOS was generated.
        eos_mask = token_ids == self.eos_id
        has_eos = eos_mask.any(dim=1)
        first_eos = eos_mask.float().argmax(dim=1)

        lengths = torch.where(
            has_eos,
            first_eos + 1,  # include the EOS token
            torch.full_like(first_eos, max_len),  # use full length
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

        # 2. Project to mu, logvar
        h = self._input_proj(pooled)  # (B, latent_dim * 2)
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
