"""AR p2g VAE (word-level).

IPA → (μ, logσ²) → z → English spelling, autoregressive char decoder.

The decoder is a standard causal transformer over the spelling chars;
z is prepended as a memory token that every position cross-attends to.
AR keeps the task tractable (each position coordinates with previous
ones) while z remains the sole bottleneck — interpolating z still yields
smooth trajectories through the manifold of real-word spellings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vocab import BOS_ID, EOS_ID, PAD_ID


@dataclass
class P2GConfig:
    input_vocab_size: int      # IPA chars + specials
    output_vocab_size: int     # spelling chars + specials
    latent_dim: int = 128
    encoder_dim: int = 256
    encoder_layers: int = 3
    encoder_heads: int = 4
    decoder_dim: int = 256
    decoder_layers: int = 3
    decoder_heads: int = 4
    max_ipa_len: int = 24
    max_spelling_len: int = 24
    dropout: float = 0.1
    # KL schedule
    kl_weight_max: float = 1.0
    kl_warmup_steps: int = 2000
    kl_free_bits: float = 0.5
    # Length loss
    length_weight: float = 0.5


class IPAEncoder(nn.Module):
    """IPA chars → (μ, logσ²) for a diagonal Gaussian posterior."""

    def __init__(self, cfg: P2GConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg.input_vocab_size, cfg.encoder_dim, padding_idx=PAD_ID,
        )
        self.pos_emb = nn.Embedding(cfg.max_ipa_len, cfg.encoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.encoder_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.encoder_layers)
        self.to_mu = nn.Linear(cfg.encoder_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.encoder_dim, cfg.latent_dim)
        self.max_len = cfg.max_ipa_len

    def forward(self, ipa_ids: Tensor) -> tuple[Tensor, Tensor]:
        # ipa_ids : (B, T)
        B, T = ipa_ids.shape
        pos = torch.arange(T, device=ipa_ids.device).clamp_max(self.max_len - 1)
        h = self.tok_emb(ipa_ids) + self.pos_emb(pos)[None, :, :]
        pad_mask = ipa_ids == PAD_ID
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # Mean-pool over non-pad positions
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.to_mu(pooled), self.to_logvar(pooled)


class SpellingDecoder(nn.Module):
    """AR char decoder conditioned on z via a prepended memory token.

    Input at training time: [<bos>, c0, c1, ..., c_{L-1}]; targets are
    [c0, c1, ..., c_{L-1}, <eos>].  A z-derived memory vector is
    prepended as position 0 so every query position cross-attends to z
    through the causal self-attention stack.
    """

    def __init__(self, cfg: P2GConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg.output_vocab_size, cfg.decoder_dim, padding_idx=PAD_ID,
        )
        # +1 slot so the z-memory token has its own position 0, and the
        # <bos>-led sequence starts at position 1.
        self.pos_emb = nn.Embedding(cfg.max_spelling_len + 2, cfg.decoder_dim)
        self.z_to_mem = nn.Linear(cfg.latent_dim, cfg.decoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.decoder_dim,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, cfg.decoder_layers)
        self.to_logits = nn.Linear(cfg.decoder_dim, cfg.output_vocab_size)
        self.max_len = cfg.max_spelling_len

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
        )

    def forward(self, z: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Teacher-forced forward. Returns logits over decoder_input_ids' positions.

        decoder_input_ids : (B, T) with T ≤ max_spelling_len + 1, starting with <bos>.
        """
        B, T = decoder_input_ids.shape
        tok = self.tok_emb(decoder_input_ids)
        mem = self.z_to_mem(z).unsqueeze(1)               # (B, 1, D)
        h = torch.cat([mem, tok], dim=1)                  # (B, T+1, D)
        pos = torch.arange(h.size(1), device=h.device)
        h = h + self.pos_emb(pos)[None]
        mask = self._causal_mask(h.size(1), h.device)
        h = self.decoder(h, mask=mask, is_causal=True)
        # Drop the z-memory position; keep logits for the token positions.
        return self.to_logits(h[:, 1:, :])

    @torch.no_grad()
    def generate(self, z: Tensor) -> list[list[int]]:
        """Greedy AR decode up to <eos> or max_spelling_len."""
        B = z.size(0)
        device = z.device
        cur = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outs: list[list[int]] = [[] for _ in range(B)]
        for _ in range(self.max_len):
            logits = self.forward(z, cur)[:, -1, :]       # (B, V)
            nxt = logits.argmax(-1)                       # (B,)
            for i in range(B):
                if finished[i]:
                    continue
                tok = int(nxt[i].item())
                if tok == EOS_ID:
                    finished[i] = True
                else:
                    outs[i].append(tok)
            if finished.all():
                break
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        return outs


class P2GVAE(nn.Module):
    def __init__(self, cfg: P2GConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = IPAEncoder(cfg)
        self.decoder = SpellingDecoder(cfg)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # Free bits applied PER LATENT DIM (Kingma 2016): clamp each dim's
        # KL to a minimum floor so the optimizer can't drive any dim to 0,
        # which is the posterior-collapse failure mode for text VAEs.
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = kl_per_dim.clamp_min(self.cfg.kl_free_bits)
        return kl_per_dim.sum(-1).mean()

    def forward(
        self,
        ipa_ids: Tensor,
        spelling_ids: Tensor,
        spelling_lens: Tensor,
        step: int,
    ) -> dict[str, Tensor]:
        mu, logvar = self.encoder(ipa_ids)
        z = self.reparameterize(mu, logvar)

        # Build decoder input/targets with teacher forcing.
        # spelling_ids: (B, L_spell) — left-aligned char ids, PAD-padded.
        # Prepend <bos>, append <eos> to each non-pad region:
        B, L = spelling_ids.shape
        bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=spelling_ids.device)
        dec_in = torch.cat([bos, spelling_ids], dim=1)    # (B, 1+L)
        # Targets: chars then <eos> at the spelling_lens position.
        eos_scatter = torch.full_like(spelling_ids, PAD_ID)
        # We place <eos> at index = spelling_lens (where the first pad sits).
        idx = spelling_lens.clamp_max(L - 1).unsqueeze(1)
        eos_scatter.scatter_(1, idx, EOS_ID)
        # Combine: non-pad chars from spelling_ids, PAD→EOS at length index,
        # remaining PAD beyond that.
        tgt = torch.where(spelling_ids == PAD_ID, eos_scatter, spelling_ids)
        # Causal shift: targets align with dec_in[1:] (predict c_t from <bos>..c_{t-1}).
        # Our decoder returns logits of shape (B, 1+L, V) for dec_in; slice
        # off the last position (no target beyond EOS) and align.
        logits = self.decoder(z, dec_in)                  # (B, 1+L, V)
        logits = logits[:, :L, :]                         # (B, L, V) — predict positions 0..L-1
        V = logits.size(-1)
        recon = F.cross_entropy(
            logits.reshape(-1, V),
            tgt.reshape(-1),
            ignore_index=PAD_ID,
            reduction="mean",
        )

        # KL with linear warmup.
        kl = self.kl_loss(mu, logvar)
        kl_w = min(1.0, step / max(1, self.cfg.kl_warmup_steps))
        kl_w = kl_w * self.cfg.kl_weight_max
        total = recon + kl_w * kl

        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = tgt != PAD_ID
            char_acc = ((pred == tgt) & mask).float().sum() / mask.float().sum().clamp_min(1)

        return {
            "loss": total,
            "recon": recon.detach(),
            "length_loss": torch.zeros((), device=total.device),  # kept for log compat
            "kl": kl.detach(),
            "kl_weight": torch.tensor(kl_w, device=total.device),
            "char_acc": char_acc,
            "len_acc": torch.zeros((), device=total.device),
            "mu": mu.detach(),
            "logvar": logvar.detach(),
        }

    @torch.no_grad()
    def generate(self, ipa_ids: Tensor, sample: bool = False) -> list[list[int]]:
        """Encode → z (μ or sampled) → AR greedy decode."""
        mu, logvar = self.encoder(ipa_ids)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decoder.generate(z)

    @torch.no_grad()
    def decode_from_z(self, z: Tensor) -> list[list[int]]:
        """Decode arbitrary z (for latent-space interpolation)."""
        return self.decoder.generate(z)
