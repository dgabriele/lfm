"""SynthLM: alien language model with optional embedding conditioning.

Phase 1 — alien LM:
  Sequence: [a0, a1, ..., aN-1, EOS]
  Input:  alien_ids[:, :-1]  →  hidden  →  predicts alien_ids[:, 1:]
  Trainable: backend._alien_emb, backend._alien_head, body.

Phase 2 — embedding conditioning:
  Sequence: [prefix(n) | a0, a1, ..., aN-1, EOS]
  Input:  [prefix | alien_ids[:, :-1]]
  hidden[:, n-1:]  →  predicts alien_ids (all T tokens)
  Last prefix position predicts a0; each alien position predicts the next.
  Trainable: PrefixProjector, LengthHead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.synth.backend import DecoderBackend
from lfm.synth.config import SynthConfig


class PrefixProjector(nn.Module):
    """Maps a source representation to n_prefix decoder input embeddings.

    Supports two source formats:
      - flat (B, source_dim): legacy mean-pool; projected via single MLP
      - multi-position (B, n_source, source_dim): last_k_concat-style; each
        source slot is projected with shared weights, preserving per-position
        structure end-to-end (no compression bottleneck).
    """

    def __init__(self, source_dim: int, d_model: int, n_prefix: int) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.source_dim = source_dim
        self.d_model = d_model
        # Shared per-position MLP — handles both flat and multi-position inputs.
        self.proj = nn.Sequential(
            nn.Linear(source_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # When source is flat (mean-pool legacy), expand to n_prefix tokens.
        self.expand = nn.Linear(d_model, d_model * n_prefix)

    def forward(self, source: Tensor) -> Tensor:
        """source: (B, source_dim) or (B, n_source, source_dim) → (B, n_prefix, d_model).

        Multi-position path requires n_source == n_prefix; the per-position MLP
        is shared across slots and the source structure passes through directly.
        """
        if source.dim() == 3:
            B, n_source, _ = source.shape
            if n_source != self.n_prefix:
                raise ValueError(
                    f"multi-position source has n_source={n_source} but n_prefix={self.n_prefix}; "
                    f"these must match for the per-position projection."
                )
            return self.proj(source)  # (B, n_prefix, d_model)
        # Flat fallback (legacy mean-pool source).
        x = self.proj(source)         # (B, d_model)
        return self.expand(x).view(source.size(0), self.n_prefix, self.d_model)


class LengthHead(nn.Module):
    """Predicts alien token count from source embedding."""

    def __init__(self, source_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(source_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """(B, source_dim) → (B,) scalar predicted length"""
        return self.mlp(embedding).squeeze(-1).clamp(min=4.0)


class SynthLM(nn.Module):
    """Decoder-only alien LM: causal LM on alien cipher text + prefix conditioning."""

    def __init__(self, backend: DecoderBackend, config: SynthConfig) -> None:
        super().__init__()
        self.backend = backend
        self.config = config
        d = backend.d_model
        self.projector = PrefixProjector(config.source_embedding_dim, d, config.n_prefix_tokens)
        self.length_head = LengthHead(config.source_embedding_dim)
        # Phase 1 coherence (RTD) discriminator: per-position "is this token a
        # corruption?" binary classifier. Trained jointly with LM CE so the body
        # learns within-sentence coherence features alongside next-token statistics.
        self.coherence_head = nn.Linear(d, 1).to(backend.dtype)

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def forward_phase1(
        self,
        alien_ids: Tensor,
        alien_labels: Tensor,
        replace_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Causal LM on alien token sequence with optional coherence (RTD) loss.

        Args:
            alien_ids: (B, T) input ids — may contain corruption (random tokens
                substituted at some positions). Length T includes the trailing
                target position.
            alien_labels: (B, T) clean labels with -100 at pad positions. Used as
                the LM target for predict-next-token.
            replace_mask: optional (B, T-1) {0,1} float mask matching alien_ids[:, :-1];
                1.0 at positions where the input token was replaced. When provided,
                the coherence head is trained to detect corruption.

        Returns:
            ce_loss:  cross-entropy over non-pad positions.
            mse_loss: MSE between current and frozen reference body hidden states
                      (zero tensor if no reference body was initialised).
            coh_loss: BCE on per-position corruption detection (zero tensor if no
                      replace_mask provided).
        """
        inputs_embeds = self.backend.embed_alien(alien_ids[:, :-1])
        hidden = self.backend.forward_hidden(inputs_embeds)
        logits = self.backend.alien_logits(hidden)  # (B, T-1, V)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), alien_labels[:, 1:].reshape(-1), ignore_index=-100
        )
        if self.backend.has_reference:
            ref_hidden = self.backend.reference_hidden(inputs_embeds)
            mse_loss = F.mse_loss(hidden, ref_hidden)
        else:
            mse_loss = ce_loss.new_zeros(())
        if replace_mask is not None:
            coh_logits = self.coherence_head(hidden).squeeze(-1)  # (B, T-1)
            valid = (alien_labels[:, 1:] != -100).float()
            coh_loss = F.binary_cross_entropy_with_logits(
                coh_logits.float(), replace_mask.float(),
                weight=valid, reduction="sum",
            ) / valid.sum().clamp(min=1)
        else:
            coh_loss = ce_loss.new_zeros(())
        return ce_loss, mse_loss, coh_loss

    def phase1_logits(self, alien_ids: Tensor) -> Tensor:
        """Teacher-forced alien logits for diagnostics. (B, T-1, V)"""
        hidden = self.backend.forward_hidden(self.backend.embed_alien(alien_ids[:, :-1]))
        return self.backend.alien_logits(hidden)

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def forward_phase2(
        self,
        source_embedding: Tensor,
        alien_ids: Tensor,
        alien_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Conditioned causal LM. Returns (lm_loss, length_loss).

        [prefix(n) | alien_ids[:, :-1]] → hidden[:, n-1:] predicts alien_labels.
        """
        n = self.config.n_prefix_tokens
        prefix = self.projector(source_embedding).to(self.backend.dtype)
        embeds = torch.cat([prefix, self.backend.embed_alien(alien_ids[:, :-1])], dim=1)
        hidden = self.backend.forward_hidden(embeds)           # (B, n+T-1, D)
        logits = self.backend.alien_logits(hidden[:, n - 1:])  # (B, T, V)
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), alien_labels.reshape(-1), ignore_index=-100
        )
        norm = float(alien_ids.size(1))
        length_loss = F.mse_loss(
            self.length_head(source_embedding) / norm,
            (alien_labels != -100).float().sum(dim=-1) / norm,
        )
        return lm_loss, length_loss

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        source_embedding: Tensor,
        eos_id: int,
        pad_id: int = 0,
        max_length: int | None = None,
    ) -> Tensor:
        """Greedy alien generation conditioned on source embedding (Phase 2)."""
        B, device = source_embedding.size(0), source_embedding.device
        if max_length is None:
            max_length = int(self.length_head(source_embedding).round().long().max()) + self.config.length_slack

        context = self.projector(source_embedding).to(self.backend.dtype)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        output: list[Tensor] = []

        for _ in range(max_length):
            hidden = self.backend.forward_hidden(context)
            next_ids = self.backend.alien_logits(hidden[:, -1:]).squeeze(1).argmax(-1)
            next_ids = next_ids.masked_fill(done, pad_id)
            output.append(next_ids)
            done |= next_ids == eos_id
            if done.all():
                break
            context = torch.cat([context, self.backend.embed_alien(next_ids.unsqueeze(1))], dim=1)

        return torch.stack(output, dim=1) if output else torch.zeros(B, 0, dtype=torch.long, device=device)

    # ── Persistence ───────────────────────────────────────────────────────────

    def phase1_state(self) -> dict:
        return {
            "alien_emb": self.backend._alien_emb.state_dict(),
            "alien_head": self.backend._alien_head.state_dict(),
            "body":       self.backend._body.state_dict(),
        }

    def load_phase1_state(self, state: dict) -> None:
        self.backend._alien_emb.load_state_dict(state["alien_emb"])
        self.backend._alien_head.load_state_dict(state["alien_head"])
        self.backend._body.load_state_dict(state["body"])

    def phase2_state(self) -> dict:
        return {"projector": self.projector.state_dict(), "length_head": self.length_head.state_dict()}

    def load_phase2_state(self, state: dict) -> None:
        self.projector.load_state_dict(state["projector"])
        self.length_head.load_state_dict(state["length_head"])

    def save_phase1(self, path: str) -> None:
        torch.save(self.phase1_state(), path)

    def load_phase1(self, path: str) -> None:
        self.load_phase1_state(torch.load(path, map_location="cpu", weights_only=True))

    def save_phase2(self, path: str) -> None:
        torch.save(self.phase2_state(), path)

    def load_phase2(self, path: str) -> None:
        self.load_phase2_state(torch.load(path, map_location="cpu", weights_only=True))
