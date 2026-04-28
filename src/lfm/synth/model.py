"""SynthLM: decoder-only LLM with alien cipher head and prefix conditioning.

Phase 1 — CipherTrainer:
  Sequence: [native_tokens | eos_sep | alien_tokens]
  Built with inputs_embeds; causal LM predicts each alien token from context.
  Trainable: backend._alien_emb, backend._alien_head.
  Body: frozen throughout.

Phase 2 — ConditioningTrainer:
  Sequence: [prefix_embeds(n_prefix) | alien_tokens]
  Prefix comes from PrefixProjector(source_embedding); no native text.
  Trainable: PrefixProjector, LengthHead.
  Body + alien emb/head: frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.synth.backend import DecoderBackend
from lfm.synth.config import SynthConfig


class PrefixProjector(nn.Module):
    """Maps a flat source embedding to n_prefix decoder input embeddings."""

    def __init__(self, source_dim: int, d_model: int, n_prefix: int) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.proj = nn.Sequential(
            nn.Linear(source_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * n_prefix),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """(B, source_dim) → (B, n_prefix, d_model)."""
        B = embedding.size(0)
        return self.proj(embedding).view(B, self.n_prefix, -1)


class LengthHead(nn.Module):
    """Predicts alien token count from source embedding."""

    def __init__(self, source_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """(B, source_dim) → (B,) scalar predicted length."""
        return self.mlp(embedding).squeeze(-1).clamp(min=4.0)


class SynthLM(nn.Module):
    """Decoder-only SynthLM: frozen causal LM + alien cipher sub-vocab + prefix projector."""

    def __init__(self, backend: DecoderBackend, config: SynthConfig) -> None:
        super().__init__()
        self.backend = backend
        self.config = config
        d = backend.d_model
        self.projector = PrefixProjector(config.source_embedding_dim, d, config.n_prefix_tokens)
        self.length_head = LengthHead(config.source_embedding_dim)

    # ---- Shared helpers ----

    def _phase1_inputs(
        self,
        native_ids: Tensor,    # (B, T_native)
        native_mask: Tensor,   # (B, T_native)
        alien_labels: Tensor,  # (B, T_alien)
    ) -> tuple[Tensor, Tensor, int]:
        """Build inputs_embeds + attention_mask for Phase 1 forward.

        Returns (inputs_embeds, attention_mask, alien_start_pos) where
        alien_start_pos is the position whose hidden state predicts alien_labels[:, 0].
        """
        B, T_native = native_ids.shape
        T_alien = alien_labels.size(1)
        device = native_ids.device

        native_embs = self.backend.embed_native(native_ids)
        eos_ids = torch.full((B, 1), self.backend.eos_token_id, device=device, dtype=torch.long)
        eos_embs = self.backend.embed_native(eos_ids)
        # -100 pads → clamp to 0 (arbitrary valid index; loss ignores these positions)
        alien_embs = self.backend.embed_alien(alien_labels.clamp(min=0))

        inputs_embeds = torch.cat([native_embs, eos_embs, alien_embs], dim=1)
        attention_mask = torch.cat([
            native_mask,
            torch.ones(B, 1 + T_alien, device=device, dtype=native_mask.dtype),
        ], dim=1)
        # EOS is at position T_native; its output predicts alien_labels[:, 0]
        return inputs_embeds, attention_mask, T_native

    def _alien_ce_loss(
        self,
        inputs_embeds: Tensor,
        alien_start_pos: int,
        alien_labels: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """CE loss at alien positions.

        hidden[alien_start_pos + i] predicts alien_labels[:, i] for i in 0..T_alien-1.
        (Causal LM: position i's output predicts the next token, i.e. alien_labels[:, i].)
        """
        hidden = self.backend.forward_hidden(inputs_embeds, attention_mask)
        T_alien = alien_labels.size(1)
        alien_hidden = hidden[:, alien_start_pos:alien_start_pos + T_alien, :]
        logits = self.backend.alien_logits(alien_hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            alien_labels.reshape(-1),
            ignore_index=-100,
        )

    # ---- Phase 1 ----

    def forward_phase1(
        self,
        native_ids: Tensor,
        native_mask: Tensor,
        alien_labels: Tensor,
    ) -> Tensor:
        """CE loss: English → alien tokens via frozen causal LM."""
        inputs_embeds, attn_mask, alien_start = self._phase1_inputs(
            native_ids, native_mask, alien_labels
        )
        return self._alien_ce_loss(inputs_embeds, alien_start, alien_labels, attn_mask)

    def alien_logits_phase1(
        self,
        native_ids: Tensor,
        native_mask: Tensor,
        alien_labels: Tensor,
    ) -> Tensor:
        """Teacher-forced alien logits (B, T_alien, alien_vocab) — for diagnostics."""
        inputs_embeds, attn_mask, alien_start = self._phase1_inputs(
            native_ids, native_mask, alien_labels
        )
        hidden = self.backend.forward_hidden(inputs_embeds, attn_mask)
        T_alien = alien_labels.size(1)
        alien_hidden = hidden[:, alien_start:alien_start + T_alien, :]
        return self.backend.alien_logits(alien_hidden)

    # ---- Phase 2 ----

    def forward_phase2(
        self,
        source_embedding: Tensor,
        alien_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """CE + length loss: source embedding → alien tokens via prefix.

        Sequence: [prefix(n_prefix) | alien_tokens]
        The last prefix position predicts alien_labels[:, 0].
        Returns (lm_loss, length_loss).
        """
        n_prefix = self.config.n_prefix_tokens

        # Cast projector output to body dtype (e.g. float32 projector → bfloat16 body)
        prefix_embs = self.projector(source_embedding).to(self.backend.dtype)
        alien_embs = self.backend.embed_alien(alien_labels.clamp(min=0))
        inputs_embeds = torch.cat([prefix_embs, alien_embs], dim=1)

        # Position n_prefix-1 (last prefix) → predicts alien_labels[:, 0]
        lm_loss = self._alien_ce_loss(inputs_embeds, n_prefix - 1, alien_labels)

        length_pred = self.length_head(source_embedding)
        norm = float(alien_labels.size(1))
        length_target = (alien_labels != -100).float().sum(dim=-1) / norm
        length_loss = F.mse_loss(length_pred / norm, length_target)

        return lm_loss, length_loss

    # ---- Inference ----

    @torch.no_grad()
    def generate(
        self,
        source_embedding: Tensor,
        alien_stop_id: int,
        alien_pad_id: int = 0,
        max_length: int | None = None,
    ) -> Tensor:
        """Greedy decode alien tokens from source embedding (Phase 2 inference).

        Args:
            source_embedding: (B, source_dim).
            alien_stop_id: Token ID that signals end of sequence.
            alien_pad_id: Padding ID written after stop (for uniform tensor shape).
            max_length: Hard cap. If None, predicted from length_head + length_slack.

        Returns:
            (B, T_generated) token ID tensor.
        """
        B = source_embedding.size(0)
        device = source_embedding.device

        if max_length is None:
            pred_len = self.length_head(source_embedding).round().long()
            max_length = int(pred_len.max().item()) + self.config.length_slack

        context = self.projector(source_embedding).to(self.backend.dtype)  # (B, n_prefix, D)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        output_ids: list[Tensor] = []

        for _ in range(max_length):
            hidden = self.backend.forward_hidden(context)
            logits = self.backend.alien_logits(hidden[:, -1:, :])  # (B, 1, V)
            next_ids = logits[:, 0, :].argmax(dim=-1)              # (B,)
            next_ids = next_ids.masked_fill(done, alien_pad_id)
            output_ids.append(next_ids)

            done = done | (next_ids == alien_stop_id)
            if done.all():
                break

            next_embs = self.backend.embed_alien(next_ids.unsqueeze(1))
            context = torch.cat([context, next_embs], dim=1)

        if not output_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.stack(output_ids, dim=1)  # (B, T_gen)

    @torch.no_grad()
    def generate_phase1(
        self,
        native_ids: Tensor,
        native_mask: Tensor,
        alien_stop_id: int,
        alien_pad_id: int = 0,
        max_length: int = 64,
    ) -> Tensor:
        """Greedy decode alien tokens from English input (Phase 1 diagnostics)."""
        B = native_ids.size(0)
        device = native_ids.device

        native_embs = self.backend.embed_native(native_ids)
        eos_ids = torch.full((B, 1), self.backend.eos_token_id, device=device, dtype=torch.long)
        eos_embs = self.backend.embed_native(eos_ids)
        context = torch.cat([native_embs, eos_embs], dim=1)  # (B, T_native+1, D)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        output_ids: list[Tensor] = []

        for _ in range(max_length):
            hidden = self.backend.forward_hidden(context)
            logits = self.backend.alien_logits(hidden[:, -1:, :])
            next_ids = logits[:, 0, :].argmax(dim=-1)
            next_ids = next_ids.masked_fill(done, alien_pad_id)
            output_ids.append(next_ids)

            done = done | (next_ids == alien_stop_id)
            if done.all():
                break

            next_embs = self.backend.embed_alien(next_ids.unsqueeze(1))
            context = torch.cat([context, next_embs], dim=1)

        if not output_ids:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.stack(output_ids, dim=1)

    # ---- Persistence ----

    # ---- Persistence helpers ----

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
        return {
            "projector":   self.projector.state_dict(),
            "length_head": self.length_head.state_dict(),
        }

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
