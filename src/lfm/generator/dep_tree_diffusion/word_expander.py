"""Autoregressive word expander for hierarchical diffusion.

The diffusion decoder produces one latent vector per dependency role.
This module takes each role vector and autoregressively generates
the 1-N subword tokens that fill that role. This eliminates the
within-word degeneration problem: the diffusion handles across-word
structure while the AR head handles within-word coherence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WordExpander(nn.Module):
    """Small autoregressive decoder that expands role vectors into tokens.

    For each role, takes a conditioning vector from the diffusion decoder
    and generates subword tokens left-to-right until EOS or max length.

    Architecture: single-layer GRU with cross-attention to the role vector,
    followed by a token projection head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_hidden: int = 256,
        max_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.max_tokens = max_tokens

        self.token_embedding = nn.Embedding(vocab_size, d_hidden)
        self.role_proj = nn.Linear(d_model, d_hidden)
        self.gru = nn.GRUCell(d_hidden, d_hidden)
        self.output_head = nn.Linear(d_hidden, vocab_size)

    def forward(
        self,
        role_vectors: Tensor,
        target_tokens: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Training forward: teacher-forced AR over target tokens.

        Args:
            role_vectors: (B, R, d_model) one vector per role from diffusion.
            target_tokens: (B, R, max_T) ground truth tokens per role.
            target_lengths: (B, R) number of valid tokens per role.

        Returns:
            loss: scalar CE loss over all valid token positions.
        """
        b, r, _ = role_vectors.shape
        max_t = target_tokens.size(2)
        device = role_vectors.device

        # Project role vectors to GRU hidden dim
        h = self.role_proj(role_vectors)  # (B, R, d_hidden)
        h = h.reshape(b * r, self.d_hidden)  # (B*R, d_hidden)

        # Flatten targets
        targets_flat = target_tokens.reshape(b * r, max_t)  # (B*R, max_T)
        lengths_flat = target_lengths.reshape(b * r)  # (B*R,)

        # Teacher-forced AR: feed ground truth token at each step
        total_loss = torch.tensor(0.0, device=device)
        n_tokens = 0

        # BOS token = 0 (padding), first input is the role vector projection
        prev_emb = torch.zeros(b * r, self.d_hidden, device=device)

        for t in range(max_t):
            h = self.gru(prev_emb, h)
            logits = self.output_head(h)  # (B*R, vocab)

            # Mask: only positions where t < length
            valid = t < lengths_flat
            if valid.any():
                loss_t = F.cross_entropy(
                    logits[valid], targets_flat[valid, t],
                    reduction="sum",
                )
                total_loss = total_loss + loss_t
                n_tokens += valid.sum().item()

            # Teacher forcing: next input is ground truth
            if t < max_t - 1:
                prev_emb = self.token_embedding(targets_flat[:, t])

        return total_loss / max(n_tokens, 1)

    @torch.no_grad()
    def generate(
        self,
        role_vectors: Tensor,
        max_len: int | None = None,
    ) -> Tensor:
        """Generate tokens for each role autoregressively.

        Args:
            role_vectors: (B, R, d_model) from diffusion decoder.
            max_len: max tokens per role (default: self.max_tokens).

        Returns:
            tokens: (B, R, max_len) generated token IDs.
        """
        if max_len is None:
            max_len = self.max_tokens

        b, r, _ = role_vectors.shape
        device = role_vectors.device

        h = self.role_proj(role_vectors).reshape(b * r, self.d_hidden)
        prev_emb = torch.zeros(b * r, self.d_hidden, device=device)

        all_tokens = []
        for t in range(max_len):
            h = self.gru(prev_emb, h)
            logits = self.output_head(h)
            token_ids = logits.argmax(dim=-1)  # (B*R,)
            all_tokens.append(token_ids)
            prev_emb = self.token_embedding(token_ids)

        tokens = torch.stack(all_tokens, dim=1)  # (B*R, max_len)
        return tokens.reshape(b, r, max_len)
