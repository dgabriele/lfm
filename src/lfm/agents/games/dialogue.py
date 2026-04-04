"""Dialogue expression game — multi-turn self-play through the linguistic bottleneck.

A single agent with two roles (Observer and Analyst) generates multi-turn
conversations about input data.  Each turn is conditioned on the original
data embedding AND all previous turns via a context transformer.  The
receiver scores the full conversation for discrimination.

The resulting corpus has temporal coherence, referential consistency,
and progressive elaboration — discourse structure that enables an LLM
to learn the emergent language much more effectively than isolated
single-expression corpora.

Usage::

    poetry run lfm agent dialogue configs/dialogue_phase1.yaml
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, Receiver
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import ExpressionDecoder, rerun_decoder_multiseg_with_grad
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DialogueGameConfig(LFMBaseConfig):
    """Configuration for the dialogue expression game."""

    # Faculty
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 96
    vq_codebook_path: str | None = None

    # Dialogue structure
    num_turns: int = 4
    context_hidden_dim: int = 512
    context_heads: int = 4

    # Diffusion z-generator (shared across turns)
    z_hidden_dim: int = 512
    max_segments: int = 4
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    variable_segments: bool = True

    # Message encoder (reads full dialogue)
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Game
    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    # Training
    batch_size: int = 20
    gradient_accumulation_steps: int = 12
    steps: int = 4000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/dialogue_game"

    # Runtime
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        """Construct the LanguageFaculty config."""
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=1,
                vq_codebook_path=self.vq_codebook_path,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Context transformer
# ---------------------------------------------------------------------------


class ContextTransformer(nn.Module):
    """Merge data embedding, role, and dialogue history into conditioning.

    At each turn, produces a conditioning vector for the diffusion z-gen
    by cross-attending from (data + role) to the accumulated context
    of previous turns' hidden states.

    Args:
        embedding_dim: Input/output embedding dimension.
        hidden_dim: Internal cross-attention dimension.
        num_heads: Attention heads.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.data_proj = nn.Linear(embedding_dim, hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        data_embedding: Tensor,
        role_embedding: Tensor,
        context: Tensor | None,
    ) -> Tensor:
        """Produce conditioning for this turn's z-gen.

        Args:
            data_embedding: ``(B, embedding_dim)`` original input.
            role_embedding: ``(hidden_dim,)`` learned role vector.
            context: ``(B, T_prev, hidden_dim)`` accumulated hidden
                states from previous turns, or None for the first turn.

        Returns:
            ``(B, embedding_dim)`` conditioning vector.
        """
        query = self.data_proj(data_embedding).unsqueeze(1) + role_embedding

        if context is not None:
            context_normed = self.context_norm(context)
            attended, _ = self.context_attn(query, context_normed, context_normed)
            query = query + attended

        return self.out_proj(self.out_norm(query.squeeze(1)))


# ---------------------------------------------------------------------------
# Dialogue game
# ---------------------------------------------------------------------------


class DialogueGame(nn.Module):
    """Multi-turn dialogue game through the linguistic bottleneck.

    A single agent alternates between Observer and Analyst roles,
    generating a conversation about each input embedding.  Each turn
    is conditioned on all previous turns via a context transformer.
    The receiver scores the full multi-turn dialogue for discrimination.

    Args:
        config: Dialogue game configuration.
        faculty: Pre-built ``LanguageFaculty`` (moved to device by caller).
    """

    def __init__(
        self, config: DialogueGameConfig, faculty: LanguageFaculty,
    ) -> None:
        super().__init__()
        self.config = config
        self.faculty = faculty

        gen = faculty.generator
        gen.eval()
        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        hidden_dim = gen.config.decoder_hidden_dim

        # Diffusion z-generator (shared across all turns)
        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=gen._latent_dim,
            d_model=config.z_hidden_dim,
            max_segments=config.max_segments,
            num_steps=config.diffusion_steps,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            variable_segments=config.variable_segments,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
        )

        # Context transformer (merges data + role + history)
        self.context_transformer = ContextTransformer(
            config.embedding_dim,
            config.context_hidden_dim,
            config.context_heads,
        )

        # Learned role embeddings (Observer and Analyst)
        self.role_embeddings = nn.Parameter(
            torch.randn(2, config.context_hidden_dim) * 0.02,
        )

        # Phrase decoder (shared autoregressive decode logic)
        self.phrase_decoder = ExpressionDecoder(gen)

        # Context projection (decoder hidden → context dim)
        self.hidden_to_context = nn.Linear(hidden_dim, config.context_hidden_dim)

        # Message encoder (reads full multi-turn dialogue)
        self.dialogue_encoder = MessageEncoder(
            hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        self.receiver = Receiver(config.embedding_dim)

    @property
    def gen(self):
        """Shortcut to the underlying generator."""
        return self.faculty.generator

    def checkpoint_state(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "z_gen": self.z_gen.state_dict(),
            "context_transformer": self.context_transformer.state_dict(),
            "role_embeddings": self.role_embeddings.data,
            "hidden_to_context": self.hidden_to_context.state_dict(),
            "dialogue_encoder": self.dialogue_encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
            "num_turns": self.config.num_turns,
            "max_segments": self.config.max_segments,
            "z_generator": "diffusion",
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict."""
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.context_transformer.load_state_dict(ckpt["context_transformer"])
        self.role_embeddings.data.copy_(ckpt["role_embeddings"])
        self.hidden_to_context.load_state_dict(ckpt["hidden_to_context"])
        self.dialogue_encoder.load_state_dict(ckpt["dialogue_encoder"])
        self.receiver.load_state_dict(ckpt["receiver"])

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        cfg = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.context_transformer.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.role_embeddings], "lr": cfg.receiver_lr},
            {"params": list(self.hidden_to_context.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.dialogue_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": cfg.receiver_lr},
        ]

    # -- Forward pass --

    def forward(
        self, anchor: Tensor, distractors: Tensor,
        *, step: int = 0, candidate_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Multi-turn dialogue forward pass.

        For each input, generates ``num_turns`` expressions via alternating
        Observer/Analyst roles.  Each turn is conditioned on the data
        embedding and all previous turns.  The full dialogue is scored
        by the receiver for discrimination.
        """
        batch = anchor.size(0)
        device = anchor.device
        num_candidates = distractors.size(1) + 1

        context = None
        all_hidden: list[Tensor] = []
        all_masks: list[Tensor] = []
        all_tokens_list: list[Tensor] = []
        all_gen_masks: list[Tensor] = []

        for turn in range(self.config.num_turns):
            role = self.role_embeddings[turn % 2]

            # Condition this turn on data + role + history
            conditioning = self.context_transformer(anchor, role, context)

            # Generate z-sequence for this turn
            z_seq, z_weights, _ = self.z_gen(conditioning)

            # Phase 1: no_grad decode through frozen decoder
            tokens, gen_mask, seg_bounds = self.phrase_decoder.decode(z_seq, z_weights)

            # Phase 2: re-run with gradients
            hidden = rerun_decoder_multiseg_with_grad(
                self.gen, z_seq, z_weights, tokens, gen_mask, seg_bounds,
            )
            trimmed_mask = gen_mask[:, :hidden.size(1)]

            all_hidden.append(hidden)
            all_masks.append(trimmed_mask)
            all_tokens_list.append(tokens)
            all_gen_masks.append(gen_mask)

            # Update context: project hidden states and accumulate
            turn_context = self.hidden_to_context(hidden.detach())
            if context is None:
                context = turn_context
            else:
                context = torch.cat([context, turn_context], dim=1)

        # Concatenate full dialogue
        full_hidden = torch.cat(all_hidden, dim=1)
        full_mask = torch.cat(all_masks, dim=1)

        # Encode full dialogue → message vector
        message = self.dialogue_encoder(full_hidden, full_mask)

        # Score against candidates
        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack([
            torch.randperm(num_candidates, device=device)
            for _ in range(batch)
        ])
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = self.receiver(message, candidates)
        loss = F.cross_entropy(logits, target_idx)

        with torch.no_grad():
            accuracy = (logits.argmax(1) == target_idx).float().mean()

            # Surface diversity from first turn
            _eos = self.gen.eos_id
            _seqs = set()
            t0, m0 = all_tokens_list[0], all_gen_masks[0]
            for row, m in zip(t0, m0):
                ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != _eos)
                _seqs.add(hash(ids))
            surface_unique = len(_seqs) / max(t0.size(0), 1)

            total_tokens = full_mask.float().sum(dim=1).mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": total_tokens.detach(),
            "num_segments": torch.tensor(float(self.config.num_turns * self.config.max_segments)),
            "surface_unique": torch.tensor(surface_unique),
            "hs_weight": torch.tensor(1.0),
            "_tokens": all_tokens_list[0].detach(),
            "_gen_mask": all_gen_masks[0].detach(),
        }
