"""Referential game with direct backprop through the linguistic bottleneck.

Two-phase forward:
  1. Generate token sequence with no_grad (fast KV-cached decode).
  2. Re-run the decoder on the generated tokens WITH gradients in one
     parallel pass.  Gradients flow through cross-attention to the latent
     memory back to ``_input_proj``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, Receiver
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import rerun_decoder_with_grad
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


class ReferentialGameConfig(LFMBaseConfig):
    """Configuration for the referential backprop game."""

    # Faculty
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    num_statements: int = 1
    max_output_len: int = 96
    vq_codebook_path: str | None = None
    vq_residual_alpha: float = 1.0

    # Message encoder
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Game
    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    # Training
    batch_size: int = 256
    steps: int = 2000
    sender_lr: float = 3e-5
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Checkpointing / output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/referential_game"

    # Runtime
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        """Construct the ``FacultyConfig`` from game settings."""
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=self.num_statements,
                vq_codebook_path=self.vq_codebook_path,
                vq_residual_alpha=self.vq_residual_alpha,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


class ReferentialGame(nn.Module):
    """Referential game with direct backprop through decoder cross-attention.

    Owns the ``MessageEncoder``, ``Receiver``, and a reference to the
    ``LanguageFaculty``.  The ``forward`` method runs the full two-phase
    pipeline and returns loss + metrics.

    Args:
        config: Game configuration.
        faculty: A pre-built ``LanguageFaculty`` (moved to device by caller).
    """

    def __init__(self, config: ReferentialGameConfig, faculty: LanguageFaculty) -> None:
        super().__init__()
        self.config = config
        self.faculty = faculty

        gen = faculty.generator
        gen.eval()

        # Trigger lazy init of _input_proj
        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        decoder_hidden = gen.config.decoder_hidden_dim
        self.msg_encoder = MessageEncoder(
            decoder_hidden, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.receiver = Receiver(config.embedding_dim)

    @property
    def gen(self):
        """Shortcut to the underlying generator."""
        return self.faculty.generator

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        sender_params = [p for p in self.gen.parameters() if p.requires_grad]
        return [
            {"params": sender_params, "lr": self.config.sender_lr},
            {"params": list(self.msg_encoder.parameters()), "lr": self.config.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": self.config.receiver_lr},
        ]

    def forward(
        self,
        anchor: Tensor,
        distractors: Tensor,
    ) -> dict[str, Tensor]:
        """Run the two-phase referential game forward pass.

        Args:
            anchor: ``(batch, embedding_dim)`` target embeddings.
            distractors: ``(batch, K, embedding_dim)`` distractor embeddings.

        Returns:
            Dict with ``loss``, ``accuracy``, ``msg_lengths``, ``logits``,
            ``target_idx``.
        """
        batch_size = anchor.size(0)
        device = anchor.device
        gen = self.gen
        num_candidates = distractors.size(1) + 1

        # Phase 1: generate tokens (no_grad, fast KV-cached decode)
        with torch.no_grad():
            lfm_outputs = self.faculty(anchor)

        tokens = lfm_outputs["generator.tokens"]
        gen_mask = lfm_outputs["generator.mask"]

        # Phase 2: recompute z with gradients, re-run decoder
        gen._ensure_input_proj(anchor.size(-1))
        embeddings_in = anchor.unsqueeze(1)
        mask_in = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pooled = gen._pool(embeddings_in, mask_in)
        h = gen._input_proj(pooled) + gen._input_refine(pooled)
        n_stmt = gen.config.num_statements
        h = h.view(batch_size, n_stmt, gen._latent_dim * 2)
        mu, _ = h.chunk(2, dim=-1)
        z = mu.reshape(batch_size * n_stmt, gen._latent_dim)

        hidden = rerun_decoder_with_grad(gen, z, tokens, gen_mask)

        # Encode message and score candidates
        message = self.msg_encoder(hidden, gen_mask)

        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack([
            torch.randperm(num_candidates, device=device)
            for _ in range(batch_size)
        ])
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = self.receiver(message, candidates)
        loss = F.cross_entropy(logits, target_idx)

        with torch.no_grad():
            accuracy = (logits.argmax(1) == target_idx).float().mean()
            msg_lengths = gen_mask.float().sum(dim=1).mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": msg_lengths,
            "logits": logits,
            "target_idx": target_idx,
        }
