"""Multi-view expression game — paragraph of N sentences per embedding.

Each sentence = one expression (3 phrases decoded in one AR pass).
N sentences = N z-gen calls, each producing 3 z-vectors.
The full paragraph is scored against candidates via CE.
Stochastic z-gen guarantees diverse sentences within a paragraph.

Usage::

    poetry run lfm agent multiview configs/multiview_v7.yaml
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, Receiver, ZDiversityLoss
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import (
    ExpressionDecoder,
    _compute_phrase_assignment,
    rerun_decoder_multiphrase_with_grad,
)
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


class MultiViewGameConfig(LFMBaseConfig):
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 109

    num_views: int = 4
    max_phrases: int = 3

    z_hidden_dim: int = 512
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    z_diversity_weight: float = 0.5

    encoder: MessageEncoderConfig = MessageEncoderConfig()

    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    steps: int = 2000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    checkpoint_every: int = 100
    log_every: int = 20
    output_dir: str = "data/multiview_game"
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=1,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


class MultiViewGame(nn.Module):
    """Paragraph game: N sentences per embedding."""

    def __init__(self, config: MultiViewGameConfig, faculty: LanguageFaculty):
        super().__init__()
        self.config = config
        self.faculty = faculty
        gen = faculty.generator
        gen.eval()
        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        hdim = gen.config.decoder_hidden_dim

        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=gen._latent_dim,
            d_model=config.z_hidden_dim,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            variable_phrases=False,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
        )

        self.decoder = ExpressionDecoder(gen)

        self.encoder = MessageEncoder(
            hdim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        self.view_weights = nn.Parameter(torch.zeros(config.num_views))
        self.receiver = Receiver(config.embedding_dim)

        if config.z_diversity_weight > 0 and gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(gen._z_mean, gen._z_std)
        else:
            self.z_diversity = None

    @property
    def gen(self):
        return self.faculty.generator

    def checkpoint_state(self):
        return {
            "z_gen": self.z_gen.state_dict(),
            "encoder": self.encoder.state_dict(),
            "view_weights": self.view_weights.data,
            "receiver": self.receiver.state_dict(),
            "config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt):
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.encoder.load_state_dict(ckpt["encoder"])
        if "view_weights" in ckpt:
            self.view_weights.data.copy_(ckpt["view_weights"])
        self.receiver.load_state_dict(ckpt["receiver"])

    def trainable_param_groups(self):
        c = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": c.gru_lr},
            {"params": list(self.encoder.parameters()), "lr": c.receiver_lr},
            {"params": [self.view_weights], "lr": c.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": c.receiver_lr},
        ]

    def _generate_sentence(self, anchor):
        """One sentence: z-gen → decode → Phase 2 → message vector."""
        z_seq, z_weights, num_phrases = self.z_gen(anchor)
        tokens, gen_mask, bounds = self.decoder.decode(z_seq, z_weights)
        hidden = rerun_decoder_multiphrase_with_grad(
            self.gen, z_seq, z_weights, tokens, gen_mask, bounds,
        )
        trimmed_mask = gen_mask[:, :hidden.size(1)]

        pa = _compute_phrase_assignment(
            bounds, trimmed_mask.size(1), z_seq.size(1), anchor.device,
        )
        ppw = z_weights.gather(1, pa)
        message = self.encoder(hidden * ppw.unsqueeze(-1), trimmed_mask)

        return message, z_seq, z_weights, num_phrases, tokens.cpu(), gen_mask.cpu()

    def forward(self, anchor, distractors, *, step=0, candidate_indices=None):
        cfg = self.config
        batch = anchor.size(0)
        device = anchor.device
        num_cand = distractors.size(1) + 1

        # Generate N sentences
        view_msgs = []
        all_z_seqs, all_z_weights = [], []
        total_phrases = torch.zeros(batch, device=device)
        all_tok_cpu, all_mask_cpu = [], []

        for _ in range(cfg.num_views):
            msg, z_seq, z_weights, n_phr, tok_cpu, mask_cpu = (
                self._generate_sentence(anchor)
            )
            view_msgs.append(msg)
            all_z_seqs.append(z_seq)
            all_z_weights.append(z_weights)
            total_phrases += n_phr
            all_tok_cpu.append(tok_cpu)
            all_mask_cpu.append(mask_cpu)

        # Aggregate sentences via learned weights
        weights = F.softmax(self.view_weights, dim=0)
        message = sum(w * m for w, m in zip(weights, view_msgs))

        # Score
        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack([torch.randperm(num_cand, device=device) for _ in range(batch)])
        candidates = torch.gather(candidates, 1, perm.unsqueeze(-1).expand_as(candidates))
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = self.receiver(message, candidates)
        loss = F.cross_entropy(logits, target_idx)

        # Z diversity per sentence
        if self.z_diversity is not None:
            for z_seq, z_weights in zip(all_z_seqs, all_z_weights):
                div_loss, _ = self.z_diversity(z_seq, z_weights)
                loss = loss + cfg.z_diversity_weight * div_loss / cfg.num_views

        with torch.no_grad():
            acc = (logits.argmax(1) == target_idx).float().mean()
            total_tok = sum(m.float().sum() for m in all_mask_cpu) / batch

        return {
            "loss": loss,
            "accuracy": acc,
            "msg_lengths": total_tok.detach(),
            "num_phrases": (total_phrases / cfg.num_views).mean().detach(),
            "hs_weight": torch.tensor(1.0),
            "_tokens": all_tok_cpu[0].detach(),
            "_gen_mask": all_mask_cpu[0].detach(),
            "_dialogue_tokens": [t.detach() for t in all_tok_cpu],
            "_dialogue_masks": [m.detach() for m in all_mask_cpu],
        }
