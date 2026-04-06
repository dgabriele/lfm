"""Dialogue expression game V2 — multi-turn self-play with bounded VRAM.

A single agent with two roles generates multi-turn conversations about
input data.  Each turn is conditioned on the original data embedding AND
fixed-size summaries of all previous turns via a context transformer.

Scoring uses progressive topology matching: at each turn, the accumulated
dialogue message must match the embedding cosine similarity structure
(KL divergence).  Later turns that don't improve the topology over earlier
turns get strong gradient signal — natural progressive elaboration.

VRAM management:
- Phase 2 (decoder rerun with gradients) is the only operation whose
  memory scales with sequence length.  It is micro-batched with an
  adaptive chunk size derived from a configurable VRAM budget.
- All other operations (z-gen, Phase 1 decode, scoring, aux losses)
  are fixed-size and run at full batch.
- The frozen decoder is offloaded to CPU during scoring.
- Context uses fixed-size turn summaries, not variable-length concatenation.

Usage::

    poetry run lfm agent dialogue configs/dialogue_phase1.yaml
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import (
    MessageEncoder,
    Receiver,
    ZDiversityLoss,
    embed_tokens_straight_through,
)
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import ExpressionDecoder, rerun_decoder_multiphrase_with_grad
from lfm.agents.diffusion import DiffusionZGenerator, length_distribution_loss
from lfm.agents.vram_monitor import VRAMMonitor
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logger = logging.getLogger(__name__)


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
    max_phrases: int = 4
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    variable_phrases: bool = True
    target_phrases: float = 2.5
    length_weight: float = 0.5
    z_diversity_weight: float = 0.0

    # Per-turn independent scoring weight.  Each turn's message alone
    # must match the topology, preventing adjacent turns from collapsing
    # to identical output.  0.0 = progressive only, 1.0 = equal weight.
    per_turn_weight: float = 0.5

    # Cross-entropy weight alongside topology KL.  CE provides a sharp
    # gradient for "get the argmax right" while KL provides smooth
    # topology matching.  Small values (0.1) nudge without dominating.
    ce_weight: float = 0.1

    # Independent turns: each turn sees only the embedding + turn position,
    # not previous turns.  Produces 4 independent views of the same input,
    # maximizing vocabulary/phrase diversity for LLM pretraining.
    # When False (default), turns are conditioned on previous turns via
    # the context transformer (progressive refinement).
    independent_turns: bool = False

    # Disable turn embeddings: each turn gets identical conditioning
    # (data + context only).  Diversity emerges from context accumulation
    # and stochastic z-gen noise rather than turn identity.
    disable_turn_embeddings: bool = False

    # Inter-turn diversity: penalize pairwise cosine similarity between
    # turn token representations above a target threshold.  Forces turns
    # to be distinct while allowing shared topic vocabulary.
    # target_turn_sim ~0.4 means turns share ~40% of their phonetic
    # content (topic coherence) but differ in the rest (diversity).
    turn_diversity_weight: float = 0.0
    target_turn_sim: float = 0.4

    # Phase 2 VRAM management
    phase2_vram_budget_mb: float = 1500.0
    phase2_min_chunk: int = 4

    # Message encoder
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Game
    num_distractors: int = 15
    min_targets: int = 1
    max_targets: int = 1
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
    """Merge target embeddings, turn position, and dialogue history.

    Produces a conditioning vector for the diffusion z-gen by
    cross-attending from (turn embedding) to both the target
    embeddings and fixed-size turn summaries from previous turns.

    With multiple targets, the model can attend to each target
    individually — learning both what they share and how they differ.

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
        targets: Tensor,
        turn_embedding: Tensor,
        context: Tensor | None,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Produce conditioning for this turn's z-gen.

        Args:
            targets: ``(B, num_targets, embedding_dim)`` target embeddings
                (padded if variable count per sample).
            turn_embedding: ``(hidden_dim,)`` learned turn-position vector.
            context: ``(B, num_prev_turns, hidden_dim)`` fixed-size
                turn summaries, or None for the first turn.
            target_mask: ``(B, num_targets)`` boolean, True = valid target.
                None means all targets are valid.

        Returns:
            ``(B, embedding_dim)`` conditioning vector.
        """
        # Project targets to hidden dim: (B, num_targets, hidden_dim)
        projected_targets = self.data_proj(targets)

        # Query = turn embedding broadcast over batch
        query = turn_embedding.unsqueeze(0).unsqueeze(0).expand(
            targets.size(0), 1, -1,
        )

        # Build KV: targets + context summaries (if any)
        if context is not None:
            kv = torch.cat([projected_targets, context], dim=1)
        else:
            kv = projected_targets

        # Build attention mask if targets are padded
        key_padding_mask = None
        if target_mask is not None:
            n_targets = targets.size(1)
            if context is not None:
                # Targets may be masked; context is always valid
                ctx_valid = torch.ones(
                    targets.size(0), context.size(1),
                    dtype=torch.bool, device=targets.device,
                )
                key_padding_mask = ~torch.cat([target_mask, ctx_valid], dim=1)
            else:
                key_padding_mask = ~target_mask

        kv_normed = self.context_norm(kv)
        attended, _ = self.context_attn(
            query, kv_normed, kv_normed,
            key_padding_mask=key_padding_mask,
        )
        query = query + attended

        return self.out_proj(self.out_norm(query.squeeze(1)))


# ---------------------------------------------------------------------------
# Turn output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TurnOutput:
    """Intermediate output from a single dialogue turn."""

    hidden: Tensor       # (B, S, H) on GPU, with gradients
    mask: Tensor         # (B, S) on GPU
    z_seq: Tensor        # (B, K, latent_dim) on GPU
    z_weights: Tensor    # (B, K) on GPU
    num_phrases: Tensor  # (B,) on GPU
    summary: Tensor      # (B, context_hidden_dim) on GPU, detached
    tokens_cpu: Tensor   # (B, S_raw) on CPU, diagnostics only
    gen_mask_cpu: Tensor # (B, S_raw) on CPU, diagnostics only


# ---------------------------------------------------------------------------
# Dialogue game V2
# ---------------------------------------------------------------------------


class DialogueGame(nn.Module):
    """Multi-turn dialogue game with bounded VRAM.

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

        self._hidden_dim = gen.config.decoder_hidden_dim

        # Diffusion z-generator (shared across all turns)
        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=gen._latent_dim,
            d_model=config.z_hidden_dim,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            variable_phrases=config.variable_phrases,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
            target_phrases=config.target_phrases,
        )

        # Context transformer (merges data + role + history)
        self.context_transformer = ContextTransformer(
            config.embedding_dim,
            config.context_hidden_dim,
            config.context_heads,
        )

        # Learned turn-position embeddings (one per turn).
        # Initialized as a regular simplex (maximally equidistant) so
        # each turn starts maximally distinguishable from all others.
        self.turn_embeddings = nn.Parameter(
            self._simplex_init(config.num_turns, config.context_hidden_dim),
        )

        # Phrase decoder (shared autoregressive decode logic)
        self.phrase_decoder = ExpressionDecoder(gen)

        # Context projection (decoder hidden → fixed-size turn summary)
        self.hidden_to_context = nn.Linear(
            self._hidden_dim, config.context_hidden_dim,
        )

        # Hidden-state encoder (encodes one turn at a time)
        self.dialogue_encoder = MessageEncoder(
            self._hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        # Surface-token encoder (straight-through differentiable tokens)
        self.surface_encoder = MessageEncoder(
            self._hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        self.receiver = Receiver(config.embedding_dim)

        # Learned turn aggregation weights for progressive scoring
        self.turn_agg_logits = nn.Parameter(torch.zeros(config.num_turns))

        # z diversity regularization
        if config.z_diversity_weight > 0 and gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(gen._z_mean, gen._z_std)
        else:
            self.z_diversity = None

        # VRAM monitor (daemon process, saves every ~1 min)
        trace_path = str(Path(config.output_dir) / "vram_trace.npz")
        self.vram_monitor = VRAMMonitor(
            interval=2.0,
            save_path=trace_path,
        )
        self.vram_monitor.start()

        # Ensure clean GPU release on exit
        import atexit
        atexit.register(self.vram_monitor.stop)

    @staticmethod
    def _simplex_init(n: int, dim: int, scale: float = 0.02) -> Tensor:
        """Initialize n vectors as a regular simplex in R^dim.

        All pairwise cosine similarities are equal (-1/(n-1) for a
        centered simplex), giving maximum distinguishability.
        """
        # Start with identity-like rows, then center and normalize
        vecs = torch.randn(n, dim)
        # Gram-Schmidt to get orthogonal basis (n << dim, so this works)
        for i in range(n):
            for j in range(i):
                vecs[i] -= (vecs[i] @ vecs[j]) / (vecs[j] @ vecs[j]) * vecs[j]
            vecs[i] = F.normalize(vecs[i], dim=0)
        # Center so all pairwise distances are equal
        vecs = vecs - vecs.mean(dim=0, keepdim=True)
        vecs = F.normalize(vecs, dim=1) * scale
        return vecs

    @property
    def gen(self):
        """Shortcut to the underlying generator."""
        return self.faculty.generator

    # -- Checkpoint interface --

    def checkpoint_state(self) -> dict:
        """Return state dict for checkpointing."""
        # Save VRAM trace alongside checkpoint
        from pathlib import Path
        trace_path = Path(self.config.output_dir) / "vram_trace.npz"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.vram_monitor.save(str(trace_path))

        return {
            "z_gen": self.z_gen.state_dict(),
            "context_transformer": self.context_transformer.state_dict(),
            "turn_embeddings": self.turn_embeddings.data,
            "hidden_to_context": self.hidden_to_context.state_dict(),
            "dialogue_encoder": self.dialogue_encoder.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
            "turn_agg_logits": self.turn_agg_logits.data,
            "num_turns": self.config.num_turns,
            "max_phrases": self.config.max_phrases,
            "z_generator": "diffusion",
            "version": 2,
            # Full training config for consistent resume (prevents
            # regime shifts when YAML is edited between restarts).
            "training_config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict (V1 and V2 compatible)."""
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.context_transformer.load_state_dict(ckpt["context_transformer"])
        key = "turn_embeddings" if "turn_embeddings" in ckpt else "role_embeddings"
        saved = ckpt[key]
        if saved.size(0) != self.turn_embeddings.size(0):
            # V1 had 2 role embeddings, V2 has num_turns — reinitialize
            logger.info("Turn embedding size mismatch — reinitializing")
        else:
            self.turn_embeddings.data.copy_(saved)
        self.hidden_to_context.load_state_dict(ckpt["hidden_to_context"])
        self.dialogue_encoder.load_state_dict(ckpt["dialogue_encoder"])
        if "surface_encoder" in ckpt:
            self.surface_encoder.load_state_dict(ckpt["surface_encoder"])
        self.receiver.load_state_dict(ckpt["receiver"])
        if "turn_agg_logits" in ckpt:
            self.turn_agg_logits.data.copy_(ckpt["turn_agg_logits"])
        version = ckpt.get("version", 1)
        if version < 2:
            logger.info("Loaded V1 checkpoint — using V2 fixed-size context")

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        cfg = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.context_transformer.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.turn_embeddings], "lr": cfg.receiver_lr},
            {"params": list(self.hidden_to_context.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.dialogue_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.turn_agg_logits], "lr": cfg.receiver_lr},
        ]

    # -- Stage methods --

    def _prepare_candidates(
        self, anchor: Tensor, distractors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Build permuted candidates and precompute teacher topology.

        When ``max_targets > 1``, a random number of distractors are
        promoted to additional targets.  The z-gen receives the full
        set of target embeddings (padded) so it can attend to each
        individually — learning both commonality and differences.

        Returns:
            candidates: ``(B, num_candidates, dim)`` permuted.
            target_idx: ``(B,)`` position of the primary target (for accuracy).
            teacher_probs: ``(B, num_candidates)`` soft topology targets.
            targets: ``(B, max_targets, dim)`` target embeddings (padded).
            target_mask: ``(B, max_targets)`` boolean or None if single target.
        """
        cfg = self.config
        batch = anchor.size(0)
        device = anchor.device
        dim = anchor.size(1)
        num_distractors = distractors.size(1)
        num_candidates = num_distractors + 1

        # Build candidate pool: anchor at position 0, then distractors
        all_cands = torch.cat([anchor.unsqueeze(1), distractors], dim=1)

        if cfg.min_targets == cfg.max_targets == 1:
            # Single-target: anchor is the only target
            perm = torch.stack([
                torch.randperm(num_candidates, device=device)
                for _ in range(batch)
            ])
            perm_expanded = perm.unsqueeze(-1).expand_as(all_cands)
            candidates = torch.gather(all_cands, 1, perm_expanded)
            target_idx = (perm == 0).long().argmax(dim=1)

            with torch.no_grad():
                teacher_sims = F.cosine_similarity(
                    anchor.unsqueeze(1), candidates, dim=-1,
                )
                teacher_probs = F.softmax(teacher_sims / 0.1, dim=-1)

            # Single target as (B, 1, dim) — no mask needed
            return candidates, target_idx, teacher_probs, anchor.unsqueeze(1), None

        # Multi-target: promote random distractors to additional targets
        num_targets_per_sample = torch.randint(
            cfg.min_targets, cfg.max_targets + 1, (batch,), device=device,
        )
        max_t = cfg.max_targets

        # Collect target embeddings (padded to max_targets)
        targets = torch.zeros(batch, max_t, dim, device=device)
        target_valid = torch.zeros(batch, max_t, dtype=torch.bool, device=device)
        is_target = torch.zeros(batch, num_candidates, dtype=torch.bool, device=device)

        targets[:, 0] = anchor
        target_valid[:, 0] = True
        is_target[:, 0] = True

        for b in range(batch):
            n_extra = num_targets_per_sample[b].item() - 1
            if n_extra > 0:
                extra_idx = torch.randperm(num_distractors, device=device)[:n_extra] + 1
                for k, idx in enumerate(extra_idx):
                    targets[b, k + 1] = all_cands[b, idx]
                    target_valid[b, k + 1] = True
                    is_target[b, idx] = True

        # Permute candidates
        perm = torch.stack([
            torch.randperm(num_candidates, device=device)
            for _ in range(batch)
        ])
        perm_expanded = perm.unsqueeze(-1).expand_as(all_cands)
        candidates = torch.gather(all_cands, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        # Teacher topology: mean of targets defines similarity center
        with torch.no_grad():
            tmask = target_valid.unsqueeze(-1).float()
            target_center = (targets * tmask).sum(dim=1) / tmask.sum(dim=1).clamp(min=1)
            teacher_sims = F.cosine_similarity(
                target_center.unsqueeze(1), candidates, dim=-1,
            )
            teacher_probs = F.softmax(teacher_sims / 0.1, dim=-1)

        return candidates, target_idx, teacher_probs, targets, target_valid

    def _summarize_turn(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Produce fixed-size turn summary from variable-length hidden states.

        Returns:
            ``(B, context_hidden_dim)`` summary vector (detached).
        """
        projected = self.hidden_to_context(hidden.detach())
        mask_f = mask.unsqueeze(-1).float()
        return (projected * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def _generate_turn(
        self,
        targets: Tensor,
        turn_emb: Tensor,
        context: Tensor | None,
        target_mask: Tensor | None = None,
    ) -> TurnOutput:
        """Generate one dialogue turn in micro-batches.

        The entire generation pipeline (z-gen → Phase 1 → Phase 2) is
        micro-batched to bound peak VRAM.  Only the context transformer
        runs at full batch (it's cheap — single-token cross-attention).
        """
        batch = targets.size(0)
        conditioning = self.context_transformer(
            targets, turn_emb, context, target_mask,
        )
        chunk = self._compute_generation_chunk(batch)

        all_hidden, all_mask, all_z_seq, all_z_weights = [], [], [], []
        all_num_phrases, all_tokens_cpu, all_gen_mask_cpu = [], [], []
        max_seq = 0

        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            cond_chunk = conditioning[start:end]

            z_seq, z_weights, num_phrases = self.z_gen(cond_chunk)
            tokens, gen_mask, seg_bounds = self.phrase_decoder.decode(z_seq, z_weights)
            hidden = rerun_decoder_multiphrase_with_grad(
                self.gen, z_seq, z_weights, tokens, gen_mask, seg_bounds,
            )

            max_seq = max(max_seq, hidden.size(1))
            all_hidden.append(hidden)
            all_mask.append(gen_mask[:, :hidden.size(1)])
            all_z_seq.append(z_seq)
            all_z_weights.append(z_weights)
            all_num_phrases.append(num_phrases)
            all_tokens_cpu.append(tokens.cpu())
            all_gen_mask_cpu.append(gen_mask.cpu())

            del tokens, gen_mask, seg_bounds
            # Only flush cache if VRAM is tight (>60% used)
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                if used / total > 0.60:
                    torch.cuda.empty_cache()

        # Pad hidden/mask to common seq length and concatenate
        for i, h in enumerate(all_hidden):
            if h.size(1) < max_seq:
                pad = h.new_zeros(h.size(0), max_seq - h.size(1), h.size(2))
                all_hidden[i] = torch.cat([h, pad], dim=1)
                old_m = all_mask[i]
                mpad = old_m.new_zeros(old_m.size(0), max_seq - old_m.size(1))
                all_mask[i] = torch.cat([old_m, mpad], dim=1)

        hidden = torch.cat(all_hidden, dim=0)
        trimmed_mask = torch.cat(all_mask, dim=0)
        z_seq = torch.cat(all_z_seq, dim=0)
        z_weights = torch.cat(all_z_weights, dim=0)
        num_phrases = torch.cat(all_num_phrases, dim=0)

        # Pad tokens/masks on CPU to common size
        max_tok = max(t.size(1) for t in all_tokens_cpu)
        for i, t in enumerate(all_tokens_cpu):
            if t.size(1) < max_tok:
                all_tokens_cpu[i] = F.pad(t, (0, max_tok - t.size(1)))
                all_gen_mask_cpu[i] = F.pad(all_gen_mask_cpu[i], (0, max_tok - all_gen_mask_cpu[i].size(1)))
        tokens_cpu = torch.cat(all_tokens_cpu, dim=0)
        gen_mask_cpu = torch.cat(all_gen_mask_cpu, dim=0)

        summary = self._summarize_turn(hidden, trimmed_mask)

        return TurnOutput(
            hidden=hidden,
            mask=trimmed_mask,
            z_seq=z_seq,
            z_weights=z_weights,
            num_phrases=num_phrases,
            summary=summary,
            tokens_cpu=tokens_cpu,
            gen_mask_cpu=gen_mask_cpu,
        )

    def _compute_generation_chunk(self, batch: int) -> int:
        """Chunk size for full turn generation (z-gen + Phase 1 + Phase 2).

        Uses the configured VRAM budget with a conservative per-sample
        estimate.  The estimate accounts for z-gen, Phase 1 KV cache,
        Phase 2 activations, causal masks, and gradient graph overhead.
        """
        # Empirically measured: batch 48 at 25 tok/turn peaked at 3.4GB
        # over a ~185MB static baseline → ~67MB per sample.
        # At 50 tok/turn this grows to ~130MB.  Use 80MB — the per-step
        # empty_cache prevents accumulator growth.
        bytes_per_sample = 80 * 1024 * 1024
        budget = self.config.phase2_vram_budget_mb * 1024 * 1024
        chunk = max(self.config.phase2_min_chunk, int(budget / bytes_per_sample))
        return min(chunk, batch)

    def _score_progressive(
        self,
        turns: list[TurnOutput],
        candidates: Tensor,
        teacher_probs: Tensor,
        target_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Progressive + independent topology matching.

        Two loss components:
        1. **Progressive**: at each turn, the weighted accumulation of
           turns 0..k must match the topology.  Rewards progressive
           information gain.
        2. **Independent**: each turn's message alone must match the
           topology.  Prevents adjacent turns from collapsing to
           identical output (the T0==T1 problem).

        Returns:
            total_loss: Scalar loss (progressive + per_turn_weight * independent).
            logits: Final turn's logits for accuracy reporting.
        """
        batch = candidates.size(0)
        device = candidates.device
        cfg = self.config
        progressive_loss = torch.tensor(0.0, device=device)
        independent_loss = torch.tensor(0.0, device=device)
        ce_loss = torch.tensor(0.0, device=device)
        gen = self.gen
        score_chunk = 48  # micro-batch for encoder/receiver scoring

        # Process one turn at a time.  Within each turn, micro-batch
        # the encoder passes to bound peak VRAM.
        turn_msgs: list[Tensor] = []
        surface_msg_list: list[Tensor] = []

        for k, t in enumerate(turns):
            # Encode hidden states in chunks → message vector
            msg_chunks = []
            for s in range(0, batch, score_chunk):
                e = min(s + score_chunk, batch)
                msg_chunks.append(self.dialogue_encoder(
                    t.hidden[s:e], t.mask[s:e],
                ))
            turn_msg = torch.cat(msg_chunks, dim=0)
            turn_msgs.append(turn_msg)

            # Progressive: weighted combination of turns 0..k
            weights = F.softmax(self.turn_agg_logits[:k + 1], dim=0)
            msg = sum(w * m for w, m in zip(weights, turn_msgs))

            logits = self.receiver(msg, candidates)
            student_log_probs = F.log_softmax(logits, dim=-1)
            progressive_loss = progressive_loss + F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean",
            ).clamp(min=0)
            if cfg.ce_weight > 0:
                ce_loss = ce_loss + F.cross_entropy(logits, target_idx)

            # Per-turn surface loss (if enabled)
            if cfg.per_turn_weight > 0:
                surf_chunks = []
                for s in range(0, batch, score_chunk):
                    e = min(s + score_chunk, batch)
                    sr = embed_tokens_straight_through(
                        t.hidden[s:e], gen.output_head, gen.token_embedding,
                    )
                    surf_chunks.append(self.surface_encoder(sr, t.mask[s:e]))
                    del sr
                surface_msg = torch.cat(surf_chunks, dim=0)
                ind_logits = self.receiver(surface_msg, candidates)
                ind_log_probs = F.log_softmax(ind_logits, dim=-1)
                independent_loss = independent_loss + F.kl_div(
                    ind_log_probs, teacher_probs, reduction="batchmean",
                ).clamp(min=0)
                del surface_msg

            # Token-level diversity: mean-pool straight-through token
            # embeddings directly (not through the encoder, which
            # compresses to a discriminative point and loses diversity).
            if cfg.turn_diversity_weight > 0:
                tok_chunks = []
                for s in range(0, batch, score_chunk):
                    e = min(s + score_chunk, batch)
                    sr = embed_tokens_straight_through(
                        t.hidden[s:e], gen.output_head, gen.token_embedding,
                    )
                    # Masked mean pool over sequence
                    m = t.mask[s:e].unsqueeze(-1).float()
                    pooled = (sr * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                    tok_chunks.append(pooled)
                    del sr
                surface_msg_list.append(torch.cat(tok_chunks, dim=0))

        # Hard token overlap: measure actual argmax token n-gram overlap
        # between turns.  This is what corpus generation produces —
        # the ground truth for whether turns are actually different.
        # Non-differentiable; used to dynamically gate the differentiable
        # diversity loss.
        # Max pairwise bigram Jaccard across turn pairs, averaged over
        # batch samples.  Tracks the worst-case overlap — catches even
        # one pair of identical turns that mean-averaging would dilute.
        max_overlaps = []
        eos_id = self.gen.eos_id
        for b in range(min(turns[0].tokens_cpu.size(0), 8)):
            pair_max = 0.0
            for i in range(len(turns)):
                for j in range(i + 1, len(turns)):
                    ids_i = [
                        t.item() for t, m in zip(
                            turns[i].tokens_cpu[b], turns[i].gen_mask_cpu[b],
                        ) if m and t.item() != eos_id
                    ]
                    ids_j = [
                        t.item() for t, m in zip(
                            turns[j].tokens_cpu[b], turns[j].gen_mask_cpu[b],
                        ) if m and t.item() != eos_id
                    ]
                    if len(ids_i) >= 2 and len(ids_j) >= 2:
                        bg_i = set(zip(ids_i[:-1], ids_i[1:]))
                        bg_j = set(zip(ids_j[:-1], ids_j[1:]))
                        jaccard = len(bg_i & bg_j) / max(len(bg_i | bg_j), 1)
                        pair_max = max(pair_max, jaccard)
            max_overlaps.append(pair_max)
        hard_overlap = sum(max_overlaps) / max(len(max_overlaps), 1)

        # Differentiable diversity loss on token embeddings, dynamically
        # gated by hard token overlap.  If argmax tokens are already
        # different (low overlap), the loss relaxes.  If they're
        # identical (high overlap), the loss activates strongly.
        diversity_loss = torch.tensor(0.0, device=device)
        if cfg.turn_diversity_weight > 0 and len(surface_msg_list) >= 2:
            # Scale weight by how much hard overlap exceeds target
            gate = max(0.0, (hard_overlap - cfg.target_turn_sim) / (1.0 - cfg.target_turn_sim))
            if gate > 0:
                msgs = torch.stack(surface_msg_list)
                msgs_norm = F.normalize(msgs, dim=-1)
                sim = torch.einsum("ibd,jbd->ijb", msgs_norm, msgs_norm)
                off_diag = ~torch.eye(
                    len(surface_msg_list), dtype=torch.bool, device=device,
                )
                diversity_loss = gate * sim[off_diag].mean()

        total = (progressive_loss
                + cfg.per_turn_weight * independent_loss
                + cfg.ce_weight * ce_loss
                + cfg.turn_diversity_weight * diversity_loss)
        return (total, logits, independent_loss.detach(), ce_loss.detach(),
                diversity_loss.detach(), hard_overlap)

    def _compute_aux_losses(
        self, turns: list[TurnOutput],
    ) -> Tensor:
        """Length regularization and z diversity across all turns."""
        cfg = self.config
        device = turns[0].hidden.device
        aux = torch.tensor(0.0, device=device)

        if cfg.length_weight > 0:
            for t in turns:
                aux = aux + cfg.length_weight * length_distribution_loss(
                    t.z_weights, cfg.target_phrases,
                )

        if self.z_diversity is not None:
            for t in turns:
                div_loss, _ = self.z_diversity(t.z_seq, t.z_weights)
                aux = aux + cfg.z_diversity_weight * div_loss

        return aux

    def _build_output(
        self,
        loss: Tensor,
        logits: Tensor,
        target_idx: Tensor,
        turns: list[TurnOutput],
        surface_loss: Tensor | None = None,
        ce_loss: Tensor | None = None,
        diversity_loss: Tensor | None = None,
        hard_overlap: float = 0.0,
    ) -> dict[str, Tensor]:
        """Assemble output dict for trainer."""
        cfg = self.config
        num_turns = len(turns)

        with torch.no_grad():
            accuracy = (logits.argmax(1) == target_idx).float().mean()

            total_num_phrases = sum(t.num_phrases for t in turns)
            total_masks = torch.cat([t.mask for t in turns], dim=1)
            total_tokens = total_masks.float().sum(dim=1).mean()

            # Surface diversity from first turn
            _eos = self.gen.eos_id
            _seqs = set()
            t0_tok, t0_mask = turns[0].tokens_cpu, turns[0].gen_mask_cpu
            for row, m in zip(t0_tok, t0_mask):
                ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != _eos)
                _seqs.add(hash(ids))
            surface_unique = len(_seqs) / max(t0_tok.size(0), 1)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": total_tokens.detach(),
            "num_phrases": (total_num_phrases / num_turns).mean().detach(),
            "surface_unique": torch.tensor(surface_unique),
            "hs_weight": torch.tensor(1.0),
            "_tokens": turns[0].tokens_cpu.detach(),
            "_gen_mask": turns[0].gen_mask_cpu.detach(),
            "_dialogue_tokens": [t.tokens_cpu.detach() for t in turns],
            "_dialogue_masks": [t.gen_mask_cpu.detach() for t in turns],
            "surface_loss": surface_loss if surface_loss is not None else torch.tensor(0.0),
            "ce_loss": ce_loss if ce_loss is not None else torch.tensor(0.0),
            "turn_sim": diversity_loss.detach() if diversity_loss is not None else torch.tensor(0.0),
            "hard_overlap": torch.tensor(hard_overlap),
        }

    @contextmanager
    def _decoder_offloaded(self):
        """Optionally move frozen decoder to CPU during scoring.

        Only offloads if VRAM is tight (>70% used).  At small batch
        sizes the overhead of CPU↔GPU transfer hurts more than the
        ~90MB saved.
        """
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            tight = used / total > 0.70
        else:
            tight = False

        if tight:
            device = next(self.gen.parameters()).device
            self.gen.cpu()
            torch.cuda.empty_cache()
            try:
                yield
            finally:
                self.gen.to(device)
        else:
            yield

    # -- Main forward --

    def forward(
        self, anchor: Tensor, distractors: Tensor,
        *, step: int = 0, candidate_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Multi-turn dialogue forward pass with bounded VRAM.

        Stages:
          1. Prepare candidates and teacher topology (fixed VRAM)
          2. Generate all turns: z-gen → Phase 1 → Phase 2 micro-batched → summary
          3. Offload decoder, score progressively (fixed VRAM)
          4. Auxiliary losses (fixed VRAM)
          5. Build output dict
        """
        cfg = self.config

        mon = self.vram_monitor
        mon.set_step(step)

        # Stage 1: candidates and teacher topology
        mon.stage = "prepare"
        candidates, target_idx, teacher_probs, targets, target_mask = (
            self._prepare_candidates(anchor, distractors)
        )

        # Stage 2: generate all turns
        context_summaries: list[Tensor] = []
        turns: list[TurnOutput] = []

        for turn_idx in range(cfg.num_turns):
            mon.stage = f"generate_turn{turn_idx}"
            if cfg.disable_turn_embeddings:
                turn_emb = torch.zeros_like(self.turn_embeddings[0])
            else:
                turn_emb = self.turn_embeddings[turn_idx]
            if cfg.independent_turns:
                context = None
            else:
                context = (
                    torch.stack(context_summaries, dim=1)
                    if context_summaries else None
                )
            turn_out = self._generate_turn(
                targets, turn_emb, context, target_mask,
            )
            turns.append(turn_out)
            if not cfg.independent_turns:
                context_summaries.append(turn_out.summary)

        # Stage 3: score with decoder offloaded
        mon.stage = "scoring"
        with self._decoder_offloaded():
            progressive_loss, logits, surface_loss, ce_loss, diversity_loss, hard_overlap = (
                self._score_progressive(
                    turns, candidates, teacher_probs, target_idx,
                )
            )

        # Stage 4: auxiliary losses
        mon.stage = "aux_losses"
        loss = progressive_loss / cfg.num_turns
        loss = loss + self._compute_aux_losses(turns)

        # Stage 5: output
        return self._build_output(
            loss, logits, target_idx, turns,
            surface_loss=surface_loss, ce_loss=ce_loss,
            diversity_loss=diversity_loss, hard_overlap=hard_overlap,
        )
