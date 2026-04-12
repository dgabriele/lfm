"""Dialogue expression game — multi-turn self-play with bounded VRAM.

A single agent generates multi-turn self-dialogue about input data.
Each turn uses a distinct learned turn-position embedding (simplex-
initialized for maximum distinguishability) and is conditioned on the
original data embedding AND fixed-size summaries of all previous turns
via a context transformer.

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
from lfm.agents.llm_pressure import LLMPressureScorer
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

    # Token marginal entropy regularization.  Maintains an EMA of the
    # corpus-level unigram token distribution across training steps and
    # penalizes its entropy, encouraging Zipfian vocabulary structure
    # (stable core vocabulary reused across documents) without collapsing
    # expressivity — the discrimination task provides diversity pressure.
    # Gradient flows through the current batch's soft distribution →
    # Phase 2 hidden states → z-generator.
    # 0.0 = disabled, typical range 0.01–0.1.
    vocab_entropy_weight: float = 0.0
    # EMA decay for the corpus-level marginal.  High α = slow-moving
    # (cross-document stability), low α = more reactive to recent batches.
    vocab_entropy_ema_alpha: float = 0.99

    # LLM-distribution pressure.  When > 0, a frozen pretrained LLM
    # (default Qwen 2.5 0.5B) scores each turn's generated logits and
    # a cross-entropy loss pressures the agent to produce sequences
    # that sit inside the LLM's own latent distribution.  Gradient
    # flows through a learned projection matrix and the agent's
    # output head, all the way back to the VAE decoder hidden states.
    # 0.0 = disabled.  Typical range 0.1–1.0.
    llm_loss_weight: float = 0.0
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    # Gumbel-softmax temperature for the straight-through one-hot that
    # bridges the agent's SPM vocab into Qwen's input-embedding space.
    llm_gumbel_tau: float = 1.0

    # Qwen round-trip REINFORCE reward.  When > 0, a frozen Qwen reads
    # the generated Neuroglot *as text*, extracts its hidden state, and
    # compares to the target embedding via cosine.  The cosine serves
    # as a reward signal for REINFORCE (policy gradient), pushing the
    # agent to produce Neuroglot whose *surface form* encodes target
    # information that Qwen can recover — not just the receiver.
    # This closes the gap between "receiver can discriminate" and
    # "Qwen can read it."  0.0 = disabled.
    qwen_roundtrip_weight: float = 0.0
    qwen_roundtrip_model: str = "Qwen/Qwen2.5-0.5B"
    qwen_roundtrip_samples: int = 20

    # Independent turns: each turn sees only the embedding + turn position,
    # not previous turns.  Produces 4 independent views of the same input,
    # maximizing vocabulary/phrase diversity for LLM pretraining.
    # When False (default), turns are conditioned on previous turns via
    # the context transformer (progressive refinement).
    independent_turns: bool = False

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

        # Corpus-level unigram EMA for vocab entropy regularization.
        # Initialized uniform; updated each step with the current batch's
        # soft marginal distribution.  Saved/loaded with checkpoints.
        vocab_size = gen._full_vocab
        self.register_buffer(
            "vocab_marginal_ema",
            torch.full((vocab_size,), 1.0 / vocab_size),
        )

        # LLM-pressure scorer (frozen Qwen + learned SPM→Qwen projection).
        # Instantiated only when the config enables it; otherwise left as
        # None so runs that don't use this feature pay no cost.
        self.llm_pressure: LLMPressureScorer | None = None
        if config.llm_loss_weight > 0:
            if gen._tokenizer is None:
                raise RuntimeError(
                    "llm_loss_weight > 0 requires a sentencepiece-backed "
                    "generator (spm_model_path must be set in GeneratorConfig)"
                )
            self.llm_pressure = LLMPressureScorer(
                spm_model=gen._tokenizer._sp,
                spm_vocab_size=gen._full_vocab,
                llm_model_name=config.llm_model_name,
            )

        # Qwen round-trip reader: frozen Qwen that reads generated
        # Neuroglot as text and produces hidden states for REINFORCE.
        # Only loaded when the weight is positive; otherwise None.
        self._qwen_reader = None
        self._roundtrip_sp = None
        if config.qwen_roundtrip_weight > 0:
            import sentencepiece as spm_lib
            from lfm.qwen_targets.config import ExtractorConfig
            from lfm.qwen_targets.extractor import HiddenStateExtractor
            self._qwen_reader = HiddenStateExtractor(
                ExtractorConfig(
                    model_name=config.qwen_roundtrip_model,
                    layer=-1,
                    pooling="last_token",
                    dtype="bfloat16",
                    batch_size=config.qwen_roundtrip_samples,
                ),
                device=config.device,
            )
            self._roundtrip_sp = spm_lib.SentencePieceProcessor(
                model_file=config.spm_path,
            )
            self._roundtrip_eos = gen.eos_id
            self._roundtrip_vocab_size = self._roundtrip_sp.vocab_size()
            logger.info(
                "Qwen round-trip reader loaded: %s (weight=%.3f, K=%d)",
                config.qwen_roundtrip_model,
                config.qwen_roundtrip_weight,
                config.qwen_roundtrip_samples,
            )

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

        # Tracks whether gen was stranded on CPU by a failed OOM recovery.
        # Checked at the start of every forward() and restored before use.
        self._gen_stranded_on_cpu: bool = False

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

        state = {
            "z_gen": self.z_gen.state_dict(),
            "context_transformer": self.context_transformer.state_dict(),
            "turn_embeddings": self.turn_embeddings.data,
            "hidden_to_context": self.hidden_to_context.state_dict(),
            "dialogue_encoder": self.dialogue_encoder.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
            "turn_agg_logits": self.turn_agg_logits.data,
            "vocab_marginal_ema": self.vocab_marginal_ema.data,
            "num_turns": self.config.num_turns,
            "max_phrases": self.config.max_phrases,
            "z_generator": "diffusion",
            "version": 2,
        }
        # LLM-pressure projection is a learned parameter (not in the
        # frozen Qwen) and must be persisted so resumes don't throw
        # away the SPM→Qwen bridge.
        if self.llm_pressure is not None:
            state["llm_pressure_projection"] = self.llm_pressure.projection.data.cpu()
        return state

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
        if "vocab_marginal_ema" in ckpt:
            self.vocab_marginal_ema.data.copy_(ckpt["vocab_marginal_ema"])
        if (
            self.llm_pressure is not None
            and "llm_pressure_projection" in ckpt
        ):
            saved_proj = ckpt["llm_pressure_projection"]
            if saved_proj.shape == self.llm_pressure.projection.shape:
                self.llm_pressure.projection.data.copy_(
                    saved_proj.to(self.llm_pressure.projection.device),
                )
                logger.info(
                    "Restored LLM-pressure projection matrix from checkpoint",
                )
            else:
                logger.warning(
                    "LLM-pressure projection shape mismatch %s vs %s — "
                    "keeping fresh MUSE init",
                    saved_proj.shape, self.llm_pressure.projection.shape,
                )
        version = ckpt.get("version", 1)
        if version < 2:
            logger.info("Loaded V1 checkpoint — using V2 fixed-size context")

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        cfg = self.config
        groups = [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.context_transformer.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.turn_embeddings], "lr": cfg.receiver_lr},
            {"params": list(self.hidden_to_context.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.dialogue_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.turn_agg_logits], "lr": cfg.receiver_lr},
        ]
        # The LLM-pressure scorer holds a frozen LLM (no grad) plus a
        # learnable SPM→LLM projection.  Only the projection needs to
        # be in the optimizer; we give it the receiver LR because the
        # bridge is a low-dimensional linear layer that adapts quickly.
        if self.llm_pressure is not None:
            groups.append({
                "params": [self.llm_pressure.projection],
                "lr": cfg.receiver_lr,
            })
        return groups

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
        """Generate one dialogue turn.

        Phase 1 (no_grad decode) runs at full batch — it stores no
        activations for backward, so VRAM cost is just the KV cache
        (~1–2MB/sample) and does not need chunking.  Chunking Phase 1
        forces many small autoregressive loops, keeping the GPU at low
        utilization.

        Only Phase 2 (gradient rerun) is chunked to stay within the
        configured VRAM budget.
        """
        batch = targets.size(0)
        conditioning = self.context_transformer(
            targets, turn_emb, context, target_mask,
        )

        # Phase 1: z-gen + no_grad decode at full batch.
        z_seq, z_weights, num_phrases = self.z_gen(conditioning)
        tokens, gen_mask, seg_bounds = self.phrase_decoder.decode(z_seq, z_weights)
        tokens_cpu = tokens.cpu()
        gen_mask_cpu = gen_mask.cpu()

        # Phase 2: gradient rerun chunked to bound activation VRAM.
        # Compute adaptive chunk using actual sequence length from Phase 1.
        avg_tok = gen_mask.float().sum(-1).mean().item()
        chunk = self._compute_generation_chunk(batch, avg_tok)
        all_hidden, all_mask = [], []
        max_seq = 0

        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            hidden = rerun_decoder_multiphrase_with_grad(
                self.gen,
                z_seq[start:end],
                z_weights[start:end],
                tokens[start:end],
                gen_mask[start:end],
                seg_bounds[start:end],
            )
            seq_len = hidden.size(1)
            max_seq = max(max_seq, seq_len)
            all_hidden.append(hidden)
            all_mask.append(gen_mask[start:end, :seq_len])

            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                if used / total > 0.60:
                    torch.cuda.empty_cache()

        del tokens, seg_bounds

        # Pad hidden/mask chunks to common seq length
        for i, h in enumerate(all_hidden):
            if h.size(1) < max_seq:
                pad = h.new_zeros(h.size(0), max_seq - h.size(1), h.size(2))
                all_hidden[i] = torch.cat([h, pad], dim=1)
                old_m = all_mask[i]
                mpad = old_m.new_zeros(old_m.size(0), max_seq - old_m.size(1))
                all_mask[i] = torch.cat([old_m, mpad], dim=1)

        hidden = torch.cat(all_hidden, dim=0)
        trimmed_mask = torch.cat(all_mask, dim=0)

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

    def _compute_generation_chunk(self, batch: int, avg_tok: float = 50.0) -> int:
        """Chunk size for Phase 2 gradient rerun.

        Phase 2 attention maps scale as O(seq_len²) so we scale the
        per-sample VRAM estimate quadratically with the observed average
        token count from Phase 1.
        Calibration: ~720 MB per sample at 113 tok/turn (measured).

        Uses actual available GPU memory (sampled after Phase 1) with a
        conservative headroom budget so chunk size adapts to whatever is
        already allocated (prior-turn hidden states, model params, etc.).
        """
        # Empirically calibrated: ~975 MB/sample at 113 tok/turn.
        # Using base=190MB so that at tok=113: 190 × (113/50)² ≈ 972MB.
        tok_scale = max(1.0, (avg_tok / 50.0) ** 2)
        bytes_per_sample = int(190 * tok_scale * 1024 * 1024)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            # Reserve 1.5 GB headroom for backward pass, scoring, aux losses.
            headroom = int(1.5 * 1024 ** 3)
            available = max(0, total - allocated - headroom)
        else:
            available = int(self.config.phase2_vram_budget_mb * 1024 * 1024)

        chunk = max(self.config.phase2_min_chunk, int(available / bytes_per_sample))
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
        score_chunk = 16  # micro-batch for encoder/receiver scoring

        # Process one turn at a time.  Within each turn, micro-batch
        # the encoder passes to bound peak VRAM.
        turn_msgs: list[Tensor] = []

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

            # Progressive: weighted combination of turns 0..k.
            # Detach previous turns so each turn's z-gen gradient
            # comes only from its own scoring step, not from the
            # entire accumulated chain.  Reduces gradient variance.
            weights = F.softmax(self.turn_agg_logits[:k + 1], dim=0)
            msg = sum(
                w * (m if i == k else m.detach())
                for i, (w, m) in enumerate(zip(weights, turn_msgs))
            )

            logits = self.receiver(msg, candidates)
            student_log_probs = F.log_softmax(logits, dim=-1)
            progressive_loss = progressive_loss + F.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean",
            )
            if cfg.ce_weight > 0:
                ce_loss = ce_loss + F.cross_entropy(logits, target_idx)

            # Independent: surface tokens in chunks
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
                )
                del surface_msg

        total = (progressive_loss
                + cfg.per_turn_weight * independent_loss
                + cfg.ce_weight * ce_loss)
        return total, logits, independent_loss.detach(), ce_loss.detach()

    def _compute_aux_losses(
        self, turns: list[TurnOutput],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Length regularization, z diversity, vocab entropy, and LLM pressure.

        Returns:
            aux: Weighted sum of all auxiliary loss terms.
            vocab_entropy: Raw (unweighted) marginal entropy scalar, detached,
                for logging.  Zero when vocab_entropy_weight == 0.
            llm_pressure: Raw (unweighted) LLM NLL scalar, detached, for
                logging.  Zero when llm_loss_weight == 0.
        """
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

            # Cross-turn z diversity: penalize if different turns
            # produce z-vectors in the same latent region.
            if len(turns) >= 2:
                turn_z_means = []
                for t in turns:
                    w = t.z_weights.unsqueeze(-1)
                    turn_z_means.append(
                        (t.z_seq * w).sum(dim=1) / w.sum(dim=1).clamp(min=1),
                    )
                stacked = torch.stack(turn_z_means)  # (V, B, latent)
                normed = F.normalize(stacked, dim=-1)
                sim = torch.einsum("ibd,jbd->ijb", normed, normed)
                off_diag = ~torch.eye(len(turns), dtype=torch.bool, device=device)
                aux = aux + cfg.z_diversity_weight * sim[off_diag].mean()

        vocab_entropy = torch.tensor(0.0, device=device)
        if cfg.vocab_entropy_weight > 0:
            # Compute the current batch's marginal token distribution.
            # Accumulate online (sum then divide) to avoid holding all turns'
            # (N_valid × vocab_size) logit tensors simultaneously — with long
            # sequences and large vocabularies this can be several hundred MB.
            p_sum = None
            total_count = 0
            for t in turns:
                valid_hidden = t.hidden[t.mask]  # (N_valid, H)
                logits = self.gen.output_head(valid_hidden)  # (N_valid, vocab_size)
                probs = torch.softmax(logits, dim=-1)
                n = probs.size(0)
                p_sum = probs.sum(dim=0) if p_sum is None else p_sum + probs.sum(dim=0)
                total_count += n
                del logits, probs  # free immediately
            p_batch = p_sum / total_count  # (vocab_size,)

            # Anchor to the corpus-level EMA.  Gradient flows through p_batch's
            # (1-α) contribution; the EMA provides cross-document context so the
            # pressure reflects accumulated vocabulary frequency, not just this batch.
            # Individual documents remain free to be diverse — the discrimination
            # task maintains expressive diversity independently.
            alpha = cfg.vocab_entropy_ema_alpha
            ema = self.vocab_marginal_ema  # (vocab_size,), no grad
            p_ema_anchored = alpha * ema + (1.0 - alpha) * p_batch
            vocab_entropy = -(p_ema_anchored * (p_ema_anchored + 1e-8).log()).sum()
            aux = aux + cfg.vocab_entropy_weight * vocab_entropy

            # Update the EMA buffer for subsequent steps (detached — not part of graph).
            with torch.no_grad():
                self.vocab_marginal_ema.copy_(
                    alpha * ema + (1.0 - alpha) * p_batch.detach()
                )

        llm_pressure = torch.tensor(0.0, device=device)
        if self.llm_pressure is not None and cfg.llm_loss_weight > 0:
            # Score each turn independently.  The agent's generated
            # logits per position come from the output head applied
            # to the Phase 2 hidden states; they are already on the
            # autograd graph, so gradient flows from the LLM NLL back
            # through the VAE decoder → z-generator.
            turn_losses: list[Tensor] = []
            for t in turns:
                logits = self.gen.output_head(t.hidden)  # (B, T, V_spm)
                turn_loss = self.llm_pressure(
                    agent_logits=logits,
                    mask=t.mask,
                    tau=cfg.llm_gumbel_tau,
                )
                turn_losses.append(turn_loss)
            llm_pressure = torch.stack(turn_losses).mean()
            aux = aux + cfg.llm_loss_weight * llm_pressure

        return aux, vocab_entropy.detach(), llm_pressure.detach()

    def _compute_qwen_roundtrip(
        self,
        turns: list[TurnOutput],
        target_embs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """REINFORCE round-trip: Neuroglot → Qwen → cosine with target.

        Subsamples K documents from the batch, detokenizes them to IPA
        text strings, feeds those strings through a frozen Qwen reader
        (no grad), and uses cosine similarity to the target as a reward
        signal for policy gradient.

        The log-probabilities of the generated tokens (from Phase 2
        hidden states) carry gradients through the decoder → z-gen →
        agent.  The reward is detached.  This pushes the agent toward
        Neuroglot whose *surface form* encodes target information that
        Qwen's own text-processing pipeline can recover.

        Returns:
            loss: Weighted REINFORCE loss to add to the total.
            reward_mean: Mean cosine reward (detached) for logging.
        """
        cfg = self.config
        device = target_embs.device
        B = target_embs.size(0)
        K = min(cfg.qwen_roundtrip_samples, B)

        sub_idx = torch.randperm(B, device="cpu")[:K]

        # 1. Detokenize subsampled documents to romanized strings.
        #    Romanization maps IPA → ASCII so Qwen's BPE tokenizer
        #    produces familiar subword splits rather than byte-fallback
        #    on rare Unicode IPA codepoints.
        from lfm.translator.romanize import romanize_iso

        sp = self._roundtrip_sp
        eos = self._roundtrip_eos
        vsize = self._roundtrip_vocab_size
        docs: list[str] = []
        for i in sub_idx:
            parts: list[str] = []
            for t in turns:
                ids = [
                    tok.item()
                    for tok, m in zip(t.tokens_cpu[i], t.gen_mask_cpu[i])
                    if m and tok.item() != eos and tok.item() < vsize
                ]
                ipa = sp.decode(ids).strip()
                if ipa:
                    parts.append(romanize_iso(ipa))
            docs.append(" ".join(parts))

        # 2. Qwen reads the Neuroglot (no grad) → hidden states
        with torch.no_grad():
            qwen_embs = self._qwen_reader.encode(docs).to(device)

        # 3. Cosine reward
        sub_targets = target_embs[sub_idx.to(device)]
        reward = F.cosine_similarity(qwen_embs, sub_targets, dim=-1)
        baseline = reward.mean()
        advantage = (reward - baseline).detach()

        # 4. Log-probabilities of generated tokens from Phase 2 hidden
        #    states (carries gradients through the decoder → z-gen)
        sub_idx_dev = sub_idx.to(device)
        per_sample_log_prob = torch.zeros(K, device=device)
        for t in turns:
            S = t.hidden.size(1)  # Phase 2 seq length (trimmed)
            sub_hidden = t.hidden[sub_idx_dev]
            logits = self.gen.output_head(sub_hidden)  # (K, S, V)
            log_p = F.log_softmax(logits, dim=-1)
            tok_ids = t.tokens_cpu[sub_idx, :S].to(device).clamp(min=0)
            gathered = log_p.gather(2, tok_ids.unsqueeze(-1)).squeeze(-1)
            mask = t.mask[sub_idx_dev].float()  # already trimmed to S
            per_sample_log_prob = per_sample_log_prob + (gathered * mask).sum(dim=1)
            del logits, log_p

        # 5. REINFORCE: -(advantage * log_prob).mean()
        reinforce_loss = -(advantage * per_sample_log_prob).mean()
        loss = cfg.qwen_roundtrip_weight * reinforce_loss
        return loss, reward.mean().detach()

    def _build_output(
        self,
        loss: Tensor,
        logits: Tensor,
        target_idx: Tensor,
        turns: list[TurnOutput],
        surface_loss: Tensor | None = None,
        ce_loss: Tensor | None = None,
        vocab_entropy: Tensor | None = None,
        llm_pressure: Tensor | None = None,
        qwen_roundtrip_reward: Tensor | None = None,
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
            "vocab_entropy": vocab_entropy if vocab_entropy is not None else torch.tensor(0.0),
            "llm_pressure": llm_pressure if llm_pressure is not None else torch.tensor(0.0),
            "qwen_roundtrip": qwen_roundtrip_reward if qwen_roundtrip_reward is not None else torch.tensor(0.0),
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
                torch.cuda.empty_cache()
                try:
                    self.gen.to(device)
                    self._gen_stranded_on_cpu = False
                except RuntimeError:
                    # GPU restore failed (likely secondary OOM after fragmentation).
                    # Mark gen as stranded; forward() will restore it next step
                    # after the trainer has cleared gradients and reduced batch size.
                    self._gen_stranded_on_cpu = True
                    logger.warning(
                        "Decoder restore to GPU failed — will retry at next forward()."
                    )
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

        # Restore decoder to GPU if it was stranded by a prior failed recovery.
        # The trainer has cleared gradients and reduced batch size by now, so
        # VRAM is much lower and the restore should succeed.
        if self._gen_stranded_on_cpu:
            device = anchor.device
            torch.cuda.empty_cache()
            self.gen.to(device)
            self._gen_stranded_on_cpu = False
            logger.info("Decoder restored to %s after stranded-on-CPU recovery.", device)

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
            progressive_loss, logits, surface_loss, ce_loss = (
                self._score_progressive(
                    turns, candidates, teacher_probs, target_idx,
                )
            )

        # Stage 4: auxiliary losses
        mon.stage = "aux_losses"
        loss = progressive_loss / cfg.num_turns
        aux, vocab_entropy, llm_pressure = self._compute_aux_losses(turns)
        loss = loss + aux

        # Stage 4b: Qwen round-trip REINFORCE reward
        qwen_roundtrip_reward = None
        if self._qwen_reader is not None and cfg.qwen_roundtrip_weight > 0:
            mon.stage = "qwen_roundtrip"
            rt_loss, rt_reward = self._compute_qwen_roundtrip(turns, anchor)
            loss = loss + rt_loss
            qwen_roundtrip_reward = rt_reward

        # Stage 5: output
        return self._build_output(
            loss, logits, target_idx, turns,
            surface_loss=surface_loss, ce_loss=ce_loss,
            vocab_entropy=vocab_entropy,
            llm_pressure=llm_pressure,
            qwen_roundtrip_reward=qwen_roundtrip_reward,
        )
