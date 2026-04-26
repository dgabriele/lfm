"""Contrastive expression game — six-term, surface-grounded loss.

A single agent maps a target embedding to one Expression (a sequence of
phrase z's, decoded into a multi-phrase IPA surface). Trained against
the full batch as in-batch negatives via two complementary InfoNCE
heads — one over decoder hidden states, one over the *surface* (token
straight-through) representation. The surface head is what guarantees
the corpus is discriminable to a downstream LLM that only ever sees
strings.

Loss = α·hidden_NCE + β·surface_NCE + γ·topology + δ·z_diversity
       + ε·corpus_KL + ζ·LLM_pressure

Each term has a distinct, complementary role:

  - **hidden_NCE** (α≈1.0): dense gradient discrimination signal in
    decoder hidden space. Fast to learn from.
  - **surface_NCE** (β≈1.0): discrimination signal over the
    straight-through-embedded *output tokens*. Forces messages to be
    distinguishable to anything that reads the IPA — the property the
    downstream Qwen pretraining actually needs.
  - **topology** (γ≈0.1): Pearson-ρ between target-pairwise-sim and
    message-pairwise-sim. Ensures latent geometry preserves embedding
    geometry → compositional generalization.
  - **z_diversity** (δ≈0.1): pairwise cosine penalty across the K
    z-positions of one expression. Prevents within-message bandwidth
    collapse.
  - **corpus_KL** (ε≈0.05): KL(batch-marginal || training unigram).
    Keeps surface marginals Zipfian — aligned with the Qwen tokenizer's
    expectations — preventing discrimination pressure from drifting z
    into corners of decoder space with unnatural output statistics.
  - **LLM_pressure** (ζ≈0.05): frozen Qwen as a fluency floor over
    decoded tokens. Aligns surface forms with what Qwen-7B can learn.

VRAM:
  - Phase 2 gradient rerun is chunked because it is intrinsically a
    grad-memory budgeting concern (O(L²) attention with stored
    activations).
  - All other OOM recovery is delegated to ``AgentTrainer``, which wraps
    each step in ``shrink_on_oom`` (utils.oom).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import (
    MessageEncoder,
    ZDiversityLoss,
    embed_tokens_straight_through,
)
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import ExpressionDecoder, rerun_decoder_multiphrase_with_grad
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.agents.llm_pressure import LLMPressureScorer
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ContrastiveGameConfig(LFMBaseConfig):
    """Configuration for the contrastive expression game."""

    # Faculty
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 96
    vq_codebook_path: str | None = None

    # Diffusion z-generator
    z_hidden_dim: int = 512
    max_phrases: int = 4
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    variable_phrases: bool = True
    target_phrases: float = 2.5
    max_tokens_per_phrase: int = 48

    # Surface encoder (also reused as reference shape for both heads)
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # ----- Six-term loss weights ------------------------------------------
    hidden_infonce_weight: float = 1.0
    surface_infonce_weight: float = 1.0
    topology_weight: float = 0.1
    z_diversity_weight: float = 0.1
    corpus_kl_weight: float = 0.05
    llm_pressure_weight: float = 0.05

    # InfoNCE temperature (shared across both heads).
    contrastive_temperature: float = 0.07

    # corpus_kl reference distribution.  If None or missing on disk,
    # corpus_kl is silently disabled (weight ignored).
    corpus_unigram_path: str | None = None

    # LLM pressure backend.
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    llm_gumbel_tau: float = 1.0

    # Game
    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    # Training (consumed by AgentTrainer)
    batch_size: int = 20
    gradient_accumulation_steps: int = 12
    steps: int = 4000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Phase 2 grad-memory chunking (NOT OOM recovery).
    phase2_vram_budget_mb: float = 1500.0
    phase2_min_chunk: int = 4

    # Output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/contrastive_game"

    # Runtime
    device: str = "cuda"
    seed: int = 42

    # Generator backend
    generator_name: str = "multilingual_vae"
    generator_vocab_size: int | None = None

    # Trainer log-line uses this to know B-way chance.
    contrastive_scoring: bool = True

    def build_faculty_config(self) -> FacultyConfig:
        """Construct the LanguageFaculty config."""
        gen_kwargs = dict(
            name=self.generator_name,
            pretrained_decoder_path=self.decoder_path,
            spm_model_path=self.spm_path,
            num_memory_tokens=self.num_memory_tokens,
            max_seq_len=self.max_output_len,
            vq_codebook_path=self.vq_codebook_path,
        )
        if self.generator_vocab_size is not None:
            gen_kwargs["vocab_size"] = self.generator_vocab_size
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(**gen_kwargs),
        )


# ---------------------------------------------------------------------------
# Generation output
# ---------------------------------------------------------------------------


@dataclass
class ExpressionOutput:
    """Phase-2 hidden states + diagnostics for one Expression."""

    hidden: Tensor       # (B, S, H) Phase 2 hidden states (with grad)
    mask: Tensor         # (B, S) valid-token mask
    z_seq: Tensor        # (B, K, latent_dim)
    z_weights: Tensor    # (B, K)
    num_phrases: Tensor  # (B,) soft phrase count
    tokens_cpu: Tensor   # (B, S) on CPU, diagnostics only
    gen_mask_cpu: Tensor # (B, S) on CPU, diagnostics only


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


class ContrastiveGame(nn.Module):
    """Single-Expression contrastive discrimination with surface grounding.

    All OOM recovery is delegated to ``AgentTrainer.shrink_on_oom``.
    Only intrinsic grad-memory chunking lives in this module.

    Args:
        config: Game configuration.
        faculty: Pre-built ``LanguageFaculty`` (already on device).
    """

    def __init__(
        self, config: ContrastiveGameConfig, faculty: LanguageFaculty,
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
        self._latent_dim = gen._latent_dim
        self._vocab_size = gen._full_vocab

        # ---- Submodules ----
        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=self._latent_dim,
            hidden_dim=config.z_hidden_dim,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            variable_phrases=config.variable_phrases,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
            target_phrases=config.target_phrases,
        )
        self.target_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.phrase_decoder = ExpressionDecoder(gen)

        # Two contrastive heads — one over hidden states, one over the
        # straight-through-embedded surface tokens.  Shared temperature.
        self.hidden_proj = nn.Linear(self._hidden_dim, config.embedding_dim)
        self.surface_encoder = MessageEncoder(
            self._hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.contrastive_temperature)),
        )

        # z-diversity needs the pretrained latent stats to know the
        # baseline pairwise cosine.  Disabled if unavailable.
        if config.z_diversity_weight > 0 and gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(gen._z_mean, gen._z_std)
        else:
            self.z_diversity = None

        # corpus unigram for KL.  Buffer (not param) so it's saved with
        # checkpoints but never updated.  None → KL skipped.
        self._corpus_unigram_loaded = False
        if config.corpus_kl_weight > 0 and config.corpus_unigram_path:
            path = Path(config.corpus_unigram_path)
            if path.exists():
                arr = np.load(path)
                if arr.shape != (self._vocab_size,):
                    logger.warning(
                        "corpus_unigram shape %s != (%d,) — skipping corpus_kl",
                        arr.shape, self._vocab_size,
                    )
                else:
                    self.register_buffer(
                        "corpus_unigram",
                        torch.tensor(arr, dtype=torch.float32).clamp(min=1e-8),
                    )
                    self._corpus_unigram_loaded = True
                    logger.info("Loaded corpus unigram from %s", path)
            else:
                logger.warning(
                    "corpus_unigram_path %s missing — corpus_kl disabled", path,
                )

        # LLM pressure scorer (loads frozen Qwen).  Optional.
        self.llm_pressure: LLMPressureScorer | None = None
        if config.llm_pressure_weight > 0:
            if gen._tokenizer is None:
                raise RuntimeError(
                    "llm_pressure_weight > 0 requires a sentencepiece-backed generator",
                )
            self.llm_pressure = LLMPressureScorer(
                spm_model=gen._tokenizer._sp,
                spm_vocab_size=self._vocab_size,
                llm_model_name=config.llm_model_name,
            )

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    @property
    def gen(self):
        """Underlying frozen generator."""
        return self.faculty.generator

    def trainable_param_groups(self) -> list[dict]:
        """Optimizer param groups with per-group LR."""
        cfg = self.config
        groups = [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.target_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.hidden_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.log_temperature], "lr": cfg.receiver_lr},
        ]
        if self.llm_pressure is not None:
            groups.append({
                "params": [self.llm_pressure.projection], "lr": cfg.receiver_lr,
            })
        return groups

    def checkpoint_state(self) -> dict:
        """Slim checkpoint dict."""
        state = {
            "z_gen": self.z_gen.state_dict(),
            "target_proj": self.target_proj.state_dict(),
            "hidden_proj": self.hidden_proj.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "version": 4,
        }
        if self.llm_pressure is not None:
            state["llm_pressure_projection"] = (
                self.llm_pressure.projection.data.cpu()
            )
        return state

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict."""
        self.z_gen.load_state_dict(ckpt["z_gen"])
        for key, mod in (
            ("target_proj", self.target_proj),
            ("hidden_proj", self.hidden_proj),
            ("surface_encoder", self.surface_encoder),
        ):
            if key in ckpt:
                mod.load_state_dict(ckpt[key])
        if "log_temperature" in ckpt:
            self.log_temperature.data.copy_(ckpt["log_temperature"])
        if self.llm_pressure is not None and "llm_pressure_projection" in ckpt:
            saved = ckpt["llm_pressure_projection"]
            if saved.shape == self.llm_pressure.projection.shape:
                self.llm_pressure.projection.data.copy_(
                    saved.to(self.llm_pressure.projection.device),
                )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, anchor: Tensor) -> ExpressionOutput:
        """z-gen → Phase 1 (no-grad) decode → Phase 2 (grad) chunked rerun."""
        cfg = self.config
        batch = anchor.size(0)
        conditioning = self.target_proj(anchor)

        # Phase 1 — full batch, no grad.
        z_seq, z_weights, num_phrases = self.z_gen(conditioning)
        tokens, gen_mask, seg_bounds = self.phrase_decoder.decode(
            z_seq, z_weights,
            max_tokens_per_phrase=cfg.max_tokens_per_phrase,
        )
        tokens_cpu = tokens.cpu()
        gen_mask_cpu = gen_mask.cpu()

        # Phase 2 — gradient rerun in grad-memory-bounded chunks.
        avg_tok = gen_mask.float().sum(-1).mean().item()
        chunk = self._phase2_chunk(batch, avg_tok)
        all_hidden, all_mask = [], []
        max_seq = 0
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            hidden = rerun_decoder_multiphrase_with_grad(
                self.gen,
                z_seq[start:end], z_weights[start:end],
                tokens[start:end], gen_mask[start:end], seg_bounds[start:end],
            )
            seq_len = hidden.size(1)
            max_seq = max(max_seq, seq_len)
            all_hidden.append(hidden)
            all_mask.append(gen_mask[start:end, :seq_len])

        # Pad chunks to common seq length.
        for i, h in enumerate(all_hidden):
            if h.size(1) < max_seq:
                pad = h.new_zeros(h.size(0), max_seq - h.size(1), h.size(2))
                all_hidden[i] = torch.cat([h, pad], dim=1)
                m = all_mask[i]
                mpad = m.new_zeros(m.size(0), max_seq - m.size(1))
                all_mask[i] = torch.cat([m, mpad], dim=1)

        return ExpressionOutput(
            hidden=torch.cat(all_hidden, dim=0),
            mask=torch.cat(all_mask, dim=0),
            z_seq=z_seq, z_weights=z_weights, num_phrases=num_phrases,
            tokens_cpu=tokens_cpu, gen_mask_cpu=gen_mask_cpu,
        )

    def _phase2_chunk(self, batch: int, avg_tok: float = 50.0) -> int:
        """Pick a chunk size for Phase 2 gradient rerun.

        Phase 2 attention activation cost grows as O(seq_len²).
        Calibrated to ~190 MB/sample at 50 tok.
        """
        cfg = self.config
        per_sample_mb = 190.0 * (max(avg_tok, 1.0) / 50.0) ** 2
        if torch.cuda.is_available():
            used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            available = total_mb - used_mb - 500.0
        else:
            available = cfg.phase2_vram_budget_mb
        budget = min(available, cfg.phase2_vram_budget_mb)
        chunk = max(int(budget // per_sample_mb), cfg.phase2_min_chunk)
        return min(chunk, batch)

    # ------------------------------------------------------------------
    # Loss terms — each a pure function over (expr, anchor).
    # Returns a scalar tensor; callers weight + sum.
    # ------------------------------------------------------------------

    def _pool_hidden(self, expr: ExpressionOutput) -> Tensor:
        """Mask-mean pool decoder hidden states → (B, H)."""
        m = expr.mask.unsqueeze(-1).float()
        lengths = expr.mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (expr.hidden * m).sum(dim=1) / lengths

    def _surface_repr(self, expr: ExpressionOutput) -> Tensor:
        """Straight-through token embedding → MessageEncoder → (B, embedding_dim)."""
        embedded = embed_tokens_straight_through(
            expr.hidden, self.gen.output_head, self.gen.token_embedding,
        )
        return self.surface_encoder(embedded, expr.mask)

    def _info_nce(self, msg: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor]:
        """Symmetric InfoNCE between message and target representations.

        Returns:
            (loss, logits) — loss is symmetric CE; logits are the (B,B)
            similarity matrix (msg → target direction) for accuracy.
        """
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        sim = msg_n @ tgt_n.t()
        logits = sim / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t = F.cross_entropy(logits, labels)
        loss_m = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_t + loss_m), logits

    def _topology_loss(self, msg: Tensor, anchor: Tensor) -> Tensor:
        """1 - Pearson ρ between msg and target pairwise cosine matrices.

        Drives latent geometry to track embedding geometry → compositional
        generalization to novel targets via interpolation.
        """
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        B = msg_n.size(0)
        idx = torch.triu_indices(B, B, offset=1, device=msg_n.device)
        msg_pair = (msg_n @ msg_n.t())[idx[0], idx[1]]
        tgt_pair = (tgt_n @ tgt_n.t())[idx[0], idx[1]]
        mc = msg_pair - msg_pair.mean()
        tc = tgt_pair - tgt_pair.mean()
        denom = (mc.norm() * tc.norm()).clamp(min=1e-8)
        rho = (mc * tc).sum() / denom
        return 1.0 - rho

    def _z_diversity_loss(self, expr: ExpressionOutput) -> Tensor:
        """Hinge penalty on mean intra-expression z cosine."""
        if self.z_diversity is None:
            return expr.hidden.new_tensor(0.0)
        loss, _ = self.z_diversity(expr.z_seq, expr.z_weights)
        return loss

    def _corpus_kl_loss(self, expr: ExpressionOutput) -> Tensor:
        """KL(batch-marginal next-token || training corpus unigram).

        Pulls the batch's aggregate output distribution toward natural
        Zipfian, keeping z exploration in the well-calibrated region of
        the frozen decoder's latent space.
        """
        if not self._corpus_unigram_loaded:
            return expr.hidden.new_tensor(0.0)
        valid = expr.hidden[expr.mask]
        if valid.numel() == 0:
            return expr.hidden.new_tensor(0.0)
        logits = self.gen.output_head(valid)         # (N, V)
        probs = F.softmax(logits, dim=-1)
        p_batch = probs.mean(dim=0).clamp(min=1e-8)  # (V,)
        target = self.corpus_unigram                 # (V,) buffer
        return (p_batch * (p_batch.log() - target.log())).sum()

    def _llm_pressure_loss(self, expr: ExpressionOutput) -> Tensor:
        """Frozen Qwen NLL over the agent's logit sequence."""
        if self.llm_pressure is None:
            return expr.hidden.new_tensor(0.0)
        logits = self.gen.output_head(expr.hidden)
        return self.llm_pressure(
            agent_logits=logits, mask=expr.mask, tau=self.config.llm_gumbel_tau,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, anchor: Tensor, distractors: Tensor,
        *, step: int = 0, candidate_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """One contrastive step.

        Distractors are kept in the signature for trainer compatibility
        but are unused — the loss is in-batch InfoNCE.
        """
        cfg = self.config
        del distractors, candidate_indices, step  # unused here

        # 1. Generate one expression per anchor.
        expr = self._generate(anchor)

        # 2. Compute representations once, reuse across heads.
        hidden_msg = self.hidden_proj(self._pool_hidden(expr))
        surface_msg = self._surface_repr(expr)

        # 3. Six-term loss assembly.  Each term is independent + weighted.
        terms: dict[str, Tensor] = {}
        terms["hidden_nce"], hidden_logits = self._info_nce(hidden_msg, anchor)
        terms["surface_nce"], _ = self._info_nce(surface_msg, anchor)
        terms["topology"] = self._topology_loss(surface_msg, anchor)
        terms["z_div_loss"] = self._z_diversity_loss(expr)
        terms["corpus_kl"] = self._corpus_kl_loss(expr)
        terms["llm_pressure"] = self._llm_pressure_loss(expr)

        weights = {
            "hidden_nce":   cfg.hidden_infonce_weight,
            "surface_nce":  cfg.surface_infonce_weight,
            "topology":     cfg.topology_weight,
            "z_div_loss":   cfg.z_diversity_weight,
            "corpus_kl":    cfg.corpus_kl_weight,
            "llm_pressure": cfg.llm_pressure_weight,
        }
        total = sum(weights[k] * terms[k] for k in weights)

        # 4. Diagnostics.
        with torch.no_grad():
            target_idx = torch.arange(hidden_logits.size(0), device=hidden_logits.device)
            accuracy = (hidden_logits.argmax(1) == target_idx).float().mean()
            msg_lengths = expr.mask.float().sum(dim=1).mean()
            eos = self.gen.eos_id
            seqs = set()
            for row, m in zip(expr.tokens_cpu, expr.gen_mask_cpu):
                ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != eos)
                seqs.add(hash(ids))
            surface_unique = len(seqs) / max(expr.tokens_cpu.size(0), 1)

        out = {
            "loss": total,
            "accuracy": accuracy,
            "msg_lengths": msg_lengths.detach(),
            "num_phrases": expr.num_phrases.mean().detach(),
            "surface_unique": torch.tensor(surface_unique),
            "_tokens": expr.tokens_cpu,
            "_gen_mask": expr.gen_mask_cpu,
        }
        # Per-term scalars for logging (detached).
        for k, v in terms.items():
            out[k] = v.detach()
        return out
