"""Contrastive expression game over a frozen ``DepTreeVAE`` decoder.

The agent maps a target embedding to ``K`` latent z-vectors via a
diffusion z-generator. Each z is decoded by the frozen dep_tree_vae
pipeline (skeleton → projector → phrase_decoder) into one IPA
constituent. ``K`` constituents concatenate into a single expression.
The expression is scored against in-batch negatives via two
complementary InfoNCE heads (hidden-state and surface) plus four
information-theoretically-motivated regularizers.

Decomposition
-------------

The forward path is decomposed into single-responsibility units:

  * ``_NgramBlocker``      — vectorized n-gram repeat suppression at
                             AR-decode time.
  * ``BigramKLLoss``       — top-K corpus bigram KL with the chunked
                             accumulator interface (``partial`` →
                             ``finalize``) so the (B, S, V) logits
                             never need to materialize at full B.
  * ``AdjDiversityLoss``   — same chunked accumulator interface for
                             the adjacent-softmax cosine hinge.
  * ``ContrastiveGame``    — composes the above. ``forward`` calls
                             dedicated helpers for: (a) generation,
                             (b) the single chunked output_head pass
                             that all (B, S, V)-dependent losses share,
                             (c) the loss assembly, (d) the output
                             dict for ``AgentTrainer``.

Loss
----

  total = α·hidden_NCE + β·surface_NCE + γ·topology
        + δ·z_diversity + ε·bigram_KL + ζ·adj_diversity + η·z_prior
        + η·LLM_pressure
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, ZDiversityLoss
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.agents.llm_pressure import LLMPressureScorer
from lfm.config.base import LFMBaseConfig
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
from lfm.generator.layers import multiscale_causal_mask, precompute_rope_freqs

logger = logging.getLogger(__name__)


# ===========================================================================
# Config
# ===========================================================================


class ContrastiveGameConfig(LFMBaseConfig):
    """Configuration for the dep_tree_vae contrastive game."""

    # Frozen VAE.
    vae_checkpoint: str = "data/models/dep_tree_vae_v1/best.pt"
    vae_config: str = "configs/dep_tree_vae_vast.yaml"
    spm_path: str = "data/models/v15b_ipa/spm.model"

    # Embedding store.
    embedding_dim: int = 384
    embedding_store_dir: str = "data/embeddings"

    # Diffusion z-generator.
    z_hidden_dim: int = 512
    max_phrases: int = 4
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    variable_phrases: bool = True
    target_phrases: float = 2.5

    # Per-phrase generation budget.
    max_tokens_per_phrase: int = 32

    # Surface InfoNCE encoder.
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Loss weights.
    hidden_infonce_weight: float = 1.0
    surface_infonce_weight: float = 1.0
    topology_weight: float = 0.1
    z_diversity_weight: float = 0.1
    bigram_kl_weight: float = 0.05
    adj_diversity_weight: float = 0.05
    z_prior_weight: float = 0.0
    adj_diversity_target: float = 0.30
    llm_pressure_weight: float = 0.0

    # Shared InfoNCE temperature.
    contrastive_temperature: float = 0.07

    # Top-K corpus bigram .npz produced by compute_corpus_bigram.py.
    bigram_kl_path: Optional[str] = None

    # N-gram orders to suppress at AR-decode time. Empty list disables.
    ngram_block: list[int] = [3, 4]

    # Position-scaled EOS bias. logits[eos] += eos_bias_max * (t / max_len).
    # Zero at t=0 (no collapse), grows to eos_bias_max at t=max_len-1.
    # Encourages phrases to close naturally before the hard cap.
    eos_bias_max: float = 0.0

    # Frozen LLM scorer for llm_pressure (loaded only if weight > 0).
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    llm_gumbel_tau: float = 1.0

    # Game / curriculum.
    num_distractors: int = 15
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    steps: int = 4000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Chunk sizes for the heavy intermediate tensors (intrinsic, not OOM
    # recovery — those go through AgentTrainer.shrink_on_oom).
    phase2_chunk: int = 32        # Phase 2 grad rerun batch chunk.
    logit_chunk: int = 32         # Output-head/softmax batch chunk.

    # Output / runtime.
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/contrastive_dep_tree_v1"
    device: str = "cuda"
    seed: int = 42
    # Trainer log-line reads this to know B-way chance.
    contrastive_scoring: bool = True


# ===========================================================================
# Output dataclasses
# ===========================================================================


@dataclass
class ExpressionOutput:
    """What flows from generation to loss computation. Slim by design."""

    hidden: Tensor       # (B, S, H) Phase-2 hidden states with grad
    mask: Tensor         # (B, S) valid-token boolean mask
    z_seq: Tensor        # (B, K, latent_dim)
    z_weights: Tensor    # (B, K) per-phrase activity
    num_phrases: Tensor  # (B,) soft phrase count from z-gen
    tokens_cpu: Tensor   # (B, S) on CPU — diagnostics only
    gen_mask_cpu: Tensor # (B, S) on CPU — diagnostics only


@dataclass
class LogitAggregations:
    """What the chunked output_head pass produces.

    The (B, S, V) tensors logits/probs are NEVER returned: they live
    only inside the chunk loop and are released between chunks. What
    survives is small: per-sample surface-token embeddings and two
    scalar loss values.
    """

    surface_emb: Tensor   # (B, S, H) straight-through token embeddings
    bigram_kl: Tensor     # scalar
    adj_diversity: Tensor # scalar


# ===========================================================================
# Decode utilities
# ===========================================================================


class _NgramBlocker:
    """Vectorized n-gram repeat suppression for AR-decode logits.

    Sets ``logits[b, t] = -inf`` for any token whose emission would
    complete an n-gram that already appeared earlier in the sample's
    history, for each n in ``orders``.
    """

    def __init__(self, orders: list[int]) -> None:
        self.orders = list(orders or ())

    def __call__(self, logits: Tensor, history: Tensor, t: int) -> Tensor:
        for n in self.orders:
            if t < n - 1 + 1:
                continue
            self._block_one(logits, history, t, n)
        return logits

    @staticmethod
    def _block_one(logits: Tensor, history: Tensor, t: int, n: int) -> None:
        """Block (n)-grams whose (n-1)-prefix matches the trailing prefix."""
        prefix = history[:, t - (n - 1) : t]                      # (B, n-1)
        grams = history[:, : t].unfold(1, n - 1, 1)               # (B, M, n-1)
        matches = (grams == prefix.unsqueeze(1)).all(dim=-1)      # (B, M)
        # Only positions where the next token (j + n - 1) is in-range.
        M = matches.size(1)
        j = torch.arange(M, device=logits.device)
        valid = (j + (n - 1)) < t
        matches = matches & valid.unsqueeze(0)
        if not matches.any():
            return
        b_idx, j_idx = matches.nonzero(as_tuple=True)
        banned = history[b_idx, j_idx + (n - 1)]
        logits[b_idx, banned] = float("-inf")


# ===========================================================================
# Chunked-accumulator loss modules
# ===========================================================================


@dataclass
class _BigramPartial:
    topk_sum: Tensor   # (K,) — sum of model's joint mass at top-K bigrams
    n_pairs: Tensor    # scalar — number of valid (t, t+1) pairs in the chunk


class BigramKLLoss(nn.Module):
    """KL(model batch-marginal bigram || corpus top-K bigram) with
    OOV bucket. Supports the chunked accumulator pattern:

        bp = self.zero_partial(...)
        for chunk in batches:
            bp = bp + self.partial(probs_chunk, mask_chunk)
        return self.finalize(bp)
    """

    def __init__(self, npz_path: Optional[Path], vocab_size: int) -> None:
        super().__init__()
        self.is_loaded = False
        if npz_path is None or not npz_path.exists():
            return
        with np.load(npz_path) as f:
            pairs = f["pairs"].astype(np.int64)
            probs = f["probs"].astype(np.float32)
            oov = float(f["oov_prob"])
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))
        self.register_buffer(
            "target_probs",
            torch.tensor(probs, dtype=torch.float32).clamp(min=1e-8),
        )
        self.register_buffer(
            "target_oov", torch.tensor(max(oov, 1e-8), dtype=torch.float32),
        )
        self.is_loaded = True
        logger.info(
            "BigramKLLoss loaded top-%d (covers %.1f%% of mass)",
            pairs.shape[0], (1.0 - oov) * 100,
        )

    @property
    def num_pairs(self) -> int:
        return int(self.pairs.shape[0]) if self.is_loaded else 0

    def zero_partial(self, device: torch.device) -> _BigramPartial:
        return _BigramPartial(
            topk_sum=torch.zeros(self.num_pairs, device=device),
            n_pairs=torch.zeros((), device=device),
        )

    def partial(self, probs: Tensor, mask: Tensor) -> _BigramPartial:
        """Per-chunk contribution: sums the joint mass at each top-K
        bigram across all valid (t, t+1) positions in the chunk.
        """
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        p_t = probs[:, :-1, :]                                    # (c, S-1, V)
        p_t1 = probs[:, 1:, :]
        pair_mask = (mask[:, :-1] & mask[:, 1:]).float()          # (c, S-1)
        # joint = p_t[..., a] * p_t1[..., b]  shape (c, S-1, K)
        joint = p_t[..., a] * p_t1[..., b]
        topk_sum = (joint * pair_mask.unsqueeze(-1)).sum(dim=(0, 1))
        return _BigramPartial(topk_sum=topk_sum, n_pairs=pair_mask.sum())

    def finalize(self, agg: _BigramPartial) -> Tensor:
        if not self.is_loaded or agg.n_pairs.item() == 0:
            return self.target_oov.new_zeros(())
        n = agg.n_pairs.clamp(min=1)
        model_top = (agg.topk_sum / n).clamp(min=1e-12)
        model_oov = (1.0 - model_top.sum()).clamp(min=1e-12, max=1.0 - 1e-8)
        kl_top = (model_top * (model_top.log() - self.target_probs.log())).sum()
        kl_oov = model_oov * (model_oov.log() - self.target_oov.log())
        return kl_top + kl_oov


@dataclass
class _AdjPartial:
    excess_sum: Tensor  # scalar — sum of (cos − target).clamp(min=0) over pairs
    n_pairs: Tensor     # scalar — number of valid adjacency pairs


class AdjDiversityLoss(nn.Module):
    """Hinge on cosine similarity between adjacent softmax distributions.

    cos(p_t, p_{t+1}) is high exactly when the model wants to emit the
    same distribution at consecutive positions — the local mode of
    cycling. Penalizing values above ``target`` is direct anti-cycling
    pressure that needs no corpus reference.
    """

    def __init__(self, target_cos: float) -> None:
        super().__init__()
        self.target_cos = float(target_cos)

    def zero_partial(self, device: torch.device) -> _AdjPartial:
        return _AdjPartial(
            excess_sum=torch.zeros((), device=device),
            n_pairs=torch.zeros((), device=device),
        )

    def partial(self, probs: Tensor, mask: Tensor) -> _AdjPartial:
        pn = F.normalize(probs, dim=-1, eps=1e-8)
        cos = (pn[:, :-1] * pn[:, 1:]).sum(dim=-1)                # (c, S-1)
        pair_mask = (mask[:, :-1] & mask[:, 1:]).float()
        excess = (cos - self.target_cos).clamp(min=0.0)
        return _AdjPartial(
            excess_sum=(excess * pair_mask).sum(),
            n_pairs=pair_mask.sum(),
        )

    def finalize(self, agg: _AdjPartial) -> Tensor:
        if agg.n_pairs.item() == 0:
            return agg.excess_sum.new_zeros(())
        return agg.excess_sum / agg.n_pairs


# ===========================================================================
# Z-prior regularization
# ===========================================================================


def _rbf_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    """RBF kernel: exp(-||x - y||² / (2σ²))."""
    return (-torch.cdist(x, y).pow(2) / (2.0 * sigma ** 2)).exp()


def _rbf_mmd(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    """Unbiased MMD² with a single RBF kernel.

    Uses off-diagonal terms for within-set expectations to remove the
    trivial self-similarity bias.
    """
    n, m = x.size(0), y.size(0)
    kxx = _rbf_kernel(x, x, sigma)
    kyy = _rbf_kernel(y, y, sigma)
    kxy = _rbf_kernel(x, y, sigma)
    return (
        (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        - 2.0 * kxy.mean()
    ).clamp(min=0.0)


class ZPriorLoss(nn.Module):
    """MMD regularization: pulls diffusion z-codes toward the VAE's
    empirical posterior distribution.

    Without this, the diffusion model can drift z-codes into regions
    the frozen decoder was not trained on.  The decoder's response to
    out-of-distribution z is to retreat to its highest-probability
    tokens — exactly the function-word cycling pathology.

    Input z-codes are normalized by the stored prior statistics before
    kernel evaluation, so the bandwidth sigma = sqrt(D) is always
    well-calibrated regardless of the raw z scale.
    """

    def __init__(self, prior_mean: Tensor, prior_std: Tensor) -> None:
        super().__init__()
        self.register_buffer("prior_mean", prior_mean.clone())
        self.register_buffer("prior_std", prior_std.clone())
        self._sigma = math.sqrt(prior_mean.size(-1))

    def forward(self, z_seq: Tensor) -> Tensor:
        """Compute MMD² between z_seq samples and the VAE prior.

        Args:
            z_seq: ``(B, K, D)`` diffusion-generated z-codes.
        """
        z = z_seq.flatten(0, 1)                          # (B*K, D)
        z_norm = (z - self.prior_mean) / self.prior_std  # ≈ N(0, I)
        z_prior = torch.randn_like(z_norm)               # exact N(0, I)
        return _rbf_mmd(z_norm, z_prior, self._sigma)


# ===========================================================================
# Game
# ===========================================================================


class ContrastiveGame(nn.Module):
    """Contrastive expression game over a frozen ``DepTreeVAE``.

    Public interface used by ``AgentTrainer``:

      * ``forward(anchor, distractors, *, step, candidate_indices)``
      * ``render_surface(token_ids, mask, eos_id, output_mode)``
      * ``trainable_param_groups()``
      * ``checkpoint_state()``  / ``load_checkpoint_state(ckpt)``

    All vae parameters are frozen; only the agent-side z-generator,
    target / hidden projections, surface encoder, log-temperature,
    and LLM-pressure projection are trainable.
    """

    # --------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------

    def __init__(self, config: ContrastiveGameConfig, vae: DepTreeVAE) -> None:
        super().__init__()
        self.config = config
        self.vae = self._freeze(vae)

        # Cached scalars derived from the vae.
        self._latent_dim = vae.cfg.latent.total_dim
        self._hidden_dim = vae.cfg.decoder_hidden_dim
        self._vocab_size = vae.cfg.spm_vocab_size + 2
        self._bos_id = vae._bos_id
        self._eos_id = vae._eos_id
        self._max_roles = vae.cfg.skeleton.max_roles

        # Decode utilities.
        self._ngram_blocker = _NgramBlocker(config.ngram_block)
        self._sp = spm.SentencePieceProcessor(model_file=config.spm_path)

        # Trainable components.
        self.z_gen = self._build_z_gen(vae)
        self.target_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.hidden_proj = nn.Linear(self._hidden_dim, config.embedding_dim)
        self.surface_encoder = MessageEncoder(
            self._hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.contrastive_temperature)),
        )

        # Optional regularizers as nn.Module sub-objects.
        self.z_diversity = self._build_z_diversity(vae)
        self.z_prior = self._build_z_prior(vae)
        self.bigram_kl = BigramKLLoss(
            Path(config.bigram_kl_path) if config.bigram_kl_path else None,
            self._vocab_size,
        )
        self.adj_diversity = AdjDiversityLoss(config.adj_diversity_target)
        self.llm_pressure = self._build_llm_pressure()

    # ---- builders kept private and one-line callers ----

    @staticmethod
    def _freeze(vae: DepTreeVAE) -> DepTreeVAE:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae

    def _build_z_gen(self, vae: DepTreeVAE) -> DiffusionZGenerator:
        cfg = self.config
        return DiffusionZGenerator(
            input_dim=cfg.embedding_dim,
            latent_dim=self._latent_dim,
            d_model=cfg.z_hidden_dim,
            num_layers=cfg.diffusion_layers,
            num_heads=cfg.diffusion_heads,
            max_phrases=cfg.max_phrases,
            num_steps=cfg.diffusion_steps,
            variable_phrases=cfg.variable_phrases,
            z_mean=vae._z_struct_mean,
            z_std=vae._z_struct_std,
            target_phrases=cfg.target_phrases,
        )

    def _build_z_diversity(self, vae: DepTreeVAE) -> Optional[ZDiversityLoss]:
        if self.config.z_diversity_weight <= 0:
            return None
        return ZDiversityLoss(vae._z_struct_mean, vae._z_struct_std)

    def _build_z_prior(self, vae: DepTreeVAE) -> Optional[ZPriorLoss]:
        if self.config.z_prior_weight <= 0:
            return None
        return ZPriorLoss(vae._z_struct_mean, vae._z_struct_std)

    def _build_llm_pressure(self) -> Optional[LLMPressureScorer]:
        if self.config.llm_pressure_weight <= 0:
            return None
        return LLMPressureScorer(
            spm_model=self._sp,
            spm_vocab_size=self._vocab_size,
            llm_model_name=self.config.llm_model_name,
        )

    # --------------------------------------------------------------
    # AgentTrainer interface
    # --------------------------------------------------------------

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        groups = [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.target_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.hidden_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.log_temperature], "lr": cfg.receiver_lr},
        ]
        if self.llm_pressure is not None:
            groups.append({"params": [self.llm_pressure.projection], "lr": cfg.receiver_lr})
        return groups

    def checkpoint_state(self) -> dict:
        state = {
            "z_gen": self.z_gen.state_dict(),
            "target_proj": self.target_proj.state_dict(),
            "hidden_proj": self.hidden_proj.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "version": 6,
        }
        if self.llm_pressure is not None:
            state["llm_pressure_projection"] = self.llm_pressure.projection.data.cpu()
        return state

    def load_checkpoint_state(self, ckpt: dict) -> None:
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

    def render_surface(
        self,
        token_ids: Tensor,
        mask: Optional[Tensor] = None,
        eos_id: Optional[int] = None,
        output_mode: str = "ipa",
    ) -> list[str]:
        """Detokenize via SPM. ``AgentTrainer`` calls this each
        checkpoint and respells each result via ``_respell_ipa``."""
        eos = self._eos_id if eos_id is None else eos_id
        spm_size = self._sp.get_piece_size()
        ids_t = token_ids.detach().cpu().tolist()
        mask_t = (
            mask.detach().cpu().tolist() if mask is not None
            else [[True] * len(row) for row in ids_t]
        )
        out: list[str] = []
        for row, mrow in zip(ids_t, mask_t):
            ids = [
                int(t) for t, m in zip(row, mrow)
                if m and int(t) < spm_size and int(t) != eos
            ]
            out.append(self._sp.decode(ids).strip())
        return out

    # --------------------------------------------------------------
    # Forward — orchestrator only. Each step is a dedicated method.
    # --------------------------------------------------------------

    def forward(
        self,
        anchor: Tensor,
        distractors: Tensor,
        *,
        step: int = 0,
        candidate_indices: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """One contrastive step. In-batch InfoNCE; ``distractors`` is
        unused (kept for ``AgentTrainer`` compatibility).
        """
        del distractors, candidate_indices, step

        expr = self._generate(anchor)
        agg = self._chunked_logit_pass(expr)

        terms, hidden_logits = self._compute_loss_terms(expr, anchor, agg)
        total = self._aggregate(terms)
        self._raise_if_nonfinite(terms)

        return self._build_output(total, terms, hidden_logits, expr)

    # --------------------------------------------------------------
    # Generation
    # --------------------------------------------------------------

    def _generate(self, anchor: Tensor) -> ExpressionOutput:
        """z-gen → Phase 1 (no-grad) decode → Phase 2 (grad) chunked rerun."""
        cfg = self.config
        B = anchor.size(0)

        conditioning = self.target_proj(anchor)
        z_seq, z_weights, num_phrases = self.z_gen(conditioning)  # (B, K, D)
        K = z_seq.size(1)
        z_flat = z_seq.reshape(B * K, self._latent_dim)

        # Phase 1: no-grad batched skeleton + AR decode.
        with torch.no_grad():
            roles_flat = self._skeleton_to_roles(z_flat)
            memory_flat = self.vae.phrase_projector(z_flat, roles_flat)
            tokens_flat, mask_flat = self._ar_decode(
                memory_flat, cfg.max_tokens_per_phrase,
            )
        T = tokens_flat.size(1)

        # Phase 2: chunked grad rerun.
        hidden_flat = self._phase2_rerun(z_flat, tokens_flat, mask_flat)

        # Concatenate the K phrases per anchor along the seq dim.
        hidden = hidden_flat.reshape(B, K * T, self._hidden_dim)
        mask = mask_flat.reshape(B, K * T)
        tokens = tokens_flat.reshape(B, K * T)

        return ExpressionOutput(
            hidden=hidden,
            mask=mask,
            z_seq=z_seq,
            z_weights=z_weights,
            num_phrases=num_phrases,
            tokens_cpu=tokens.detach().cpu(),
            gen_mask_cpu=mask.detach().cpu(),
        )

    @torch.no_grad()
    def _skeleton_to_roles(self, z: Tensor) -> Tensor:
        """Greedy skeleton decode → padded role tensor (Bz, max_roles)."""
        skel = self.vae.skeleton_decoder(z)[0]                    # (Bz, R+1)
        Bz = skel.size(0)
        out = torch.zeros(Bz, self._max_roles, dtype=torch.long, device=z.device)
        for i in range(Bz):
            roles = self._extract_roles(skel[i].tolist())
            roles = roles[:self._max_roles] or [DEP_RELATIONS.index("root")]
            pad = roles[-1]
            roles += [pad] * (self._max_roles - len(roles))
            out[i] = torch.tensor(roles, device=z.device)
        return out

    @staticmethod
    def _extract_roles(skel_row: list[int]) -> list[int]:
        roles: list[int] = []
        for tv in skel_row:
            if tv == SKEL_BOS:
                continue
            if tv == SKEL_EOS:
                break
            if tv < len(DEP_RELATIONS):
                roles.append(int(tv))
        return roles

    @torch.no_grad()
    def _ar_decode(self, memory: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
        """Batched greedy AR through frozen phrase_decoder with n-gram blocking."""
        cfg = self.vae.cfg
        Bz = memory.size(0)
        device = memory.device
        rope = self._ensure_rope(max_len, device)

        tokens = torch.full((Bz, 1), self._bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(Bz, dtype=torch.bool, device=device)
        generated = torch.zeros(Bz, max_len, dtype=torch.long, device=device)
        gen_mask = torch.zeros(Bz, max_len, dtype=torch.bool, device=device)

        eos_bias_max = self.config.eos_bias_max
        for t in range(max_len):
            logits = self._ar_step_logits(tokens, memory, rope, cfg)
            if eos_bias_max > 0.0 and t > 0:
                logits[~finished, self._eos_id] += eos_bias_max * (t / max_len)
            logits = self._ngram_blocker(logits, generated, t)
            next_tok = logits.argmax(dim=-1)

            active = ~finished
            generated[:, t] = torch.where(active, next_tok, generated[:, t])
            gen_mask[:, t] = active                              # EOS is valid
            finished = finished | (next_tok == self._eos_id)

            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                gen_mask = gen_mask[:, : t + 1]
                generated = generated[:, : t + 1]
                break
        return generated, gen_mask

    def _ar_step_logits(
        self, tokens: Tensor, memory: Tensor,
        rope: Optional[Tensor], cfg,
    ) -> Tensor:
        tok_emb = self.vae.dec_token_embedding(tokens)
        seq_len = tok_emb.size(1)
        tgt_mask = multiscale_causal_mask(
            seq_len, cfg.decoder_num_heads,
            tuple(cfg.attention_head_windows),
            cfg.attention_global_every, device=tokens.device,
        )
        rope_t = rope[:seq_len] if rope is not None else None
        hidden = self.vae.phrase_decoder(
            tok_emb, memory, tgt_mask=tgt_mask, rope_freqs=rope_t,
        )
        return self.vae.output_head(hidden[:, -1, :])

    def _ensure_rope(self, max_len: int, device: torch.device) -> Optional[Tensor]:
        rope = self.vae._rope_freqs
        if rope is not None and rope.size(0) < max_len + 1:
            cfg = self.vae.cfg
            rope = precompute_rope_freqs(
                self._hidden_dim // cfg.decoder_num_heads,
                max_len + 2, device=device,
            )
            self.vae._rope_freqs = rope
        return rope

    def _phase2_rerun(
        self, z_flat: Tensor, tokens_flat: Tensor, mask_flat: Tensor,
    ) -> Tensor:
        """Re-run phrase_decoder teacher-forced on (z, tokens) with grad,
        chunked along the (B*K) dim to bound activation memory.
        """
        chunks: list[Tensor] = []
        N = z_flat.size(0)
        chunk = max(self.config.phase2_chunk, 1)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            chunks.append(self._phase2_step(z_flat[s:e], tokens_flat[s:e], mask_flat[s:e]))
        return torch.cat(chunks, dim=0)

    def _phase2_step(self, z: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
        cfg = self.vae.cfg
        device = z.device
        with torch.no_grad():
            roles = self._skeleton_to_roles(z)
        memory = self.vae.phrase_projector(z, roles)              # grad path
        T = tokens.size(1)
        bos = torch.full((tokens.size(0), 1), self._bos_id, dtype=torch.long, device=device)
        input_ids = torch.cat([bos, tokens[:, :-1]], dim=1)
        tok_emb = self.vae.dec_token_embedding(input_ids)
        rope = self.vae._rope_freqs
        rope_t = rope[:T] if rope is not None else None
        tgt_mask = multiscale_causal_mask(
            T, cfg.decoder_num_heads,
            tuple(cfg.attention_head_windows),
            cfg.attention_global_every, device=device,
        )
        return self.vae.phrase_decoder(tok_emb, memory, tgt_mask=tgt_mask, rope_freqs=rope_t)

    # --------------------------------------------------------------
    # The chunked output_head pass — one per forward, owns (B, S, V)
    # --------------------------------------------------------------

    def _chunked_logit_pass(self, expr: ExpressionOutput) -> LogitAggregations:
        """Iterate the (B, S, V) output_head pass in chunks of
        ``cfg.logit_chunk`` along B.

        Within each chunk: compute logits + softmax, derive the
        straight-through surface-token embeddings, accumulate per-loss
        partials. The (B, S, V) tensor never materializes at full B.
        """
        device = expr.hidden.device
        B = expr.hidden.size(0)
        chunk = max(self.config.logit_chunk, 1)

        bp = self.bigram_kl.zero_partial(device)
        ap = self.adj_diversity.zero_partial(device)
        emb_chunks: list[Tensor] = []

        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            h = expr.hidden[s:e]
            m = expr.mask[s:e]
            logits = self.vae.output_head(h)                      # (c, S, V)
            probs = F.softmax(logits, dim=-1)

            emb_chunks.append(self._straight_through(logits, probs))
            if self.bigram_kl.is_loaded:
                p = self.bigram_kl.partial(probs, m)
                bp = _BigramPartial(bp.topk_sum + p.topk_sum, bp.n_pairs + p.n_pairs)
            if self.config.adj_diversity_weight > 0:
                p = self.adj_diversity.partial(probs, m)
                ap = _AdjPartial(ap.excess_sum + p.excess_sum, ap.n_pairs + p.n_pairs)
            del logits, probs

        return LogitAggregations(
            surface_emb=torch.cat(emb_chunks, dim=0),
            bigram_kl=self.bigram_kl.finalize(bp),
            adj_diversity=self.adj_diversity.finalize(ap),
        )

    def _straight_through(self, logits: Tensor, probs: Tensor) -> Tensor:
        """Straight-through token embedding: forward = hard one-hot,
        backward = soft probs.
        """
        hard = F.one_hot(logits.argmax(dim=-1), logits.size(-1)).to(probs.dtype)
        st = (hard - probs).detach() + probs                     # (c, S, V)
        return st @ self.vae.dec_token_embedding.weight          # (c, S, H)

    # --------------------------------------------------------------
    # Loss term computation
    # --------------------------------------------------------------

    def _compute_loss_terms(
        self,
        expr: ExpressionOutput,
        anchor: Tensor,
        agg: LogitAggregations,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Each loss is one method. Returns the term dict and the
        hidden-NCE logits matrix for accuracy reporting.
        """
        hidden_msg = self.hidden_proj(self._pool_hidden(expr))
        surface_msg = self._encode_surface(agg.surface_emb, expr.mask)

        hidden_loss, hidden_logits = self._info_nce(hidden_msg, anchor)
        surface_loss, _ = self._info_nce(surface_msg, anchor)

        terms: dict[str, Tensor] = {
            "hidden_nce":   hidden_loss,
            "surface_nce":  surface_loss,
            "topology":     self._topology(surface_msg, anchor),
            "z_div_loss":   self._z_diversity_loss(expr),
            "bigram_kl":    agg.bigram_kl,
            "adj_div":      agg.adj_diversity,
            "llm_pressure": self._llm_pressure_loss(expr),
            "z_prior":      self._z_prior_loss(expr),
        }
        return terms, hidden_logits

    @staticmethod
    def _pool_hidden(expr: ExpressionOutput) -> Tensor:
        m = expr.mask.unsqueeze(-1).float()
        lengths = expr.mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (expr.hidden * m).sum(dim=1) / lengths

    def _encode_surface(self, surface_emb: Tensor, mask: Tensor) -> Tensor:
        # An all-False row would NaN the encoder's attention softmax;
        # patch position 0 to True for any such row.
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return self.surface_encoder(surface_emb, mask)

    def _info_nce(self, msg: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor]:
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        logits = (msg_n @ tgt_n.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return loss, logits

    def _topology(self, msg: Tensor, anchor: Tensor) -> Tensor:
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        B = msg_n.size(0)
        idx = torch.triu_indices(B, B, offset=1, device=msg_n.device)
        mp = (msg_n @ msg_n.t())[idx[0], idx[1]]
        tp = (tgt_n @ tgt_n.t())[idx[0], idx[1]]
        mc, tc = mp - mp.mean(), tp - tp.mean()
        denom = (mc.norm() * tc.norm()).clamp(min=1e-8)
        return 1.0 - (mc * tc).sum() / denom

    def _z_diversity_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.z_diversity is None:
            return expr.hidden.new_zeros(())
        loss, _ = self.z_diversity(expr.z_seq, expr.z_weights)
        return loss

    def _z_prior_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.z_prior is None:
            return expr.hidden.new_zeros(())
        return self.z_prior(expr.z_seq)

    def _llm_pressure_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.llm_pressure is None:
            return expr.hidden.new_zeros(())
        # llm_pressure recomputes its own logits from hidden — it
        # operates on a different chunk-size budget downstream.
        logits = self.vae.output_head(expr.hidden)
        return self.llm_pressure(
            agent_logits=logits, mask=expr.mask, tau=self.config.llm_gumbel_tau,
        )

    # --------------------------------------------------------------
    # Aggregation + output
    # --------------------------------------------------------------

    def _loss_weights(self) -> dict[str, float]:
        cfg = self.config
        return {
            "hidden_nce":   cfg.hidden_infonce_weight,
            "surface_nce":  cfg.surface_infonce_weight,
            "topology":     cfg.topology_weight,
            "z_div_loss":   cfg.z_diversity_weight,
            "bigram_kl":    cfg.bigram_kl_weight,
            "adj_div":      cfg.adj_diversity_weight,
            "llm_pressure": cfg.llm_pressure_weight,
            "z_prior":      cfg.z_prior_weight,
        }

    def _aggregate(self, terms: dict[str, Tensor]) -> Tensor:
        weights = self._loss_weights()
        return sum(weights[k] * terms[k] for k in weights)

    @staticmethod
    def _raise_if_nonfinite(terms: dict[str, Tensor]) -> None:
        for k, t in terms.items():
            if torch.isnan(t) or torch.isinf(t):
                others = {k2: float(v.detach()) for k2, v in terms.items() if k2 != k}
                logger.error("Non-finite loss term: %s = %s. Others: %s", k, t.item(), others)
                raise RuntimeError(f"NaN/Inf in loss term '{k}'")

    def _build_output(
        self,
        total: Tensor,
        terms: dict[str, Tensor],
        hidden_logits: Tensor,
        expr: ExpressionOutput,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            target_idx = torch.arange(hidden_logits.size(0), device=hidden_logits.device)
            accuracy = (hidden_logits.argmax(1) == target_idx).float().mean()
            msg_lengths = expr.mask.float().sum(dim=1).mean()
            surface_unique = self._surface_uniqueness(expr)

        out: dict[str, Tensor] = {
            "loss": total,
            "accuracy": accuracy,
            "msg_lengths": msg_lengths.detach(),
            "num_phrases": expr.num_phrases.mean().detach(),
            "surface_unique": torch.tensor(surface_unique),
            "_tokens": expr.tokens_cpu,
            "_gen_mask": expr.gen_mask_cpu,
        }
        for k, v in terms.items():
            out[k] = v.detach()
        return out

    def _surface_uniqueness(self, expr: ExpressionOutput) -> float:
        eos = self._eos_id
        seen: set[int] = set()
        for row, m in zip(expr.tokens_cpu, expr.gen_mask_cpu):
            ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != eos)
            seen.add(hash(ids))
        return len(seen) / max(expr.tokens_cpu.size(0), 1)
