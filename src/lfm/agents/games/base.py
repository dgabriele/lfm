"""Shared generation infrastructure for DepTreeVAE expression games.

Provides :class:`BaseExpressionGame` — a frozen VAE + diffusion z-generator
+ autoregressive decoder stack.  Subclasses add only their game-specific loss
heads on top.

Decomposition
-------------
* ``_NgramBlocker``         — vectorized n-gram repeat suppression at AR-decode.
* ``ZPriorLoss``            — MMD regularisation toward the VAE posterior.
* ``ExpressionOutput``      — slim dataclass flowing from generation to loss.
* ``BaseExpressionGame``    — generation + decode + shared regularisers.
  Subclasses implement ``forward``, ``trainable_param_groups``,
  ``checkpoint_state``, and ``load_checkpoint_state``.
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

from lfm.agents.components import ZDiversityLoss
from lfm.agents.config import CurriculumConfig
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.config.base import LFMBaseConfig
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
from lfm.generator.layers import KVCache, multiscale_causal_mask, precompute_rope_freqs  # noqa: F401

logger = logging.getLogger(__name__)


# ===========================================================================
# Config
# ===========================================================================


class BaseExpressionGameConfig(LFMBaseConfig):
    """Config fields shared by all DepTreeVAE expression games."""

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

    # N-gram orders to suppress at AR-decode time.
    ngram_block: list[int] = [3, 4]

    # Position-scaled EOS bias.
    eos_bias_max: float = 0.0

    # Hard floor: block EOS for the first N positions of each phrase.
    # Combined with phrase_length_weight for a hard + soft constraint.
    min_tokens_per_phrase: int = 0

    # Soft phrase-length regulariser weight — analogous to length_distribution_loss
    # for phrase count.  Penalises E[soft_length] deviating from target_tokens_per_phrase.
    # Differentiable via the EOS logit survival function in the Phase-2 pass.
    phrase_length_weight: float = 0.0
    # Target mean tokens per phrase.  Set to the VAE training distribution mean.
    target_tokens_per_phrase: int = 0

    # Shared regulariser weights (zero = disabled).
    topology_weight: float = 0.0
    z_diversity_weight: float = 0.0
    z_prior_weight: float = 0.0
    # CE of Phase-1 greedy tokens under the frozen decoder Phase-2 rerun.
    # Low = z is on the decoder manifold; high = off-manifold cycling.
    decoder_fluency_weight: float = 0.0

    # Training.
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    steps: int = 4000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    num_distractors: int = 0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Memory-management chunk sizes.
    phase2_chunk: int = 32
    logit_chunk: int = 32

    # Output / runtime.
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/expression_game"
    device: str = "cuda"
    seed: int = 42


# ===========================================================================
# Shared output dataclass
# ===========================================================================


@dataclass
class ExpressionOutput:
    """What flows from generation to loss computation."""

    hidden: Tensor        # (B, S, H)  Phase-2 hidden states — has grad
    mask: Tensor          # (B, S)     valid-token boolean mask
    z_seq: Tensor         # (B, K, latent_dim)
    z_weights: Tensor     # (B, K)     per-phrase activity
    num_phrases: Tensor   # (B,)       soft phrase count from z-gen
    tokens_cpu: Tensor    # (B, S)     on CPU — diagnostics / fluency target
    gen_mask_cpu: Tensor  # (B, S)     on CPU — diagnostics


# ===========================================================================
# N-gram blocker
# ===========================================================================


class _NgramBlocker:
    """Vectorized n-gram repeat suppression for AR-decode logits."""

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
        prefix = history[:, t - (n - 1) : t]
        grams = history[:, :t].unfold(1, n - 1, 1)
        matches = (grams == prefix.unsqueeze(1)).all(dim=-1)
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
# Z-prior regularisation
# ===========================================================================


def _rbf_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    return (-torch.cdist(x, y).pow(2) / (2.0 * sigma ** 2)).exp()


def _rbf_mmd(x: Tensor, y: Tensor, sigma: float) -> Tensor:
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
    """MMD regularisation: pulls diffusion z-codes toward the VAE posterior."""

    def __init__(self, prior_mean: Tensor, prior_std: Tensor) -> None:
        super().__init__()
        self.register_buffer("prior_mean", prior_mean.clone())
        self.register_buffer("prior_std", prior_std.clone())
        self._sigma = math.sqrt(prior_mean.size(-1))

    def forward(self, z_seq: Tensor) -> Tensor:
        z = z_seq.flatten(0, 1)
        z_norm = (z - self.prior_mean) / self.prior_std
        z_prior = torch.randn_like(z_norm)
        return _rbf_mmd(z_norm, z_prior, self._sigma)


# ===========================================================================
# Shared chunked-accumulator loss modules (used by multiple game subclasses)
# ===========================================================================


@dataclass
class _BigramPartial:
    topk_sum: Tensor
    n_pairs: Tensor


class BigramKLLoss(nn.Module):
    """KL(model batch-marginal bigram || corpus top-K bigram) with OOV bucket."""

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
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        p_t = probs[:, :-1, :]
        p_t1 = probs[:, 1:, :]
        pair_mask = (mask[:, :-1] & mask[:, 1:]).float()
        joint = p_t[..., a] * p_t1[..., b]
        topk_sum = (joint * pair_mask.unsqueeze(-1)).sum(dim=(0, 1))
        return _BigramPartial(topk_sum=topk_sum, n_pairs=pair_mask.sum())

    def finalize(self, agg: _BigramPartial) -> Tensor:
        if not self.is_loaded or agg.n_pairs.item() == 0:
            return agg.n_pairs.new_zeros(())
        n = agg.n_pairs.clamp(min=1)
        model_top = (agg.topk_sum / n).clamp(min=1e-12)
        model_oov = (1.0 - model_top.sum()).clamp(min=1e-12, max=1.0 - 1e-8)
        kl_top = (model_top * (model_top.log() - self.target_probs.log())).sum()
        kl_oov = model_oov * (model_oov.log() - self.target_oov.log())
        return kl_top + kl_oov


@dataclass
class _AdjPartial:
    excess_sum: Tensor
    n_pairs: Tensor


class AdjDiversityLoss(nn.Module):
    """Hinge on cosine similarity between adjacent softmax distributions."""

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
        cos = (pn[:, :-1] * pn[:, 1:]).sum(dim=-1)
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
# Base game
# ===========================================================================


class BaseExpressionGame(nn.Module):
    """Frozen VAE + diffusion z-gen + AR decoder — shared by all expression games.

    Subclasses must implement:
      * ``forward(anchor, distractors, *, step, candidate_indices)``
      * ``trainable_param_groups() -> list[dict]``
      * ``checkpoint_state() -> dict``
      * ``load_checkpoint_state(ckpt: dict) -> None``
    """

    def __init__(self, config: BaseExpressionGameConfig, vae: DepTreeVAE) -> None:
        super().__init__()
        self.config = config
        self.vae = self._freeze(vae)

        self._latent_dim = vae.cfg.latent.total_dim
        self._hidden_dim = vae.cfg.decoder_hidden_dim
        self._vocab_size = vae.cfg.spm_vocab_size + 2
        self._bos_id = vae._bos_id
        self._eos_id = vae._eos_id
        self._max_roles = vae.cfg.skeleton.max_roles

        self._ngram_blocker = _NgramBlocker(config.ngram_block)
        self._sp = spm.SentencePieceProcessor(model_file=config.spm_path)

        self.z_gen = self._build_z_gen(vae)
        self.target_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.z_diversity = self._build_z_diversity(vae)
        self.z_prior = self._build_z_prior(vae)

    # ---- builders ----

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

    # ---- AgentTrainer interface (render_surface is the only shared one) ----

    def render_surface(
        self,
        token_ids: Tensor,
        mask: Optional[Tensor] = None,
        eos_id: Optional[int] = None,
        output_mode: str = "ipa",
    ) -> list[str]:
        """Detokenize via SPM."""
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

    # ---- Generation ----

    def _generate(self, anchor: Tensor) -> ExpressionOutput:
        """z-gen → Phase 1 (no-grad) decode → Phase 2 (grad) chunked rerun."""
        cfg = self.config
        B = anchor.size(0)

        conditioning = self.target_proj(anchor)
        z_seq, z_weights, num_phrases = self.z_gen(conditioning)
        K = z_seq.size(1)
        z_flat = z_seq.reshape(B * K, self._latent_dim)

        with torch.no_grad():
            roles_flat = self._skeleton_to_roles(z_flat)
            memory_flat = self.vae.phrase_projector(z_flat, roles_flat)
            tokens_flat, mask_flat = self._ar_decode(memory_flat, cfg.max_tokens_per_phrase)
        T = tokens_flat.size(1)

        hidden_flat = self._phase2_rerun(z_flat, tokens_flat, mask_flat)

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
        skel = self.vae.skeleton_decoder(z)[0]
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
        """KV-cached batched greedy AR decode."""
        cfg = self.vae.cfg
        Bz = memory.size(0)
        device = memory.device
        rope = self._ensure_rope(max_len, device)

        full_mask = multiscale_causal_mask(
            max_len + 1, cfg.decoder_num_heads,
            tuple(cfg.attention_head_windows),
            cfg.attention_global_every, device=device,
        )
        cache = self.vae.phrase_decoder.make_kv_cache(
            Bz, max_len + 1, device, dtype=memory.dtype,
        )

        curr_tok = torch.full((Bz, 1), self._bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(Bz, dtype=torch.bool, device=device)
        generated = torch.zeros(Bz, max_len, dtype=torch.long, device=device)
        gen_mask = torch.zeros(Bz, max_len, dtype=torch.bool, device=device)

        eos_bias_max = self.config.eos_bias_max

        for t in range(max_len):
            pos = cache.seq_len
            tok_emb = self.vae.dec_token_embedding(curr_tok)
            mask_row = full_mask[:, pos : pos + 1, : pos + 1]
            hidden = self.vae.phrase_decoder.forward_cached(
                tok_emb, memory, cache,
                rope_freqs=rope,
                tgt_mask_row=mask_row,
            )
            cache.advance()

            logits = self.vae.output_head(hidden[:, -1, :])
            if eos_bias_max > 0.0 and t > 0:
                logits[~finished, self._eos_id] += eos_bias_max * (t / max_len)
            if t < self.config.min_tokens_per_phrase:
                logits[:, self._eos_id] = float("-inf")
            logits = self._ngram_blocker(logits, generated, t)
            next_tok = logits.argmax(dim=-1)

            active = ~finished
            generated[:, t] = torch.where(active, next_tok, generated[:, t])
            gen_mask[:, t] = active
            finished = finished | (next_tok == self._eos_id)

            curr_tok = next_tok.unsqueeze(1)
            if finished.all():
                gen_mask = gen_mask[:, : t + 1]
                generated = generated[:, : t + 1]
                break

        return generated, gen_mask

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
        """Re-run phrase_decoder teacher-forced on (z, tokens) with grad."""
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
        memory = self.vae.phrase_projector(z, roles)
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

    # ---- Shared utilities ----

    def _straight_through(self, logits: Tensor, probs: Tensor) -> Tensor:
        """Straight-through token embedding: forward = hard one-hot, backward = soft probs."""
        hard = F.one_hot(logits.argmax(dim=-1), logits.size(-1)).to(probs.dtype)
        st = (hard - probs).detach() + probs             # (B, S, V)
        return st @ self.vae.dec_token_embedding.weight  # (B, S, H)

    # ---- Shared loss helpers ----

    @staticmethod
    def _pool_hidden(expr: ExpressionOutput) -> Tensor:
        """Mean-pool valid hidden states: (B, S, H) → (B, H)."""
        m = expr.mask.unsqueeze(-1).float()
        lengths = expr.mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (expr.hidden * m).sum(dim=1) / lengths

    @staticmethod
    def _topology(msg: Tensor, anchor: Tensor) -> Tensor:
        """1 − Pearson correlation between pairwise cosines in msg- and anchor-space."""
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        B = msg_n.size(0)
        idx = torch.triu_indices(B, B, offset=1, device=msg_n.device)
        mp = (msg_n @ msg_n.t())[idx[0], idx[1]]
        tp = (tgt_n @ tgt_n.t())[idx[0], idx[1]]
        mc, tc = mp - mp.mean(), tp - tp.mean()
        denom = (mc.norm() * tc.norm()).clamp(min=1e-8)
        return 1.0 - (mc * tc).sum() / denom

    def _phrase_length_loss_chunk(
        self, logits_phr: Tensor, mask_phr: Tensor, target: float,
    ) -> tuple[Tensor, Tensor]:
        """Per-chunk contribution to the soft phrase-length loss.

        Args:
            logits_phr: ``(B_chunk * K, T, V)`` output_head logits reshaped
                        to one row per phrase.
            mask_phr:   ``(B_chunk * K, T)`` valid-token mask per phrase.

        Returns:
            ``(soft_len_sum, count)`` — accumulate across chunks, then compute
            ``(sum / count - target)^2`` to get the final loss.
        """
        # Use softmax EOS prob — properly normalized over the full vocab.
        # sigmoid(eos_logit) would give 0.2-0.5 regardless of competing tokens;
        # softmax gives the true generative probability of EOS which is tiny
        # when many non-EOS tokens have higher logits.
        eos_prob = F.softmax(logits_phr, dim=-1)[:, :, self._eos_id]
        eos_prob = eos_prob * mask_phr.float()
        survival = torch.cumprod(1.0 - eos_prob + 1e-8, dim=1)
        soft_len = survival.sum(dim=1)                 # (B_chunk * K,)
        return soft_len.sum(), torch.tensor(float(soft_len.numel()), device=soft_len.device)

    def _z_diversity_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.z_diversity is None:
            return expr.hidden.new_zeros(())
        loss, _ = self.z_diversity(expr.z_seq, expr.z_weights)
        return loss

    def _z_prior_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.z_prior is None:
            return expr.hidden.new_zeros(())
        return self.z_prior(expr.z_seq)

    # ---- Diagnostics ----

    def _surface_uniqueness(self, expr: ExpressionOutput) -> float:
        eos = self._eos_id
        seen: set[int] = set()
        for row, m in zip(expr.tokens_cpu, expr.gen_mask_cpu):
            ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != eos)
            seen.add(hash(ids))
        return len(seen) / max(expr.tokens_cpu.size(0), 1)

    def _mean_unique_tokens(self, expr: ExpressionOutput) -> float:
        eos = self._eos_id
        total = 0
        for row, m in zip(expr.tokens_cpu, expr.gen_mask_cpu):
            ids = {t.item() for t, v in zip(row, m) if v and t.item() != eos}
            total += len(ids)
        return total / max(expr.tokens_cpu.size(0), 1)

    @staticmethod
    def _raise_if_nonfinite(terms: dict[str, Tensor]) -> None:
        for k, t in terms.items():
            if torch.isnan(t) or torch.isinf(t):
                others = {k2: float(v.detach()) for k2, v in terms.items() if k2 != k}
                logger.error("Non-finite loss term: %s = %s. Others: %s", k, t.item(), others)
                raise RuntimeError(f"NaN/Inf in loss term '{k}'")
