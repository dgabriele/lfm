"""Contrastive expression game on a frozen ``DepTreeVAE`` decoder.

The agent maps a target embedding to ``K`` latent z-vectors via a
diffusion z-generator. Each z is decoded by the frozen dep_tree_vae
pipeline (skeleton → projector → phrase_decoder) into one IPA
constituent. ``K`` constituents concatenate into a single expression.
The expression is scored against in-batch negatives via two
complementary InfoNCE heads (hidden-state and surface) plus four
information-theoretically-motivated regularizers.

Loss = α·hidden_NCE + β·surface_NCE + γ·topology + δ·z_diversity
       + ε·corpus_KL + ζ·LLM_pressure

  - **hidden_NCE**: dense gradient discrimination signal in decoder
    hidden space.
  - **surface_NCE**: discrimination over straight-through-embedded
    output tokens — what the downstream LLM actually sees.
  - **topology**: 1 − Pearson ρ between target-pairwise and
    message-pairwise cosine, enforcing compositional latent geometry.
  - **z_diversity**: pairwise cosine penalty across the K z-positions
    of one expression — prevents within-message bandwidth collapse.
  - **corpus_KL**: KL(batch-marginal || training unigram) — pulls
    aggregate marginals toward Zipfian for tokenizer alignment.
  - **LLM_pressure**: frozen Qwen NLL — fluency floor aligned with
    downstream pretraining.

VRAM:
  - Phase 2 grad re-run is chunked along the batch dim (intrinsic
    O(L²) attention activation cost).
  - All other OOM recovery is delegated to ``AgentTrainer`` which
    wraps each step in ``shrink_on_oom``.
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
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.agents.llm_pressure import LLMPressureScorer
from lfm.config.base import LFMBaseConfig
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
from lfm.generator.layers import multiscale_causal_mask, precompute_rope_freqs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ContrastiveGameConfig(LFMBaseConfig):
    """Configuration for the dep_tree_vae contrastive game."""

    # Frozen VAE — checkpoint produced by ``lfm pretrain dep_tree_vae``.
    vae_checkpoint: str = "data/models/dep_tree_vae_v1/best.pt"
    vae_config: str = "configs/dep_tree_vae_vast.yaml"
    spm_path: str = "data/models/v15b_ipa/spm.model"

    # Embedding store
    embedding_dim: int = 384
    embedding_store_dir: str = "data/embeddings"

    # Diffusion z-generator
    z_hidden_dim: int = 512
    max_phrases: int = 4
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    variable_phrases: bool = True
    target_phrases: float = 2.5

    # Per-phrase generation budget
    max_tokens_per_phrase: int = 32

    # Surface encoder (used by surface InfoNCE head)
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # ----- Loss weights -----
    hidden_infonce_weight: float = 1.0
    surface_infonce_weight: float = 1.0
    topology_weight: float = 0.1
    z_diversity_weight: float = 0.1
    # Bigram-KL replaces unigram corpus_kl. Cycles ("the the", "of of")
    # produce bigrams not present in the corpus; bigram-KL penalizes
    # them at the layer where they appear.
    bigram_kl_weight: float = 0.05
    # adj_diversity: cosine-similarity hinge between adjacent softmax
    # distributions. Direct anti-cycling pressure that does not depend
    # on inter-sample comparisons.
    adj_diversity_weight: float = 0.05
    adj_diversity_target: float = 0.30
    llm_pressure_weight: float = 0.0

    # InfoNCE temperature (shared across both heads).
    contrastive_temperature: float = 0.07

    # Precomputed top-K corpus bigrams (.npz with pairs/probs/oov_prob).
    # Skipped if missing.
    bigram_kl_path: str | None = None

    # N-gram blocking at Phase 1 decode. Empty list disables. List of
    # n-gram orders to block (e.g. [3, 4] blocks any 3- or 4-gram from
    # repeating within a phrase).
    ngram_block: list[int] = [3, 4]

    # Frozen LLM scorer for llm_pressure. Loaded only if weight > 0.
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    llm_gumbel_tau: float = 1.0

    # Game / curriculum
    num_distractors: int = 15
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    steps: int = 4000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Phase 2 grad-rerun chunking (intrinsic — not OOM recovery).
    phase2_chunk: int = 8

    # Output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/contrastive_dep_tree_v1"

    # Runtime
    device: str = "cuda"
    seed: int = 42

    # AgentTrainer log-line uses this to know B-way chance.
    contrastive_scoring: bool = True


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------


@dataclass
class ExpressionOutput:
    """Concatenated K-phrase expression with hidden states + diagnostics."""

    hidden: Tensor       # (B, S, H) Phase-2 hidden states (with grad)
    logits: Tensor       # (B, S, V) output_head(hidden), computed ONCE
    probs: Tensor        # (B, S, V) softmax(logits), computed ONCE
    mask: Tensor         # (B, S) valid-token mask
    z_seq: Tensor        # (B, K, latent_dim)
    z_weights: Tensor    # (B, K) per-phrase activity
    num_phrases: Tensor  # (B,) soft phrase count from z-gen
    tokens_cpu: Tensor   # (B, S) on CPU, diagnostics only
    gen_mask_cpu: Tensor # (B, S) on CPU, diagnostics only


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


class ContrastiveGame(nn.Module):
    """Contrastive expression game over a frozen ``DepTreeVAE``.

    All vae parameters are frozen; only the agent-side z-generator,
    target projection, and loss heads are trainable.

    Args:
        config: Game configuration.
        vae: A pre-loaded ``DepTreeVAE`` (already on device).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: ContrastiveGameConfig, vae: DepTreeVAE) -> None:
        super().__init__()
        self.config = config
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.latent_dim = vae.cfg.latent.total_dim
        self.hidden_dim = vae.cfg.decoder_hidden_dim
        self.vocab_size = vae.cfg.spm_vocab_size + 2
        self.bos_id = vae._bos_id
        self.eos_id = vae._eos_id

        # ---- z-generator ----
        z_mean = getattr(vae, "_z_struct_mean", None)
        z_std = getattr(vae, "_z_struct_std", None)
        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=self.latent_dim,
            d_model=config.z_hidden_dim,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            variable_phrases=config.variable_phrases,
            z_mean=z_mean,
            z_std=z_std,
            target_phrases=config.target_phrases,
        )

        # ---- contrastive heads ----
        self.target_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.hidden_proj = nn.Linear(self.hidden_dim, config.embedding_dim)
        self.surface_encoder = MessageEncoder(
            self.hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / config.contrastive_temperature)),
        )

        # ---- z-diversity (intra-expression hinge) ----
        if config.z_diversity_weight > 0 and z_mean is not None and z_std is not None:
            self.z_diversity = ZDiversityLoss(z_mean, z_std)
        else:
            self.z_diversity = None

        # ---- precomputed top-K corpus bigrams for bigram_kl ----
        self._bigram_loaded = False
        if config.bigram_kl_weight > 0 and config.bigram_kl_path:
            path = Path(config.bigram_kl_path)
            if path.exists():
                with np.load(path) as f:
                    pairs = f["pairs"].astype(np.int64)
                    probs = f["probs"].astype(np.float32)
                    oov = float(f["oov_prob"])
                self.register_buffer(
                    "bigram_pairs", torch.tensor(pairs, dtype=torch.long),
                )  # (K, 2)
                self.register_buffer(
                    "bigram_probs",
                    torch.tensor(probs, dtype=torch.float32).clamp(min=1e-8),
                )  # (K,)
                self.register_buffer(
                    "bigram_oov_prob",
                    torch.tensor(max(oov, 1e-8), dtype=torch.float32),
                )
                self._bigram_loaded = True
                logger.info(
                    "Loaded bigram top-%d from %s (covers %.2f%% of mass)",
                    pairs.shape[0], path, (1.0 - oov) * 100,
                )
            else:
                logger.warning(
                    "bigram_kl_path %s missing — bigram_kl disabled", path,
                )

        # ---- frozen LLM scorer ----
        self.llm_pressure: LLMPressureScorer | None = None
        if config.llm_pressure_weight > 0:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=config.spm_path)
            self.llm_pressure = LLMPressureScorer(
                spm_model=sp,
                spm_vocab_size=self.vocab_size,
                llm_model_name=config.llm_model_name,
            )

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    def render_surface(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
        eos_id: int | None = None,
        output_mode: str = "ipa",
    ) -> list[str]:
        """Decode token IDs → IPA strings via the dep_tree_vae's SPM.

        Picked up by ``AgentTrainer`` at every checkpoint to print live
        samples.  The trainer's ``_respell_ipa`` then converts each
        string to a Latin-respelled form for readability.  This lets us
        watch linguistic quality (cycles, well-formedness) per
        checkpoint and short-circuit a degenerate run instead of
        burning GPU hours on it.
        """
        if not hasattr(self, "_sp"):
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(
                model_file=self.config.spm_path,
            )
        sp = self._sp
        spm_size = sp.get_piece_size()
        eos = eos_id if eos_id is not None else self.eos_id
        out: list[str] = []
        ids_t = token_ids.detach().cpu().tolist()
        if mask is not None:
            mask_t = mask.detach().cpu().tolist()
        else:
            mask_t = [[True] * len(row) for row in ids_t]
        for row, mrow in zip(ids_t, mask_t):
            ids = [int(t) for t, m in zip(row, mrow) if m and int(t) < spm_size and int(t) != eos]
            out.append(sp.decode(ids).strip())
        return out

    def trainable_param_groups(self) -> list[dict]:
        """Per-group LR. Frozen vae params are excluded."""
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
        """Slim checkpoint dict — vae weights live in their own checkpoint."""
        state = {
            "z_gen": self.z_gen.state_dict(),
            "target_proj": self.target_proj.state_dict(),
            "hidden_proj": self.hidden_proj.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "version": 5,
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
    # Decode helpers — sit inside the game class because their interface
    # is dep_tree_vae-specific and not reused elsewhere.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _skeleton_to_roles(self, z: Tensor, max_roles: int) -> Tensor:
        """Greedy skeleton decode → padded role tensor (Bz, R).

        Each row is the role sequence stripped of BOS/EOS, padded with
        the last role to ``max_roles`` (so all rows share shape).
        """
        skel = self.vae.skeleton_decoder(z)[0]  # (Bz, R+1)
        Bz = skel.size(0)
        out = torch.zeros(Bz, max_roles, dtype=torch.long, device=z.device)
        for i in range(Bz):
            roles: list[int] = []
            for t in skel[i]:
                tv = t.item()
                if tv == SKEL_BOS:
                    continue
                if tv == SKEL_EOS:
                    break
                if tv < len(DEP_RELATIONS):
                    roles.append(tv)
            if not roles:
                roles = [DEP_RELATIONS.index("root")]
            roles = roles[:max_roles]
            pad = roles[-1]
            roles += [pad] * (max_roles - len(roles))
            out[i] = torch.tensor(roles, device=z.device)
        return out

    def _block_repeated_ngrams(
        self, logits: Tensor, generated: Tensor, t: int,
    ) -> Tensor:
        """Set ``logits[b, banned] = -inf`` for any token that would
        complete a previously-emitted n-gram, for n in ``cfg.ngram_block``.

        Vectorized: for each n, find positions in the history whose
        (n-1)-gram matches the current trailing (n-1)-gram; the token
        right after that match is the banned next token.
        """
        ns = self.config.ngram_block or []
        if not ns or t == 0:
            return logits
        Bz = logits.size(0)
        for n in ns:
            if t < n - 1:
                continue
            prefix = generated[:, t - (n - 1):t]            # (Bz, n-1)
            if t < n - 1 + 1:
                continue
            grams = generated[:, : t].unfold(1, n - 1, 1)   # (Bz, t-(n-2), n-1)
            matches = (grams == prefix.unsqueeze(1)).all(dim=-1)  # (Bz, M)
            # The banned next-token after each match position j is at
            # generated[:, j + (n-1)]. Valid only when j + (n-1) < t.
            M = matches.size(1)
            j = torch.arange(M, device=logits.device)
            valid_j = (j + (n - 1)) < t
            matches = matches & valid_j.unsqueeze(0)
            if not matches.any():
                continue
            b_idx, j_idx = matches.nonzero(as_tuple=True)
            banned = generated[b_idx, j_idx + (n - 1)]
            logits[b_idx, banned] = float("-inf")
        return logits

    @torch.no_grad()
    def _ar_decode(self, memory: Tensor, max_len: int) -> tuple[Tensor, Tensor]:
        """Batched greedy AR token decode through frozen phrase_decoder.

        Includes n-gram blocking (per ``cfg.ngram_block``) at every step
        so cycles are prevented at sample time, not just penalized in the
        loss. Phase 2 grad rerun then sees a clean, non-cyclic target.

        Args:
            memory: ``(Bz, R, H)`` per-role memory.
            max_len: Hard cap on tokens per phrase (excluding BOS/EOS).

        Returns:
            tokens: ``(Bz, T)`` token ids (no BOS, EOS truncated).
            mask: ``(Bz, T)`` boolean (True = valid).
        """
        cfg = self.vae.cfg
        Bz, _, _ = memory.shape
        device = memory.device

        # Ensure RoPE table is long enough.
        rope = self.vae._rope_freqs
        if rope is not None and rope.size(0) < max_len + 1:
            rope = precompute_rope_freqs(
                self.hidden_dim // cfg.decoder_num_heads,
                max_len + 2, device=device,
            )
            self.vae._rope_freqs = rope

        tokens = torch.full((Bz, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(Bz, dtype=torch.bool, device=device)
        generated = torch.zeros(Bz, max_len, dtype=torch.long, device=device)
        gen_mask = torch.zeros(Bz, max_len, dtype=torch.bool, device=device)

        for t in range(max_len):
            tok_emb = self.vae.dec_token_embedding(tokens)
            seq_len = tok_emb.size(1)
            tgt_mask = multiscale_causal_mask(
                seq_len, cfg.decoder_num_heads,
                tuple(cfg.attention_head_windows),
                cfg.attention_global_every, device=device,
            )
            rope_t = rope[:seq_len] if rope is not None else None
            hidden = self.vae.phrase_decoder(
                tok_emb, memory, tgt_mask=tgt_mask, rope_freqs=rope_t,
            )
            logits = self.vae.output_head(hidden[:, -1, :])
            logits = self._block_repeated_ngrams(logits, generated, t)
            next_tok = logits.argmax(dim=-1)

            # Record for non-finished rows; freeze finished rows.  EOS is
            # itself a valid token and stays in the mask — excluding it
            # produced all-False masks for samples that EOS'd at t=0,
            # which then NaN'd the surface encoder's attention softmax.
            active = ~finished
            generated[:, t] = torch.where(active, next_tok, generated[:, t])
            gen_mask[:, t] = active

            # Mark newly-finished.
            finished = finished | (next_tok == self.eos_id)

            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                # Trim trailing zeros.
                gen_mask = gen_mask[:, : t + 1]
                generated = generated[:, : t + 1]
                break

        return generated, gen_mask

    def _phrase_hidden_with_grad(
        self, z: Tensor, tokens: Tensor, mask: Tensor, max_roles: int,
    ) -> Tensor:
        """Re-run phrase_decoder teacher-forced on (z, tokens) with grad.

        Memory is recomputed from ``z`` (which carries grad), so the
        gradient flows: hidden → memory → phrase_projector → z.
        ``phrase_decoder`` itself is frozen.
        """
        cfg = self.vae.cfg
        device = z.device

        with torch.no_grad():
            roles = self._skeleton_to_roles(z, max_roles)

        # Memory: gradients flow through phrase_projector(z, ...).
        memory = self.vae.phrase_projector(z, roles)  # (B, R, H)

        # Prepend BOS for teacher-forced input; targets start at position 1.
        T = tokens.size(1)
        bos = torch.full((tokens.size(0), 1), self.bos_id, dtype=torch.long, device=device)
        input_ids = torch.cat([bos, tokens[:, :-1]], dim=1)  # (B, T)
        tok_emb = self.vae.dec_token_embedding(input_ids)

        rope = self.vae._rope_freqs
        rope_t = rope[:T] if rope is not None else None
        tgt_mask = multiscale_causal_mask(
            T, cfg.decoder_num_heads,
            tuple(cfg.attention_head_windows),
            cfg.attention_global_every, device=device,
        )
        hidden = self.vae.phrase_decoder(
            tok_emb, memory, tgt_mask=tgt_mask, rope_freqs=rope_t,
        )
        return hidden

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, anchor: Tensor) -> ExpressionOutput:
        """Encode anchor → K z-vectors → K decoded phrases → concat."""
        cfg = self.config
        B = anchor.size(0)
        max_roles = self.vae.cfg.skeleton.max_roles

        conditioning = self.target_proj(anchor)
        z_seq, z_weights, num_phrases = self.z_gen(conditioning)  # (B, K, D)
        K = z_seq.size(1)
        z_flat = z_seq.reshape(B * K, self.latent_dim)

        # Phase 1 — no-grad batched skeleton + AR decode.
        with torch.no_grad():
            roles_flat = self._skeleton_to_roles(z_flat, max_roles)
            memory_flat = self.vae.phrase_projector(z_flat, roles_flat)
            tokens_flat, mask_flat = self._ar_decode(
                memory_flat, cfg.max_tokens_per_phrase,
            )
        T = tokens_flat.size(1)

        # Phase 2 — chunked grad re-run. The per-phrase decode is small
        # enough that we can chunk over (B*K) without much overhead.
        chunk = max(cfg.phase2_chunk, 1)
        hidden_chunks: list[Tensor] = []
        for s in range(0, B * K, chunk):
            e = min(s + chunk, B * K)
            h = self._phrase_hidden_with_grad(
                z_flat[s:e], tokens_flat[s:e], mask_flat[s:e], max_roles,
            )
            hidden_chunks.append(h)
        hidden_flat = torch.cat(hidden_chunks, dim=0)  # (B*K, T, H)

        # Reshape and concat K phrases along seq dim → one expression per anchor.
        hidden = hidden_flat.reshape(B, K * T, self.hidden_dim)
        mask = mask_flat.reshape(B, K * T)
        tokens = tokens_flat.reshape(B, K * T)

        # Compute output logits + softmax ONCE — at B=256, S=320, V=8050
        # this tensor is ~6.6 GB fp32, and was previously being allocated
        # 3-4 times (in _bigram_kl, _adj_diversity, _surface_repr,
        # _llm_pressure) during the same forward, which OOM'd the 24 GB
        # GPU and forced auto-shrink to bs=63.  Cache once on the
        # ExpressionOutput so all loss callers reuse the same tensor.
        logits = self.vae.output_head(hidden)
        probs = F.softmax(logits, dim=-1)

        return ExpressionOutput(
            hidden=hidden,
            logits=logits,
            probs=probs,
            mask=mask,
            z_seq=z_seq,
            z_weights=z_weights,
            num_phrases=num_phrases,
            tokens_cpu=tokens.detach().cpu(),
            gen_mask_cpu=mask.detach().cpu(),
        )

    # ------------------------------------------------------------------
    # Loss terms — pure functions over (expr, anchor); zero-weight terms
    # short-circuit to a zero scalar so they're free.
    # ------------------------------------------------------------------

    def _pool_hidden(self, expr: ExpressionOutput) -> Tensor:
        """Mask-mean pool over the (B, S, H) hidden states → (B, H)."""
        m = expr.mask.unsqueeze(-1).float()
        lengths = expr.mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (expr.hidden * m).sum(dim=1) / lengths

    def _surface_repr(self, expr: ExpressionOutput) -> Tensor:
        """Straight-through-embed tokens, encode → (B, embedding_dim).

        Reuses the cached logits/probs from ``expr`` (computed once in
        ``_generate``) instead of re-running ``output_head(hidden)``.

        Patches any all-False rows in the mask to mark position 0 as
        valid — an all-False row would NaN the surface encoder's
        attention softmax and contaminate the whole batch.
        """
        # Straight-through: forward = hard-onehot, backward = soft probs.
        soft = expr.probs                                          # (B, S, V)
        hard = F.one_hot(expr.logits.argmax(dim=-1), soft.size(-1)).to(soft.dtype)
        st = (hard - soft).detach() + soft                          # (B, S, V)
        embedded = st @ self.vae.dec_token_embedding.weight         # (B, S, H)

        mask = expr.mask
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True
        return self.surface_encoder(embedded, mask)

    def _info_nce(self, msg: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor]:
        """Symmetric InfoNCE; returns (loss, msg→tgt similarity logits)."""
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        sim = msg_n @ tgt_n.t()
        logits = sim / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return loss, logits

    def _topology_loss(self, msg: Tensor, anchor: Tensor) -> Tensor:
        """1 − Pearson ρ between message-pairwise and target-pairwise cosine."""
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
        """Hinge penalty on intra-expression z cosine."""
        if self.z_diversity is None:
            return expr.hidden.new_tensor(0.0)
        loss, _ = self.z_diversity(expr.z_seq, expr.z_weights)
        return loss

    def _bigram_kl_loss(self, expr: ExpressionOutput) -> Tensor:
        """KL(batch-marginal bigram || top-K corpus bigrams) + OOV.

        Counts only adjacent positions where both tokens are valid.
        Computes the model's expected mass on each of the K precomputed
        corpus bigrams plus an OOV bucket for everything else, then
        KL-divergences against the corpus distribution.

        Catches the unigram-loophole: cycling on `the/of/with` matches
        a unigram target but produces (`the`,`the`)-style bigrams
        absent from the corpus — the OOV bucket inflates and pushes
        back through softmax → logits → memory → z_gen.
        """
        if not self._bigram_loaded:
            return expr.hidden.new_tensor(0.0)
        # Reuses cached probs from expr.  The naive joint = p_t * p_t1
        # of shape (B, S-1, K) at B=256, S=320, K=50000 is huge, so we
        # chunk along the batch dim and accumulate.
        p_t_full = expr.probs[:, :-1]                         # (B, S-1, V)
        p_t1_full = expr.probs[:, 1:]
        pair_mask_full = (expr.mask[:, :-1] & expr.mask[:, 1:]).float()
        n_pairs = pair_mask_full.sum().clamp(min=1)

        a = self.bigram_pairs[:, 0]
        b = self.bigram_pairs[:, 1]
        K = a.numel()

        B = p_t_full.size(0)
        # Pick a chunk whose joint tensor is bounded.  Target ~256 MB:
        # chunk × (S-1) × K × 4B ≤ 256 MB  →  chunk ≤ 256e6 / (80 × K × 4).
        # For K=50000, S=80, this caps chunk at ~16. Keep chunk modest.
        chunk = max(1, min(B, int(64_000_000 // max(K * p_t_full.size(1) // 32, 1))))

        sum_topK = p_t_full.new_zeros(K)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            jt = p_t_full[s:e, :, a] * p_t1_full[s:e, :, b]   # (c, S-1, K)
            sum_topK = sum_topK + (jt * pair_mask_full[s:e].unsqueeze(-1)).sum(dim=(0, 1))
            del jt

        model_topK = (sum_topK / n_pairs).clamp(min=1e-12)
        sum_top = model_topK.sum().clamp(max=1.0 - 1e-8)
        model_oov = (1.0 - sum_top).clamp(min=1e-12)

        kl_top = (model_topK * (model_topK.log() - self.bigram_probs.log())).sum()
        kl_oov = model_oov * (model_oov.log() - self.bigram_oov_prob.log())
        return kl_top + kl_oov

    def _adj_diversity_loss(self, expr: ExpressionOutput) -> Tensor:
        """Hinge on cosine similarity between adjacent softmax distributions.

        cos(p_t, p_t+1) is high when the model wants to emit similar
        token distributions at consecutive positions — exactly the
        local mode of cycling. Penalizing values above
        ``adj_diversity_target`` (default 0.30) is direct anti-cycling
        pressure, no corpus reference needed; complements bigram_kl
        which catches the same failure at the post-softmax level.
        """
        cfg = self.config
        if cfg.adj_diversity_weight <= 0:
            return expr.hidden.new_tensor(0.0)
        pn = F.normalize(expr.probs, dim=-1, eps=1e-8)
        cos = (pn[:, :-1] * pn[:, 1:]).sum(dim=-1)            # (B, S-1)
        pair_mask = (expr.mask[:, :-1] & expr.mask[:, 1:]).float()
        excess = (cos - cfg.adj_diversity_target).clamp(min=0.0)
        return (excess * pair_mask).sum() / pair_mask.sum().clamp(min=1)

    def _llm_pressure_loss(self, expr: ExpressionOutput) -> Tensor:
        """Frozen Qwen NLL over agent logits."""
        if self.llm_pressure is None:
            return expr.hidden.new_tensor(0.0)
        return self.llm_pressure(
            agent_logits=expr.logits, mask=expr.mask, tau=self.config.llm_gumbel_tau,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, anchor: Tensor, distractors: Tensor,
        *, step: int = 0, candidate_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """One contrastive step.

        Distractors are kept in the signature for ``AgentTrainer``
        compatibility but unused — InfoNCE uses in-batch negatives.
        """
        cfg = self.config
        del distractors, candidate_indices, step

        expr = self._generate(anchor)

        hidden_msg = self.hidden_proj(self._pool_hidden(expr))
        surface_msg = self._surface_repr(expr)

        terms: dict[str, Tensor] = {}
        terms["hidden_nce"], hidden_logits = self._info_nce(hidden_msg, anchor)
        terms["surface_nce"], _ = self._info_nce(surface_msg, anchor)
        terms["topology"] = self._topology_loss(surface_msg, anchor)
        terms["z_div_loss"] = self._z_diversity_loss(expr)
        terms["bigram_kl"] = self._bigram_kl_loss(expr)
        terms["adj_div"] = self._adj_diversity_loss(expr)
        terms["llm_pressure"] = self._llm_pressure_loss(expr)

        weights = {
            "hidden_nce":   cfg.hidden_infonce_weight,
            "surface_nce":  cfg.surface_infonce_weight,
            "topology":     cfg.topology_weight,
            "z_div_loss":   cfg.z_diversity_weight,
            "bigram_kl":    cfg.bigram_kl_weight,
            "adj_div":      cfg.adj_diversity_weight,
            "llm_pressure": cfg.llm_pressure_weight,
        }

        # Per-term NaN/Inf guard — names the first culprit so we can
        # tune that one specifically rather than killing all six.
        for k, t in terms.items():
            if torch.isnan(t) or torch.isinf(t):
                logger.error(
                    "Non-finite loss term: %s = %s. Other terms: %s",
                    k, t.item(),
                    {k2: float(v.detach()) for k2, v in terms.items() if k2 != k},
                )
                raise RuntimeError(f"NaN/Inf in loss term '{k}'")

        total = sum(weights[k] * terms[k] for k in weights)

        with torch.no_grad():
            target_idx = torch.arange(hidden_logits.size(0), device=hidden_logits.device)
            accuracy = (hidden_logits.argmax(1) == target_idx).float().mean()
            msg_lengths = expr.mask.float().sum(dim=1).mean()
            seqs = set()
            eos = self.eos_id
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
        for k, v in terms.items():
            out[k] = v.detach()
        return out
