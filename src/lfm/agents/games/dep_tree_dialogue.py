"""DepTree Dialogue Game — multi-turn self-play with disentangled structure/content.

Architecture:
  - **StructureGen**: embedding → z_struct (once per dialogue, shared grammar).
  - **ContentGen** (DiffusionZGenerator): embedding + context → z_content (per turn).
  - **Skeleton decoder**: z_struct → dependency role sequence (frozen).
  - **PhraseProjector**: z_content + roles → per-role decoder memory (frozen, WITH grad).
  - **Scoring**: attention-pooled role memory → contrastive InfoNCE + topology.
  - **Diagnostic decode**: KV-cached multi-role decode (one short phrase per role,
    EOS between roles, KV cache persists for cross-role coherence).

Scoring operates on role memory projections, not decoded tokens. This gives
direct gradient flow to the z-generators without requiring a decoder rerun.
The multi-role decode runs only at checkpoint steps for diagnostic logging.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.agents.config import CurriculumConfig
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.config import LFMBaseConfig
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS, DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
from lfm.generator.layers import multiscale_causal_mask, precompute_rope_freqs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DepTreeDialogueConfig(LFMBaseConfig):
    """Configuration for the DepTree dialogue game."""

    # Paths
    vae_checkpoint: str = ""
    spm_path: str = ""
    vae_dataset_path: str = ""
    output_dir: str = ""
    embedding_store_dir: str = ""
    embedding_dim: int = 896

    # VAE architecture (must match checkpoint)
    latent_total_dim: int = 256
    latent_struct_dim: int = 64
    latent_content_dim: int = 192
    spm_vocab_size: int = 8000
    vae_max_seq_len: int = 160

    # Dialogue
    num_turns: int = 4

    # Content z-generator (per-turn diffusion flow-matching)
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    content_gen_hidden: int = 512

    # Structure z-generator (one-shot MLP)
    struct_gen_hidden: int = 256

    # Contrastive scoring
    contrastive_scoring: bool = True
    contrastive_temperature: float = 0.07
    topo_weight: float = 1.0

    # Decode — multi-role: one short phrase per skeleton role
    max_tokens_per_role: int = 20

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    steps: int = 6000
    struct_lr: float = 5e-5
    content_lr: float = 5e-5
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    num_distractors: int = 15
    min_targets: int = 1
    max_targets: int = 1

    # Curriculum
    curriculum: CurriculumConfig = CurriculumConfig(warmup_steps=1875)

    # Runtime
    device: str = "cuda"
    seed: int = 42
    log_every: int = 20
    checkpoint_every: int = 200


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------

class StructureGen(nn.Module):
    """Embedding → z_struct (one shot per dialogue, determines grammar)."""

    def __init__(self, input_dim: int, struct_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, struct_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.net(embedding)


class MemoryPooler(nn.Module):
    """Learned-query attention pooler over variable-length role memories.

    A single learned query attends to the per-role memory vectors,
    producing a fixed-size summary that preserves role-specific
    information in the gradient (unlike mean pooling which rewards
    uniform roles and causes repetitive decoder output).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, memory: Tensor, role_mask: Tensor) -> Tensor:
        """Pool (B, R, H) memory with (B, R) mask → (B, H)."""
        b = memory.size(0)
        query = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(query, memory, memory, key_padding_mask=~role_mask)
        return pooled.squeeze(1)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class DepTreeDialogueGame(nn.Module):
    """Multi-turn dialogue game with disentangled structure/content.

    Trainer interface: ``forward(anchor, distractors) → dict``,
    ``trainable_param_groups()``, ``checkpoint_state()``,
    ``load_checkpoint_state()``.
    """

    def __init__(self, config: DepTreeDialogueConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config

        self._load_vae(cfg)

        hdim = self.vae_cfg.decoder_hidden_dim

        self.struct_gen = StructureGen(
            cfg.embedding_dim, cfg.latent_struct_dim, cfg.struct_gen_hidden,
        )
        self.content_gen = DiffusionZGenerator(
            input_dim=cfg.embedding_dim,
            latent_dim=cfg.latent_content_dim,
            d_model=cfg.content_gen_hidden,
            max_phrases=1,
            num_steps=cfg.diffusion_steps,
            num_layers=cfg.diffusion_layers,
            num_heads=cfg.diffusion_heads,
            variable_phrases=False,
        )

        self.turn_embeddings = nn.Parameter(
            _simplex_init(cfg.num_turns, cfg.embedding_dim),
        )
        self.memory_pooler = MemoryPooler(hdim)
        self.struct_to_memory = nn.Linear(cfg.latent_struct_dim, hdim)
        self.hidden_to_context = nn.Linear(hdim, cfg.embedding_dim)
        self.context_proj = nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim)
        self.message_projector = nn.Linear(hdim, cfg.embedding_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / cfg.contrastive_temperature)),
        )
        self.turn_agg_logits = nn.Parameter(torch.zeros(cfg.num_turns))

        self._causal_mask: Tensor | None = None

    # ---- VAE loading ----

    def _load_vae(self, cfg: DepTreeDialogueConfig) -> None:
        ckpt = torch.load(cfg.vae_checkpoint, map_location="cpu", weights_only=False)
        if "model_state" not in ckpt:
            raise ValueError("Expected model_state in VAE checkpoint")

        self.vae_cfg = DepTreeVAEConfig(
            spm_vocab_size=cfg.spm_vocab_size,
            max_seq_len=cfg.vae_max_seq_len,
            latent={
                "total_dim": cfg.latent_total_dim,
                "struct_dim": cfg.latent_struct_dim,
                "content_dim": cfg.latent_content_dim,
            },
        )
        self.vae = DepTreeVAE(
            self.vae_cfg, cfg.spm_vocab_size + 2 + len(DEP_RELATIONS),
        )
        self.vae.load_state_dict(ckpt["model_state"], strict=False)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(cfg.spm_path)
        logger.info("Loaded frozen DepTreeVAE from %s", cfg.vae_checkpoint)

        self._init_z_calibration(cfg)

    # ---- Z calibration ----

    def _init_z_calibration(self, cfg: DepTreeDialogueConfig) -> None:
        """Load z distribution stats from the VAE checkpoint or compute them.

        The decoder expects z from the encoder's distribution. These stats
        let _calibrate_z normalize the z-generators' output to match.
        """
        vae = self.vae

        # Try checkpoint buffers first (stored during VAE training).
        # Uninitialized buffers have std=1.0 and mean=0.0 — skip them.
        has_stats = (
            hasattr(vae, "_z_struct_mean")
            and vae._z_struct_mean.abs().sum().item() > 0.01
        )
        if has_stats:
            self.register_buffer("_z_struct_mean", vae._z_struct_mean.clone())
            self.register_buffer("_z_struct_std", vae._z_struct_std.clone())
            self.register_buffer("_z_content_mean", vae._z_content_mean.clone())
            self.register_buffer("_z_content_std", vae._z_content_std.clone())
            logger.info("Z calibration from checkpoint buffers")
        elif cfg.vae_dataset_path:
            self._compute_z_stats_from_data(cfg)
        else:
            logger.warning("No z stats available — calibration uses N(0,1) defaults")
            self.register_buffer("_z_struct_mean", torch.zeros(cfg.latent_struct_dim))
            self.register_buffer("_z_struct_std", torch.ones(cfg.latent_struct_dim))
            self.register_buffer("_z_content_mean", torch.zeros(cfg.latent_content_dim))
            self.register_buffer("_z_content_std", torch.ones(cfg.latent_content_dim))

        logger.info(
            "Z stats: struct_norm=%.2f struct_std=%.4f, content_norm=%.2f content_std=%.4f",
            self._z_struct_mean.norm(), self._z_struct_std.mean(),
            self._z_content_mean.norm(), self._z_content_std.mean(),
        )

    @torch.no_grad()
    def _compute_z_stats_from_data(self, cfg: DepTreeDialogueConfig) -> None:
        """Fallback: compute z stats by encoding a sample of real sentences."""
        from pathlib import Path as _P
        from lfm.generator.dep_tree_vae.data import DepTreeDataset

        cache_dir = _P(cfg.vae_dataset_path) / "cache"
        if not cache_dir.exists():
            logger.warning("No cache at %s — using N(0,1) defaults", cache_dir)
            self.register_buffer("_z_struct_mean", torch.zeros(cfg.latent_struct_dim))
            self.register_buffer("_z_struct_std", torch.ones(cfg.latent_struct_dim))
            self.register_buffer("_z_content_mean", torch.zeros(cfg.latent_content_dim))
            self.register_buffer("_z_content_std", torch.ones(cfg.latent_content_dim))
            return

        ds = DepTreeDataset(cache_dir)
        n = min(512, len(ds))
        samples = [ds[i] for i in range(n)]

        # Pad interleaved sequences into a batch
        max_len = min(max(len(s["interleaved"]) for s in samples), cfg.vae_max_seq_len)
        tokens = torch.zeros(n, max_len, dtype=torch.long)
        lengths = torch.zeros(n, dtype=torch.long)
        for i, s in enumerate(samples):
            seq = torch.tensor(s["interleaved"][:max_len])
            tokens[i, :len(seq)] = seq
            lengths[i] = len(seq)

        mu, logvar = self.vae.encoder(tokens, lengths)
        z = self.vae.latent.reparameterize(mu, logvar)
        z_struct, z_content = self.vae.latent.split(z)

        self.register_buffer("_z_struct_mean", z_struct.mean(dim=0))
        self.register_buffer("_z_struct_std", z_struct.std(dim=0).clamp(min=1e-6))
        self.register_buffer("_z_content_mean", z_content.mean(dim=0))
        self.register_buffer("_z_content_std", z_content.std(dim=0).clamp(min=1e-6))
        logger.info("Z calibration computed from %d samples", n)

    def _calibrate_z(self, z: Tensor, target_mean: Tensor, target_std: Tensor) -> Tensor:
        """Per-dimension calibration: match z's marginals to encoder distribution."""
        if z.size(0) < 2:
            return z
        obs_mean = z.mean(dim=0, keepdim=True)
        obs_std = z.std(dim=0, keepdim=True).clamp(min=1e-6)
        return (z - obs_mean) / obs_std * target_std + target_mean

    # ---- Skeleton parsing ----

    @torch.no_grad()
    def _parse_skeleton(
        self, z_struct: Tensor,
    ) -> tuple[Tensor, Tensor, list[list[str]]]:
        """z_struct → padded role IDs, mask, and human-readable skeleton strings."""
        b = z_struct.size(0)
        device = z_struct.device
        num_dep = len(DEP_RELATIONS)
        root_id = DEP_RELATIONS.index("root")

        skel_tokens = self.vae.skeleton_decoder(z_struct)[0]

        is_valid = (skel_tokens != SKEL_BOS) & (skel_tokens != SKEL_EOS) & (skel_tokens < num_dep)
        eos_pos = torch.where(
            skel_tokens == SKEL_EOS,
            torch.arange(skel_tokens.size(1), device=device),
            skel_tokens.size(1),
        )
        first_eos = eos_pos.min(dim=1).values
        is_valid = is_valid & (torch.arange(skel_tokens.size(1), device=device) < first_eos.unsqueeze(1))

        max_r = max(is_valid.sum(dim=1).max().item(), 1)
        padded_roles = torch.full((b, max_r), root_id, dtype=torch.long, device=device)
        role_mask = torch.zeros(b, max_r, dtype=torch.bool, device=device)
        skeletons: list[list[str]] = []

        for i in range(b):
            idx = is_valid[i].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                padded_roles[i, 0] = root_id
                role_mask[i, 0] = True
                skeletons.append(["root"])
            else:
                padded_roles[i, :n] = skel_tokens[i, idx]
                role_mask[i, :n] = True
                skeletons.append([DEP_RELATIONS[t.item()] for t in skel_tokens[i, idx]])

        return padded_roles, role_mask, skeletons

    # ---- Memory projection (differentiable) ----

    def _project_and_pool(
        self, z_struct: Tensor, z_content: Tensor,
        padded_roles: Tensor, role_mask: Tensor,
    ) -> Tensor:
        """z_struct + z_content + roles → attention-pooled memory (B, H).

        z_struct is projected and prepended to the role memories so that
        both generators receive gradient from the contrastive loss.
        Gradient flows: loss → pooler → phrase_projector → z_content,
        and loss → pooler → struct_proj → z_struct.
        """
        memory = self.vae.phrase_projector(z_content, padded_roles, role_mask)
        # Prepend a z_struct summary token so struct_gen gets gradient
        struct_token = self.struct_to_memory(z_struct).unsqueeze(1)  # (B, 1, H)
        memory = torch.cat([struct_token, memory], dim=1)
        # Extend mask for the struct token (always valid)
        struct_mask = torch.ones(z_struct.size(0), 1, dtype=torch.bool, device=z_struct.device)
        full_mask = torch.cat([struct_mask, role_mask], dim=1)
        return self.memory_pooler(memory, full_mask)

    # ---- Multi-role decode (diagnostic only) ----

    @torch.no_grad()
    def _decode_multirole(
        self, z_content: Tensor, padded_roles: Tensor, role_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """KV-cached autoregressive decode: one short phrase per skeleton role.

        The KV cache persists across roles for cross-role coherence.
        Each role gets its own memory from the phrase projector.
        Decoding stops per-role on EOS or max_tokens_per_role.

        Returns:
            tokens: (B, S) generated token IDs.
            token_mask: (B, S) bool validity mask.
        """
        vae = self.vae
        cfg = self.vae_cfg
        b = z_content.size(0)
        device = z_content.device
        max_per_role = self.config.max_tokens_per_role
        num_roles = padded_roles.size(1)
        nheads = cfg.decoder_num_heads

        # Per-role memory: (B, R, mem_tokens, H) — each role gets 1 memory vec
        all_memory = vae.phrase_projector(z_content, padded_roles, role_mask)

        max_total = max_per_role * num_roles
        decoder = vae.phrase_decoder

        # Precompute causal mask
        if self._causal_mask is None or self._causal_mask.size(1) < max_total + 1:
            self._causal_mask = multiscale_causal_mask(
                max_total + 1, nheads,
                tuple(cfg.attention_head_windows),
                cfg.attention_global_every,
                device=device,
            )
        rope = vae._rope_freqs
        if rope is not None and rope.size(0) < max_total + 1:
            rope = precompute_rope_freqs(
                cfg.decoder_hidden_dim // nheads, max_total + 1, device=device,
            )

        kv_cache = decoder.make_kv_cache(b, max_total + 1, device, dtype=torch.float32)

        # Output buffers
        tokens = torch.zeros(b, max_total, dtype=torch.long, device=device)
        token_mask = torch.zeros(b, max_total, dtype=torch.bool, device=device)
        batch_idx = torch.arange(b, device=device)

        # Per-sample state
        current_role = torch.zeros(b, dtype=torch.long, device=device)
        tokens_in_role = torch.zeros(b, dtype=torch.long, device=device)
        total_pos = torch.zeros(b, dtype=torch.long, device=device)
        role_counts = role_mask.sum(dim=1)  # (B,) actual roles per sample
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        def _current_memory() -> Tensor:
            idx = current_role.clamp(max=num_roles - 1)
            return all_memory[batch_idx, idx].unsqueeze(1)  # (B, 1, H)

        # Prime with BOS
        bos_embed = vae.dec_token_embedding(
            torch.full((b, 1), vae._bos_id, dtype=torch.long, device=device),
        )
        mask_row = self._causal_mask[:, 0:1, 0:1]
        out = decoder.forward_cached(
            bos_embed, _current_memory(), kv_cache,
            rope_freqs=rope, tgt_mask_row=mask_row,
        )
        kv_cache.advance()

        for _ in range(max_total):
            next_tok = vae.output_head(out[:, -1]).argmax(dim=-1)  # (B,)
            active = ~finished

            # Store token
            tokens[batch_idx, total_pos] = next_tok * active.long()
            token_mask[batch_idx, total_pos] = active & (next_tok != vae._eos_id)
            total_pos += active.long()
            tokens_in_role += active.long()

            # Role switching: EOS or max tokens per role
            hit_eos = (next_tok == vae._eos_id) & active
            hit_max = (tokens_in_role >= max_per_role) & active
            should_switch = hit_eos | hit_max

            current_role += should_switch.long()
            tokens_in_role *= ~should_switch

            finished = finished | (current_role >= role_counts)
            if finished.all():
                break

            new_embed = vae.dec_token_embedding(next_tok.unsqueeze(1))
            seq_so_far = kv_cache.seq_len + 1
            mask_row = self._causal_mask[
                :, kv_cache.seq_len:kv_cache.seq_len + 1, :seq_so_far
            ]
            out = decoder.forward_cached(
                new_embed, _current_memory(), kv_cache,
                rope_freqs=rope, tgt_mask_row=mask_row,
            )
            kv_cache.advance()

        # Trim to actual max length
        max_len = total_pos.max().item()
        return tokens[:, :max_len], token_mask[:, :max_len]

    # ---- Surface rendering ----

    def render_surface(self, token_ids: Tensor, mask: Tensor | None = None) -> list[str]:
        """Decode token IDs to IPA strings via sentencepiece."""
        sp = self._sp
        bos, eos = self.vae._bos_id, self.vae._eos_id
        results = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].tolist()
            if mask is not None:
                ids = [t for t, v in zip(ids, mask[i].tolist()) if v]
            ids = [t for t in ids if t not in (bos, eos, 0) and t < sp.GetPieceSize()]
            results.append(sp.DecodeIds(ids))
        return results

    # ---- Forward (training) ----

    def forward(
        self, anchor: Tensor, distractors: Tensor | None = None, **kwargs,
    ) -> dict[str, Tensor]:
        b, d = anchor.shape
        device = anchor.device
        cfg = self.config
        step = kwargs.get("step", 0)
        do_decode = (step % cfg.checkpoint_every == 0)

        # 1. Structure (shared across turns)
        z_struct_raw = self.struct_gen(anchor)
        z_struct = self._calibrate_z(z_struct_raw, self._z_struct_mean, self._z_struct_std)
        padded_roles, role_mask, skeletons = self._parse_skeleton(z_struct)

        # 2. Per-turn content generation + memory scoring
        turn_surfaces: list[Tensor] = []
        turn_tokens: list[Tensor] = []
        turn_masks: list[Tensor] = []
        context = torch.zeros(b, d, device=device)

        for t in range(cfg.num_turns):
            turn_emb = self.turn_embeddings[t].unsqueeze(0).expand(b, -1)
            cond = self.context_proj(torch.cat([anchor + turn_emb, context], dim=-1))

            z_content_raw = self.content_gen(cond)[0][:, 0, :]  # (B, content_dim)
            z_content = self._calibrate_z(z_content_raw, self._z_content_mean, self._z_content_std)

            # Score on attention-pooled memory (direct gradient to both z-generators)
            pooled = self._project_and_pool(z_struct, z_content, padded_roles, role_mask)
            turn_surfaces.append(pooled)
            context = context + self.hidden_to_context(pooled).detach()

            # Multi-role decode for diagnostic logging only
            if do_decode:
                tok, mask = self._decode_multirole(z_content, padded_roles, role_mask)
                turn_tokens.append(tok)
                turn_masks.append(mask)

        # 3. Aggregate turns
        weights = F.softmax(self.turn_agg_logits[:cfg.num_turns], dim=0)
        agg = sum(w * s for w, s in zip(weights, turn_surfaces))

        # 4. Contrastive InfoNCE
        msg = F.normalize(self.message_projector(agg), dim=-1)
        tgt = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        sim = msg @ tgt.t()
        logits = sim / temperature
        labels = torch.arange(b, device=device)
        contrastive_loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        # 5. Topology (Pearson ρ on pairwise similarities)
        idx = torch.triu_indices(b, b, offset=1, device=device)
        tgt_flat = (tgt @ tgt.t())[idx[0], idx[1]]
        msg_flat = sim[idx[0], idx[1]]
        if tgt_flat.numel() > 2:
            tc = tgt_flat - tgt_flat.mean()
            mc = msg_flat - msg_flat.mean()
            rho = (tc * mc).sum() / (tc.norm() * mc.norm()).clamp(min=1e-8)
            topo_loss = 1.0 - rho
        else:
            rho = torch.tensor(0.0, device=device)
            topo_loss = torch.tensor(0.0, device=device)

        loss = contrastive_loss + cfg.topo_weight * topo_loss

        # 6. Diagnostics
        avg_len = torch.tensor(0.0, device=device)
        if turn_tokens:
            avg_len = sum(m.float().sum() for m in turn_masks) / (b * len(turn_masks))

        result: dict[str, Tensor] = {
            "loss": loss,
            "accuracy": accuracy,
            "topo_loss": rho,
            "msg_lengths": avg_len,
            "_dialogue_skeletons": [skeletons] * cfg.num_turns,
        }
        if turn_tokens:
            result["_tokens"] = turn_tokens[0]
            result["_dialogue_tokens"] = turn_tokens
            result["_dialogue_masks"] = turn_masks
        return result

    # ---- Trainer interface ----

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        return [
            {"params": list(self.struct_gen.parameters()), "lr": cfg.struct_lr},
            {"params": list(self.content_gen.parameters()), "lr": cfg.content_lr},
            {"params": [self.turn_embeddings], "lr": cfg.receiver_lr},
            {"params": list(self.memory_pooler.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.struct_to_memory.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.hidden_to_context.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.context_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.message_projector.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.log_temperature, self.turn_agg_logits], "lr": cfg.receiver_lr},
        ]

    def checkpoint_state(self) -> dict:
        return {
            "struct_gen": self.struct_gen.state_dict(),
            "content_gen": self.content_gen.state_dict(),
            "turn_embeddings": self.turn_embeddings.data,
            "memory_pooler": self.memory_pooler.state_dict(),
            "struct_to_memory": self.struct_to_memory.state_dict(),
            "hidden_to_context": self.hidden_to_context.state_dict(),
            "context_proj": self.context_proj.state_dict(),
            "message_projector": self.message_projector.state_dict(),
            "log_temperature": self.log_temperature.data,
            "turn_agg_logits": self.turn_agg_logits.data,
            "training_config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.struct_gen.load_state_dict(ckpt["struct_gen"])
        self.content_gen.load_state_dict(ckpt["content_gen"])
        self.turn_embeddings.data.copy_(ckpt["turn_embeddings"])
        if "memory_pooler" in ckpt:
            self.memory_pooler.load_state_dict(ckpt["memory_pooler"])
        if "struct_to_memory" in ckpt:
            self.struct_to_memory.load_state_dict(ckpt["struct_to_memory"])
        if "hidden_to_context" in ckpt:
            self.hidden_to_context.load_state_dict(ckpt["hidden_to_context"])
        self.context_proj.load_state_dict(ckpt["context_proj"])
        self.message_projector.load_state_dict(ckpt["message_projector"])
        self.log_temperature.data.copy_(ckpt["log_temperature"])
        self.turn_agg_logits.data.copy_(ckpt["turn_agg_logits"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplex_init(n: int, d: int) -> Tensor:
    """Initialize n vectors in d dimensions as a regular simplex."""
    vecs = torch.randn(n, d)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    for i in range(1, n):
        for j in range(i):
            vecs[i] -= (vecs[i] @ vecs[j]) * vecs[j]
        vecs[i] = vecs[i] / vecs[i].norm()
    return vecs * 0.1
