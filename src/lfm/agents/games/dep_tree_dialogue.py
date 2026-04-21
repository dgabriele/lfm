"""DepTree Dialogue Game — multi-turn self-play with disentangled structure/content.

Leverages the DepTreeVAE's z_struct/z_content split:
  - **StructureGen**: produces z_struct once from the target embedding.
    Shared across all turns — the agent maintains consistent grammar.
  - **ContentGen**: produces z_content per turn, conditioned on the target
    embedding, turn position, and accumulated context.
  - **DepTreeVAE decoder**: skeleton(z_struct) → projector(z_content, roles)
    → PhraseDecoder → IPA tokens.

The contrastive loss operates on the decoder's hidden states, scored
against all targets in the batch via InfoNCE.
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
from lfm.generator.layers import PhraseDecoder, multiscale_causal_mask, precompute_rope_freqs

logger = logging.getLogger(__name__)


class DepTreeDialogueConfig(LFMBaseConfig):
    """Configuration for the DepTree dialogue game."""

    # Paths
    vae_checkpoint: str = ""
    spm_path: str = ""
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

    # Content z-generator (per-turn)
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    content_gen_hidden: int = 512

    # Structure z-generator (once per dialogue)
    struct_gen_hidden: int = 256

    # Contrastive scoring
    contrastive_scoring: bool = True
    contrastive_temperature: float = 0.07
    topo_weight: float = 1.0

    # KL prior penalty — keeps generated z on the VAE's learned manifold
    # so the decoder produces coherent surface forms, not word salad
    z_kl_weight: float = 0.1

    # Decode
    max_decode_len: int = 60

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    steps: int = 6000
    struct_lr: float = 3e-4
    content_lr: float = 3e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    num_distractors: int = 15
    min_targets: int = 1
    max_targets: int = 1

    # Curriculum
    curriculum: CurriculumConfig = CurriculumConfig(warmup_steps=2500)

    # Runtime
    device: str = "cuda"
    seed: int = 42
    log_every: int = 20
    checkpoint_every: int = 200


class StructureGen(nn.Module):
    """Produce z_struct from the target embedding. One shot per dialogue."""

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
        """(B, input_dim) → (B, struct_dim)"""
        return self.net(embedding)


class DepTreeDialogueGame(nn.Module):
    """Multi-turn dialogue game with disentangled structure/content z-generation.

    The game interface matches what ``AgentTrainer`` expects:
      - ``forward(anchor, distractors) → dict``
      - ``trainable_param_groups() → list[dict]``
      - ``save_checkpoint() / load_checkpoint()``
    """

    def __init__(self, config: DepTreeDialogueConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config

        # Load frozen DepTreeVAE decoder
        self._load_vae(cfg)

        # Structure generator: embedding → z_struct (shared across turns)
        self.struct_gen = StructureGen(
            cfg.embedding_dim, cfg.latent_struct_dim, cfg.struct_gen_hidden,
        )

        # Content generator: embedding + turn context → z_content (per turn)
        # Uses DiffusionZGenerator with max_phrases=1 (single z per turn)
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

        # Turn embeddings (simplex-initialized for maximal equidistance)
        self.turn_embeddings = nn.Parameter(
            _simplex_init(cfg.num_turns, cfg.embedding_dim),
        )

        # Context accumulation: project decoder hidden → embedding space
        self.hidden_to_context = nn.Linear(
            self.vae_cfg.decoder_hidden_dim, cfg.embedding_dim,
        )
        self.context_proj = nn.Linear(cfg.embedding_dim * 2, cfg.embedding_dim)

        # Contrastive scoring: project decoder hidden states to embedding space
        self.message_projector = nn.Linear(
            self.vae_cfg.decoder_hidden_dim, cfg.embedding_dim,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / cfg.contrastive_temperature)),
        )

        # Turn aggregation weights
        self.turn_agg_logits = nn.Parameter(torch.zeros(cfg.num_turns))

        # Cached causal mask (lazily built on first decode)
        self._full_causal_mask: Tensor | None = None

    def _load_vae(self, cfg: DepTreeDialogueConfig) -> None:
        """Load the frozen DepTreeVAE components needed for decoding."""
        ckpt = torch.load(cfg.vae_checkpoint, map_location="cpu", weights_only=False)

        # Handle both resume.pt and model_state formats
        if "model_state" in ckpt:
            ms = ckpt["model_state"]
        else:
            raise ValueError("Expected model_state in VAE checkpoint")

        self.vae_cfg = DepTreeVAEConfig(
            spm_vocab_size=cfg.spm_vocab_size,
            max_seq_len=cfg.vae_max_seq_len,
            latent={"total_dim": cfg.latent_total_dim, "struct_dim": cfg.latent_struct_dim, "content_dim": cfg.latent_content_dim},
        )
        self.vae = DepTreeVAE(self.vae_cfg, cfg.spm_vocab_size + 2 + len(DEP_RELATIONS))
        self.vae.load_state_dict(ms)
        self.vae.eval()

        # Freeze all VAE params
        for p in self.vae.parameters():
            p.requires_grad = False

        logger.info("Loaded frozen DepTreeVAE from %s", cfg.vae_checkpoint)

        # Load SPM for surface rendering
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(cfg.spm_path)

    def render_surface(self, token_ids: Tensor, mask: Tensor | None = None) -> list[str]:
        """Decode token IDs to IPA strings via sentencepiece."""
        sp = self._sp
        bos = self.vae._bos_id
        eos = self.vae._eos_id
        results = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].tolist()
            if mask is not None:
                m = mask[i].tolist()
                ids = [t for t, v in zip(ids, m) if v]
            ids = [t for t in ids if t != bos and t != eos and t != 0 and t < sp.GetPieceSize()]
            results.append(sp.DecodeIds(ids))
        return results

    @torch.no_grad()
    def _decode_turn(
        self, z_struct: Tensor, z_content: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[list[str]]]:
        """Decode one turn: skeleton + projector + PhraseDecoder.

        Fully batched — all samples decoded in parallel.

        Returns:
            hidden: (B, S, H) decoder hidden states for contrastive scoring.
            valid_mask: (B, S) bool mask of valid positions.
            tokens: (B, S) generated token IDs.
            token_mask: (B, S) bool mask for tokens.
            skeletons: list of role name lists per sample (for logging).
        """
        b = z_struct.size(0)
        device = z_struct.device
        vae = self.vae
        cfg = self.vae_cfg
        num_dep = len(DEP_RELATIONS)
        root_id = DEP_RELATIONS.index("root")

        # 1. Batch skeleton decode
        skel_tokens = vae.skeleton_decoder(z_struct)[0]  # (B, max_roles+2)

        # 2. Parse roles from skeleton tokens — vectorized
        is_bos = skel_tokens == SKEL_BOS
        is_eos = skel_tokens == SKEL_EOS
        is_valid = (~is_bos) & (~is_eos) & (skel_tokens < num_dep)

        # Find first EOS per sample to mask everything after it
        eos_positions = torch.where(is_eos, torch.arange(skel_tokens.size(1), device=device), skel_tokens.size(1))
        first_eos = eos_positions.min(dim=1).values  # (B,)
        position_idx = torch.arange(skel_tokens.size(1), device=device).unsqueeze(0)
        before_eos = position_idx < first_eos.unsqueeze(1)
        is_valid = is_valid & before_eos

        # Compact valid roles into padded (B, max_R) tensor
        role_counts = is_valid.sum(dim=1)  # (B,)
        max_r = max(role_counts.max().item(), 1)

        padded_roles = torch.full((b, max_r), root_id, dtype=torch.long, device=device)
        role_mask = torch.zeros(b, max_r, dtype=torch.bool, device=device)
        all_skeletons: list[list[str]] = []

        for i in range(b):
            valid_idx = is_valid[i].nonzero(as_tuple=True)[0]
            n = valid_idx.numel()
            if n == 0:
                padded_roles[i, 0] = root_id
                role_mask[i, 0] = True
                all_skeletons.append(["root"])
            else:
                padded_roles[i, :n] = skel_tokens[i, valid_idx]
                role_mask[i, :n] = True
                all_skeletons.append([DEP_RELATIONS[t.item()] for t in skel_tokens[i, valid_idx]])

        # 3. Batch phrase projection
        memory = vae.phrase_projector(z_content, padded_roles, role_mask)  # (B, max_R, H)

        # Cross-attention mask: block padded memory positions
        # nn.MultiheadAttention float mask: 0 = attend, -inf = block
        nheads = cfg.decoder_num_heads
        mem_attn_mask_1d = (~role_mask).float() * (-1e9)  # (B, max_R)
        # Shape for cached forward: (B*H, 1, max_R)
        xattn_mask = mem_attn_mask_1d.unsqueeze(1).repeat_interleave(nheads, dim=0)

        # 4. KV-cached batched autoregressive decode
        max_len = self.config.max_decode_len + 1
        decoder = vae.phrase_decoder
        kv_cache = decoder.make_kv_cache(b, max_len, device, dtype=torch.float32)

        # Precompute full causal mask
        if self._full_causal_mask is None or self._full_causal_mask.size(1) < max_len:
            self._full_causal_mask = multiscale_causal_mask(
                max_len, nheads,
                tuple(cfg.attention_head_windows),
                cfg.attention_global_every,
                device=device,
            )

        rope = vae._rope_freqs
        if rope is not None and rope.size(0) < max_len:
            rope = precompute_rope_freqs(
                cfg.decoder_hidden_dim // nheads, max_len, device=device,
            )

        # Prime with BOS
        bos_embed = vae.dec_token_embedding(
            torch.full((b, 1), vae._bos_id, dtype=torch.long, device=device),
        )
        mask_row = self._full_causal_mask[:, 0:1, 0:1]
        out = decoder.forward_cached(
            bos_embed, memory, kv_cache,
            rope_freqs=rope, tgt_mask_row=mask_row, xattn_mask=xattn_mask,
        )
        kv_cache.advance()

        # Collect hidden states for contrastive scoring
        all_hidden = [out]
        tokens_list = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for t in range(self.config.max_decode_len):
            next_tok = vae.output_head(out[:, -1]).argmax(dim=-1)  # (B,)
            next_tok = next_tok.masked_fill(finished, vae._pad_id)
            newly_finished = (next_tok == vae._eos_id)
            finished = finished | newly_finished
            tokens_list.append(next_tok)

            if finished.all():
                break

            new_embed = vae.dec_token_embedding(next_tok.unsqueeze(1))
            seq_so_far = kv_cache.seq_len + 1
            mask_row = self._full_causal_mask[
                :, kv_cache.seq_len:kv_cache.seq_len + 1, :seq_so_far
            ]
            out = decoder.forward_cached(
                new_embed, memory, kv_cache,
                rope_freqs=rope, tgt_mask_row=mask_row, xattn_mask=xattn_mask,
            )
            kv_cache.advance()
            all_hidden.append(out)

        # 5. Stack hidden states and build validity mask
        hidden = torch.cat(all_hidden, dim=1)  # (B, seq_len, H) — includes BOS
        hidden = hidden[:, 1:, :]  # strip BOS position

        if tokens_list:
            gen_tokens = torch.stack(tokens_list, dim=1)  # (B, seq_len)
            valid_mask = (gen_tokens != vae._pad_id) & (gen_tokens != vae._eos_id)
        else:
            gen_tokens = torch.zeros(b, 0, dtype=torch.long, device=device)
            valid_mask = torch.zeros(b, 0, dtype=torch.bool, device=device)

        # Align lengths
        seq_len = hidden.size(1)
        if valid_mask.size(1) > seq_len:
            valid_mask = valid_mask[:, :seq_len]
            gen_tokens = gen_tokens[:, :seq_len]
        elif valid_mask.size(1) < seq_len:
            pad_m = torch.zeros(b, seq_len - valid_mask.size(1), dtype=torch.bool, device=device)
            pad_t = torch.zeros(b, seq_len - gen_tokens.size(1), dtype=torch.long, device=device)
            valid_mask = torch.cat([valid_mask, pad_m], dim=1)
            gen_tokens = torch.cat([gen_tokens, pad_t], dim=1)

        return hidden, valid_mask, gen_tokens, valid_mask, all_skeletons

    def forward(
        self, anchor: Tensor, distractors: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Run one game step.

        Args:
            anchor: (B, D) target embeddings.
            distractors: unused (batch-wide InfoNCE uses all anchors).
            **kwargs: absorbed from AgentTrainer (step, candidate_indices).

        Returns:
            Dict with loss, accuracy, and diagnostic tensors.
        """
        b, d = anchor.shape
        device = anchor.device
        cfg = self.config

        # 1. Structure: shared across all turns
        z_struct = self.struct_gen(anchor)  # (B, struct_dim)

        # 2. Per-turn content generation + decode
        turn_surfaces: list[Tensor] = []  # pooled hidden per turn
        turn_skeletons: list[list[list[str]]] = []
        turn_tokens: list[Tensor] = []
        turn_masks: list[Tensor] = []
        all_z: list[Tensor] = [z_struct]

        context = torch.zeros(b, d, device=device)

        for t in range(cfg.num_turns):
            # Condition content gen on target + turn embedding + context
            turn_emb = self.turn_embeddings[t].unsqueeze(0).expand(b, -1)
            cond = self.context_proj(torch.cat([anchor + turn_emb, context], dim=-1))

            # Generate z_content for this turn
            z_content_raw, _, _ = self.content_gen(cond)
            z_content = z_content_raw[:, 0, :]  # (B, content_dim)
            all_z.append(z_content)

            # Decode
            hidden, mask, tokens, token_mask, skeletons = self._decode_turn(z_struct, z_content)
            turn_skeletons.append(skeletons)
            turn_tokens.append(tokens)
            turn_masks.append(token_mask)

            # Pool hidden states (mean over valid positions)
            masked_h = hidden * mask.unsqueeze(-1).float()
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_h.sum(dim=1) / lengths.squeeze(-1).unsqueeze(-1)
            turn_surfaces.append(pooled)

            # Update context (project decoder hidden → embedding space)
            context = context + self.hidden_to_context(pooled).detach()

        # 3. Aggregate turns
        weights = F.softmax(self.turn_agg_logits[: cfg.num_turns], dim=0)
        agg = sum(w * s for w, s in zip(weights, turn_surfaces))

        # 4. Contrastive scoring
        msg = F.normalize(self.message_projector(agg), dim=-1)
        tgt = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        sim = msg @ tgt.t()
        logits = sim / temperature
        labels = torch.arange(b, device=device)
        contrastive_loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        # 5. Topology loss — differentiable Pearson ρ between pairwise
        # message cosine similarities and pairwise target cosine similarities.
        # Pushes message space to preserve embedding-space distance structure.
        idx = torch.triu_indices(b, b, offset=1, device=device)
        tgt_sim = (tgt @ tgt.t())[idx[0], idx[1]]
        msg_sim = sim[idx[0], idx[1]]

        if tgt_sim.numel() > 2:
            tc = tgt_sim - tgt_sim.mean()
            mc = msg_sim - msg_sim.mean()
            denom = (tc.norm() * mc.norm()).clamp(min=1e-8)
            rho = (tc * mc).sum() / denom
            topo_loss = 1.0 - rho
        else:
            rho = torch.tensor(0.0, device=device)
            topo_loss = torch.tensor(0.0, device=device)

        # 6. KL prior penalty — keep z on the VAE's learned manifold.
        # The VAE was trained with KL toward N(0,1), so z near the prior
        # produces coherent surface forms. Penalty = 0.5 * (mu^2 + sigma^2 - 1)
        # simplified to 0.5 * ||z||^2 since we have point estimates, not distributions.
        z_all = torch.cat(all_z, dim=-1)  # (B, struct_dim + num_turns * content_dim)
        z_kl = 0.5 * (z_all ** 2).mean()

        loss = contrastive_loss + cfg.topo_weight * topo_loss + cfg.z_kl_weight * z_kl

        return {
            "loss": loss,
            "accuracy": accuracy,
            "topo_loss": rho,
            "z_kl": z_kl,
            "msg_lengths": torch.tensor(float(cfg.max_decode_len)),
            "_tokens": turn_tokens[0],
            "_dialogue_tokens": turn_tokens,
            "_dialogue_masks": turn_masks,
            "_dialogue_skeletons": turn_skeletons,
        }

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        return [
            {"params": list(self.struct_gen.parameters()), "lr": cfg.struct_lr},
            {"params": list(self.content_gen.parameters()), "lr": cfg.content_lr},
            {"params": [self.turn_embeddings], "lr": cfg.receiver_lr},
            {"params": list(self.hidden_to_context.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.context_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.message_projector.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.log_temperature, self.turn_agg_logits], "lr": cfg.receiver_lr},
        ]

    def checkpoint_state(self) -> dict:
        """Return state dict for AgentTrainer checkpointing."""
        return {
            "struct_gen": self.struct_gen.state_dict(),
            "content_gen": self.content_gen.state_dict(),
            "turn_embeddings": self.turn_embeddings.data,
            "hidden_to_context": self.hidden_to_context.state_dict(),
            "context_proj": self.context_proj.state_dict(),
            "message_projector": self.message_projector.state_dict(),
            "log_temperature": self.log_temperature.data,
            "turn_agg_logits": self.turn_agg_logits.data,
            "training_config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict (AgentTrainer interface)."""
        self.struct_gen.load_state_dict(ckpt["struct_gen"])
        self.content_gen.load_state_dict(ckpt["content_gen"])
        self.turn_embeddings.data.copy_(ckpt["turn_embeddings"])
        if "hidden_to_context" in ckpt:
            self.hidden_to_context.load_state_dict(ckpt["hidden_to_context"])
        self.context_proj.load_state_dict(ckpt["context_proj"])
        self.message_projector.load_state_dict(ckpt["message_projector"])
        self.log_temperature.data.copy_(ckpt["log_temperature"])
        self.turn_agg_logits.data.copy_(ckpt["turn_agg_logits"])


def _simplex_init(n: int, d: int) -> Tensor:
    """Initialize n vectors in d dimensions as a regular simplex."""
    vecs = torch.randn(n, d)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    # Gram-Schmidt-like orthogonalization for equidistance
    for i in range(1, n):
        for j in range(i):
            vecs[i] -= (vecs[i] @ vecs[j]) * vecs[j]
        vecs[i] = vecs[i] / vecs[i].norm()
    return vecs * 0.1  # small initial scale
