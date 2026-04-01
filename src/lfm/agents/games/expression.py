"""Expression game with GRU z-sequence generation and PonderNet halting.

Replaces the REINFORCE tree-topology expression system with a fully
differentiable GRU that autoregressively produces z vectors.  The
frozen decoder decodes each z as a segment (until EOS), with KV cache
persisting across boundaries for coarticulation.

Halting uses a geometric prior KL (PonderNet, Banino et al. 2021):
the learned halt distribution is regularized toward a geometric
distribution p(k) = lambda * (1-lambda)^{k-1}, naturally producing
Zipf-like expression lengths.

Two-phase forward (same as referential):
  1. GRU emits z sequence, decode segments with no_grad (KV-cached).
  2. Re-run GRU + decoder in parallel with gradients through
     cross-attention to position-wise memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, Receiver
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import rerun_decoder_multiseg_with_grad
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ExpressionGameConfig(LFMBaseConfig):
    """Configuration for the GRU expression game."""

    # Faculty
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 96
    vq_codebook_path: str | None = None
    vq_residual_alpha: float = 1.0

    # GRU z-sequence generator
    z_hidden_dim: int = 512
    max_segments: int = 8
    max_tokens_per_segment: int = 48

    # PonderNet geometric prior: lambda_p controls expected segment count.
    # E[K] = 1/lambda_p.  lambda_p=0.5 → ~2 segments, 0.3 → ~3.3, 0.2 → ~5.
    lambda_p: float = 0.4
    kl_beta: float = 0.5

    # Message encoder
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Game
    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    # Training
    batch_size: int = 256
    steps: int = 2000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/expression_game"

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
                num_statements=1,
                vq_codebook_path=self.vq_codebook_path,
                vq_residual_alpha=self.vq_residual_alpha,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# GRU z-sequence generator with PonderNet halting
# ---------------------------------------------------------------------------


class ZSequenceGenerator(nn.Module):
    """Autoregressive z-sequence generator with geometric-prior halting.

    z_0 is a direct projection of the input (discriminative from step 0).
    z_1..z_K are produced by a GRU conditioned on previous z's.  The
    halt distribution is regularized toward a geometric prior via KL
    divergence (PonderNet).

    Args:
        input_dim: Anchor embedding dimension.
        hidden_dim: GRU hidden dimension.
        latent_dim: Output z dimension (must match decoder).
        max_segments: Upper bound on segment count.
        z_mean: Pretrained z distribution mean for initialization.
        z_std: Pretrained z distribution std for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_segments: int = 8,
        z_mean: Tensor | None = None,
        z_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.max_segments = max_segments
        self.latent_dim = latent_dim

        # z_0: direct projection (discriminative from step 0)
        self.z0_proj = nn.Linear(input_dim, latent_dim)

        # GRU for z_1..z_K
        self.h_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, latent_dim)
        self.halt_head = nn.Linear(hidden_dim, 1)

        # Scale z projections to match the pretrained z distribution
        if z_mean is not None and z_std is not None:
            with torch.no_grad():
                target_std = z_std.mean().item()
                for proj in [self.z0_proj, self.z_proj]:
                    fan_in = proj.weight.size(1)
                    current_std = proj.weight.data.std().item() * (fan_in ** 0.5)
                    scale = target_std / max(current_std, 1e-6)
                    proj.weight.data.mul_(scale)
                    proj.bias.data.copy_(z_mean)

    def forward(
        self, embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate z sequence with per-step halt probabilities.

        Args:
            embedding: ``(batch, input_dim)`` anchor embeddings.

        Returns:
            z_sequence: ``(batch, max_segments, latent_dim)``.
            halt_probs: ``(batch, max_segments)`` per-step halt probs.
            z_weights: ``(batch, max_segments)`` effective weights.
            num_segments: ``(batch,)`` soft segment count.
        """
        batch = embedding.size(0)
        device = embedding.device
        K = self.max_segments

        z_seq = torch.zeros(batch, K, self.latent_dim, device=device)
        halts = torch.zeros(batch, K, device=device)
        weights = torch.ones(batch, K, device=device)

        # z_0: direct projection (discriminative from step 0)
        z = self.z0_proj(embedding)
        z_seq[:, 0] = z

        # GRU hidden state
        h = self.h_proj(embedding)

        # Track which samples have halted
        running = torch.ones(batch, device=device)

        for i in range(1, K):
            h = self.gru(z, h)
            z = self.z_proj(h)
            lam = torch.sigmoid(self.halt_head(h)).squeeze(-1)

            halts[:, i] = lam
            z_seq[:, i] = z

            # Probability of being active at step i = prod(1-halt) up to i-1
            # Weight = running probability (not yet halted)
            weights[:, i] = running * (1.0 - lam)
            running = running * (1.0 - lam)

        # Soft segment count = sum of weights
        num_segments = weights.sum(dim=-1)

        return z_seq, halts, weights, num_segments


def geometric_kl(halt_probs: Tensor, lambda_p: float) -> Tensor:
    """KL divergence between learned halt distribution and geometric prior.

    From PonderNet (Banino et al., "PonderNet: Learning to Ponder",
    ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application
    Grounding of Deep Learning).

    The learned distribution q(k) is derived from per-step halt probs.
    The geometric prior is p(k) = lambda_p * (1 - lambda_p)^{k-1}.

    Args:
        halt_probs: ``(batch, max_segments)`` per-step halt probabilities.
            halt_probs[:, 0] is unused (first segment always active).
        lambda_p: Geometric prior parameter (E[K] = 1/lambda_p).

    Returns:
        Scalar KL divergence averaged over the batch.
    """
    K = halt_probs.size(1)
    eps = 1e-8

    # Build q(k): probability of halting at step k
    # q(k) = halt_probs[k] * prod(1 - halt_probs[j] for j < k)
    # For k=0: q(0) = 1 (always use first segment... actually we model
    # halting starting from step 1)
    # Simpler: treat halt_probs[1:] as the per-step stopping probability
    q_continue = torch.ones_like(halt_probs[:, 0])
    q_halt = []

    for i in range(1, K):
        q_k = q_continue * halt_probs[:, i]
        q_halt.append(q_k)
        q_continue = q_continue * (1.0 - halt_probs[:, i])

    # Remaining probability mass (didn't halt before K)
    q_halt.append(q_continue)
    q = torch.stack(q_halt, dim=-1)  # (B, K)
    q = q.clamp(min=eps)

    # Geometric prior: p(k) = lambda * (1-lambda)^k for k=0..K-2, remainder at K-1
    p = torch.zeros(K, device=halt_probs.device)
    for k in range(K - 1):
        p[k] = lambda_p * ((1.0 - lambda_p) ** k)
    p[-1] = (1.0 - lambda_p) ** (K - 1)  # tail mass
    p = p.clamp(min=eps)
    p = p / p.sum()  # normalize

    # KL(q || p) = sum q * log(q/p)
    kl = (q * (q.log() - p.log().unsqueeze(0))).sum(dim=-1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Expression game
# ---------------------------------------------------------------------------


class ExpressionGame(nn.Module):
    """Expression game with GRU z-sequence through the linguistic bottleneck.

    Args:
        config: Expression game configuration.
        faculty: Pre-built ``LanguageFaculty`` (moved to device by caller).
    """

    def __init__(
        self, config: ExpressionGameConfig, faculty: LanguageFaculty,
    ) -> None:
        super().__init__()
        self.config = config
        self.faculty = faculty

        gen = faculty.generator
        gen.eval()

        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        self.z_gen = ZSequenceGenerator(
            input_dim=config.embedding_dim,
            hidden_dim=config.z_hidden_dim,
            latent_dim=gen._latent_dim,
            max_segments=config.max_segments,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
        )
        self.msg_encoder = MessageEncoder(
            gen.config.decoder_hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.receiver = Receiver(config.embedding_dim)

    @property
    def gen(self):
        """Shortcut to the underlying generator."""
        return self.faculty.generator

    def checkpoint_state(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            "z_gen": self.z_gen.state_dict(),
            "msg_encoder": self.msg_encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict."""
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.msg_encoder.load_state_dict(ckpt["msg_encoder"])
        self.receiver.load_state_dict(ckpt["receiver"])

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        return [
            {"params": list(self.z_gen.parameters()), "lr": self.config.gru_lr},
            {"params": list(self.msg_encoder.parameters()), "lr": self.config.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": self.config.receiver_lr},
        ]

    def forward(
        self, anchor: Tensor, distractors: Tensor,
    ) -> dict[str, Tensor]:
        """Two-phase expression game forward pass."""
        batch = anchor.size(0)
        device = anchor.device
        num_candidates = distractors.size(1) + 1

        # GRU z-sequence (with grad)
        z_seq, halt_probs, z_weights, num_segments = self.z_gen(anchor)

        # Phase 1: multi-segment decode (no_grad)
        with torch.no_grad():
            tokens, gen_mask, seg_bounds = self._multiseg_decode(
                z_seq, z_weights,
            )

        # Phase 2: re-run decoder with gradients
        hidden = rerun_decoder_multiseg_with_grad(
            self.gen, z_seq, z_weights, tokens, gen_mask, seg_bounds,
        )
        trimmed_mask = gen_mask[:, :hidden.size(1)]

        # Encode message and score candidates
        message = self.msg_encoder(hidden, trimmed_mask)

        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack([
            torch.randperm(num_candidates, device=device)
            for _ in range(batch)
        ])
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = self.receiver(message, candidates)
        receiver_loss = F.cross_entropy(logits, target_idx)

        # PonderNet KL: regularize halt distribution toward geometric prior
        total_tokens = trimmed_mask.float().sum(dim=1)
        kl_loss = geometric_kl(halt_probs, self.config.lambda_p)

        # Z redundancy: penalize consecutive z's that are too similar.
        # If z_i ≈ z_{i-1}, the segment is redundant — the GRU should
        # either halt or produce a distinct z.
        z_normed = F.normalize(z_seq, dim=-1)
        consec_sim = (z_normed[:, 1:] * z_normed[:, :-1]).sum(dim=-1)  # (B, K-1)
        # Penalize positive similarity only — don't reward anti-correlation
        consec_sim = consec_sim.clamp(min=0)
        pair_weight = torch.min(z_weights[:, 1:], z_weights[:, :-1])
        z_redundancy = (consec_sim * pair_weight).sum(dim=-1).mean()

        loss = (
            receiver_loss
            + self.config.kl_beta * kl_loss
            + self.config.kl_beta * z_redundancy
        )

        with torch.no_grad():
            accuracy = (logits.argmax(1) == target_idx).float().mean()

            # Intra-expression z diversity
            K = z_seq.size(1)
            z_normed = F.normalize(z_seq, dim=-1)
            z_sim = torch.bmm(z_normed, z_normed.transpose(1, 2))
            active = z_weights > 0.01
            active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)
            diag_mask = ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
            pair_mask = active_pair & diag_mask
            n_pairs = pair_mask.float().sum(dim=(1, 2)).clamp(min=1)
            z_intra_sim = (z_sim.clamp(min=0) * pair_mask.float()).sum(dim=(1, 2)) / n_pairs

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": total_tokens.mean().detach(),
            "logits": logits,
            "target_idx": target_idx,
            "halt_cost": kl_loss.detach(),
            "num_segments": num_segments.mean().detach(),
            "z_intra_sim": z_intra_sim.mean().detach(),
        }

    @torch.no_grad()
    def _multiseg_decode(
        self, z_seq: Tensor, z_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Phase 1: KV-cached multi-segment decode with z-switching."""
        from lfm.generator.layers import LinguisticDecoder, multiscale_causal_mask

        gen = self.gen
        cfg = self.config
        batch, K, _ = z_seq.shape
        device = z_seq.device
        max_tok_per_seg = cfg.max_tokens_per_segment
        max_total = max_tok_per_seg * K

        weighted_z = z_weights.unsqueeze(-1) * z_seq
        n_mem = gen._num_memory_tokens
        hidden_dim = gen.config.decoder_hidden_dim
        memories = gen.latent_to_decoder(
            weighted_z.reshape(batch * K, -1),
        ).reshape(batch, K, n_mem, hidden_dim)

        decoder = gen.decoder
        is_linguistic = isinstance(decoder, LinguisticDecoder)

        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_total + 1:
            gen._full_causal_mask = multiscale_causal_mask(
                max_total + 1,
                num_heads=gen.config.decoder_num_heads,
                head_windows=gen.config.attention_head_windows,
                global_every=gen.config.attention_global_every,
                device=device,
            )

        all_tokens = torch.zeros(batch, max_total, dtype=torch.long, device=device)
        all_mask = torch.zeros(batch, max_total, dtype=torch.bool, device=device)
        seg_bounds = torch.zeros(batch, K, dtype=torch.long, device=device)

        cur_seg = torch.zeros(batch, dtype=torch.long, device=device)
        tokens_in_seg = torch.zeros(batch, dtype=torch.long, device=device)
        total_pos = torch.zeros(batch, dtype=torch.long, device=device)

        seg_active = z_weights > 0.01
        finished = ~seg_active[:, 0]

        # Extend RoPE freqs for multi-segment decode beyond max_seq_len
        rope_freqs = gen._rope_freqs
        if rope_freqs is not None and rope_freqs.size(0) < max_total + 1:
            from lfm.generator.layers import precompute_rope_freqs
            rope_freqs = precompute_rope_freqs(
                gen.config.decoder_hidden_dim // gen.config.decoder_num_heads,
                max_total + 1, device=device,
            )

        if is_linguistic:
            kv_cache = decoder.make_kv_cache(
                batch, max_total + 1, device, dtype=torch.float16,
            )

        cur_embed = gen.token_embedding(
            torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device),
        )
        batch_idx = torch.arange(batch, device=device)

        def _gather_memory() -> Tensor:
            idx = cur_seg.clamp(max=K - 1)
            mem = memories[batch_idx, idx]
            active_mask = seg_active[batch_idx, idx] & ~finished
            return mem * active_mask.unsqueeze(-1).unsqueeze(-1).float()

        memory = _gather_memory()

        if is_linguistic:
            mask_row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(
                cur_embed, memory, kv_cache,
                rope_freqs=rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()
        else:
            out = decoder(cur_embed, memory)

        for t in range(max_total):
            logits = gen.output_head(out[:, -1])
            next_token = logits.argmax(dim=-1)

            # Vectorized token storage
            active = ~finished
            all_tokens[batch_idx, total_pos] = next_token * active.long()
            all_mask[batch_idx, total_pos] = active
            total_pos += active.long()
            tokens_in_seg += active.long()

            # Vectorized z-switch
            hit_eos = (next_token == gen.eos_id) & (tokens_in_seg >= 1)
            hit_max = tokens_in_seg >= max_tok_per_seg
            should_switch = (hit_eos | hit_max) & active

            cur_seg += should_switch.long()
            tokens_in_seg *= ~should_switch

            # Record segment boundaries
            switched_valid = should_switch & (cur_seg < K)
            if switched_valid.any():
                seg_bounds[batch_idx[switched_valid], cur_seg[switched_valid]] = (
                    total_pos[switched_valid]
                )

            # Mark finished
            clamped = cur_seg.clamp(max=K - 1)
            next_inactive = ~seg_active[batch_idx, clamped]
            finished = finished | (cur_seg >= K) | (should_switch & next_inactive)

            if finished.all():
                break

            memory = _gather_memory()
            new_embed = gen.token_embedding(next_token.unsqueeze(1))

            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
            else:
                cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
                all_embed = gen.token_embedding(cur_ids)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    cur_ids.size(1), device=device,
                )
                out = decoder(all_embed, memory, tgt_mask=tgt_mask)

        return all_tokens, all_mask, seg_bounds
