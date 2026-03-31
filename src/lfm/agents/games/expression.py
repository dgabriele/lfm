"""Expression game with GRU z-sequence generation and ACT halting.

Replaces the REINFORCE tree-topology expression system with a fully
differentiable GRU that autoregressively produces z vectors.  The
frozen decoder decodes each z as a segment (until EOS), with KV cache
persisting across boundaries for coarticulation.

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

    # Parsimony
    halt_alpha: float = 0.1
    segment_beta: float = 0.05

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
# GRU z-sequence generator with ACT halting
# ---------------------------------------------------------------------------


class ZSequenceGenerator(nn.Module):
    """Autoregressive z-sequence generator with differentiable halting.

    Replaces the tree-topology REINFORCE system.  A GRU emits a
    sequence of z vectors, with Adaptive Computation Time (ACT)
    determining the effective number of segments.

    Args:
        input_dim: Anchor embedding dimension.
        hidden_dim: GRU hidden dimension.
        latent_dim: Output z dimension (must match decoder).
        max_segments: Upper bound on segment count.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_segments: int = 8,
    ) -> None:
        super().__init__()
        self.max_segments = max_segments
        self.latent_dim = latent_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, latent_dim)
        self.halt_head = nn.Linear(hidden_dim, 1)

        # Bias toward continuing (low initial halt probability)
        with torch.no_grad():
            self.halt_head.bias.fill_(-2.0)

    def forward(
        self, embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate z sequence with ACT halting.

        Args:
            embedding: ``(batch, input_dim)`` anchor embeddings.

        Returns:
            z_sequence: ``(batch, max_segments, latent_dim)``.
            halt_probs: ``(batch, max_segments)`` per-step halt probs.
            z_weights: ``(batch, max_segments)`` ACT weights (1.0 except
                last active segment which gets remainder R).
            num_segments: ``(batch,)`` soft segment count.
        """
        batch = embedding.size(0)
        device = embedding.device
        K = self.max_segments

        z_seq = torch.zeros(batch, K, self.latent_dim, device=device)
        halts = torch.zeros(batch, K, device=device)
        weights = torch.zeros(batch, K, device=device)

        # Initial hidden state and first z
        h = self.input_proj(embedding)
        z = self.z_proj(h)
        z_seq[:, 0] = z

        cumulative = torch.zeros(batch, device=device)
        # First segment always used — no halt check
        weights[:, 0] = 1.0

        for i in range(1, K):
            h = self.gru(z, h)
            z = self.z_proj(h)
            lam = torch.sigmoid(self.halt_head(h)).squeeze(-1)
            halts[:, i] = lam

            # ACT: check if cumulative + lambda >= 1
            new_cumulative = cumulative + lam
            remainder = 1.0 - cumulative

            # Where we've already halted, weight = 0
            # Where we halt this step, weight = remainder
            # Where we continue, weight = 1
            halted_now = new_cumulative >= 1.0
            w = torch.where(
                cumulative >= 1.0,
                torch.zeros_like(lam),
                torch.where(halted_now, remainder.clamp(min=0.01), torch.ones_like(lam)),
            )
            weights[:, i] = w
            z_seq[:, i] = z

            cumulative = new_cumulative

        # Soft segment count
        num_segments = weights.sum(dim=-1)

        return z_seq, halts, weights, num_segments


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
        """Two-phase expression game forward pass.

        Args:
            anchor: ``(batch, embedding_dim)`` target embeddings.
            distractors: ``(batch, K, embedding_dim)`` distractors.

        Returns:
            Dict with ``loss``, ``accuracy``, ``msg_lengths``, ``logits``,
            ``target_idx``, ``halt_cost``, ``token_cost``, ``num_segments``.
        """
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

        # Trim mask to match hidden (rerun trims to actual max length)
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

        # Parsimony losses
        halt_cost = self.config.halt_alpha * halt_probs.sum(dim=-1).mean()
        total_tokens = trimmed_mask.float().sum(dim=1)
        token_cost = self.config.segment_beta * torch.log1p(total_tokens).mean()
        loss = receiver_loss + halt_cost + token_cost

        with torch.no_grad():
            accuracy = (logits.argmax(1) == target_idx).float().mean()

            # Intra-expression z diversity: mean pairwise cosine distance
            # between active z vectors within each expression
            z_normed = F.normalize(z_seq, dim=-1)
            z_sim = torch.bmm(z_normed, z_normed.transpose(1, 2))  # (B, K, K)
            # Mask to active segments only
            active = z_weights > 0.01  # (B, K)
            active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)  # (B, K, K)
            # Exclude diagonal
            diag_mask = ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
            pair_mask = active_pair & diag_mask
            n_pairs = pair_mask.float().sum(dim=(1, 2)).clamp(min=1)
            z_intra_sim = (z_sim * pair_mask.float()).sum(dim=(1, 2)) / n_pairs

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": total_tokens.mean().detach(),
            "logits": logits,
            "target_idx": target_idx,
            "halt_cost": halt_cost.detach(),
            "token_cost": token_cost.detach(),
            "num_segments": num_segments.mean().detach(),
            "z_intra_sim": z_intra_sim.mean().detach(),
        }

    @torch.no_grad()
    def _multiseg_decode(
        self, z_seq: Tensor, z_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Phase 1: KV-cached multi-segment decode with z-switching.

        Args:
            z_seq: ``(batch, max_segments, latent_dim)``.
            z_weights: ``(batch, max_segments)`` ACT weights.

        Returns:
            tokens: ``(batch, max_total_tokens)`` generated token IDs.
            mask: ``(batch, max_total_tokens)`` boolean validity mask.
            segment_boundaries: ``(batch, max_segments)`` start positions.
        """
        from lfm.generator.layers import LinguisticDecoder, multiscale_causal_mask

        gen = self.gen
        cfg = self.config
        batch, K, _ = z_seq.shape
        device = z_seq.device
        max_tok_per_seg = cfg.max_tokens_per_segment
        max_total = max_tok_per_seg * K

        # Weighted z → memory per segment
        weighted_z = z_weights.unsqueeze(-1) * z_seq
        n_mem = gen._num_memory_tokens
        hidden_dim = gen.config.decoder_hidden_dim
        memories = gen.latent_to_decoder(
            weighted_z.reshape(batch * K, -1),
        ).reshape(batch, K, n_mem, hidden_dim)

        decoder = gen.decoder
        is_linguistic = isinstance(decoder, LinguisticDecoder)

        # Ensure causal mask is big enough
        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_total + 1:
            gen._full_causal_mask = multiscale_causal_mask(
                max_total + 1,
                num_heads=gen.config.decoder_num_heads,
                head_windows=gen.config.attention_head_windows,
                global_every=gen.config.attention_global_every,
                device=device,
            )

        # Output buffers
        all_tokens = torch.zeros(batch, max_total, dtype=torch.long, device=device)
        all_mask = torch.zeros(batch, max_total, dtype=torch.bool, device=device)
        seg_bounds = torch.zeros(batch, K, dtype=torch.long, device=device)

        cur_seg = torch.zeros(batch, dtype=torch.long, device=device)
        tokens_in_seg = torch.zeros(batch, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        total_pos = torch.zeros(batch, dtype=torch.long, device=device)

        # Active segment mask: segments with weight > 0
        seg_active = z_weights > 0.01

        # KV cache persists across segment boundaries
        if is_linguistic:
            kv_cache = decoder.make_kv_cache(
                batch, max_total + 1, device, dtype=torch.float16,
            )

        # BOS
        cur_ids = torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device)
        cur_embed = gen.token_embedding(cur_ids)

        def _get_memory() -> Tensor:
            mem = torch.zeros(batch, n_mem, hidden_dim, device=device)
            for bi in range(batch):
                s = cur_seg[bi].item()
                if s < K and seg_active[bi, s]:
                    mem[bi] = memories[bi, s]
            return mem

        memory = _get_memory()

        # Prime with BOS
        if is_linguistic:
            mask_row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(
                cur_embed, memory, kv_cache,
                rope_freqs=gen._rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()
        else:
            out = decoder(cur_embed, memory)

        # AR loop with z-switching
        for t in range(max_total):
            logits = gen.output_head(out[:, -1])
            next_token = logits.argmax(dim=-1)

            for bi in range(batch):
                if not finished[bi]:
                    pos = total_pos[bi].item()
                    all_tokens[bi, pos] = next_token[bi]
                    all_mask[bi, pos] = True
                    total_pos[bi] += 1
                    tokens_in_seg[bi] += 1

            # Z-switch decisions
            for bi in range(batch):
                if finished[bi]:
                    continue
                should_switch = (
                    (next_token[bi] == gen.eos_id)
                    or (tokens_in_seg[bi] >= max_tok_per_seg)
                )
                if should_switch:
                    cur_seg[bi] += 1
                    tokens_in_seg[bi] = 0
                    if cur_seg[bi] < K:
                        seg_bounds[bi, cur_seg[bi]] = total_pos[bi]
                    s = cur_seg[bi].item()
                    if s >= K or not seg_active[bi, min(s, K - 1)]:
                        finished[bi] = True

            if finished.all():
                break

            memory = _get_memory()
            new_embed = gen.token_embedding(next_token.unsqueeze(1))

            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=gen._rope_freqs, tgt_mask_row=mask_row,
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
