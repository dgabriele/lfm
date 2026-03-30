"""Referential game with direct backprop through the linguistic bottleneck.

The sender's embedding is projected to z, the frozen decoder generates
variable-length IPA text via differentiable Gumbel-Softmax, the decoder's
hidden states are re-encoded via attention into a fixed vector, and the
receiver scores candidates against it.

Two-phase forward:
  1. Generate token sequence with no_grad (fast KV-cached decode).
  2. Re-run the decoder on the generated tokens WITH gradients in one
     parallel pass.  Gradients flow through cross-attention to the latent
     memory back to _input_proj.

Usage::

    python scripts/precompute_embeddings.py  # or real embeddings
    python scripts/run_referential_reinforce.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig
from lfm.generator.layers import multiscale_causal_mask

# Force line-buffered output
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message encoder: attention over decoder hidden states
# ---------------------------------------------------------------------------


class MessageEncoder(nn.Module):
    """Encode variable-length decoder hidden states into a fixed message vector.

    Uses self-attention to process the decoder's multi-scale hidden states,
    then a learned query cross-attention readout to produce a fixed-size
    vector.  This preserves the rich per-position structure from the
    multi-head decoder rather than destroying it with mean-pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.readout = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        """Encode decoder hidden states into a message vector.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) decoder states.
            mask: (batch, seq_len) boolean mask (True = valid).

        Returns:
            (batch, output_dim) message vector.
        """
        pad_mask = ~mask
        x = hidden_states
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        b = x.size(0)
        query = self.query.expand(b, -1, -1)
        readout, _ = self.readout(query, x, x, key_padding_mask=pad_mask)
        readout = self.out_norm(readout.squeeze(1))
        return self.proj(readout)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


class Receiver(nn.Module):
    """Score candidates against the message via learned dot-product."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        projected = self.proj(message)
        return torch.bmm(
            projected.unsqueeze(1), candidates.transpose(1, 2),
        ).squeeze(1)


# ---------------------------------------------------------------------------
# Differentiable decoder re-run
# ---------------------------------------------------------------------------


def rerun_decoder_with_grad(
    gen, z: Tensor, tokens: Tensor, mask: Tensor,
) -> Tensor:
    """Re-run the frozen decoder on generated tokens WITH gradients.

    Phase 2 of the two-phase forward: takes the token sequence from
    phase 1 (no_grad generation) and runs it through the decoder in
    one parallel pass.  Gradients flow through the decoder's cross-
    attention to memory, back to z and _input_proj.

    The token embeddings are detached (integer lookup) — only the
    cross-attention path to memory carries gradient signal.

    Args:
        gen: The MultilingualVAEGenerator (frozen decoder).
        z: Latent codes (batch, latent_dim) WITH gradient.
        tokens: Generated token IDs (batch, seq_len) from phase 1.
        mask: Boolean mask (batch, seq_len).

    Returns:
        Decoder hidden states (batch, seq_len, hidden_dim) with gradients.
    """
    # Calibrate / quantize z (same path as generation)
    if gen._vq_codebook is not None:
        z_dec = gen._quantize_z(z)
    elif gen._z_stats_initialized:
        z_dec = gen.calibrate_z(z)
    else:
        z_dec = z

    # z → memory (differentiable through frozen linear)
    memory = gen.latent_to_decoder(z_dec).reshape(
        z.size(0), gen._num_memory_tokens, -1,
    )

    # Embed the generated tokens (detached — no grad through token chain)
    tok_emb = gen.token_embedding(tokens)

    # Full causal mask for the sequence
    seq_len = tokens.size(1)
    causal_mask = multiscale_causal_mask(
        seq_len,
        num_heads=gen.config.decoder_num_heads,
        head_windows=gen.config.attention_head_windows,
        global_every=gen.config.attention_global_every,
        device=z.device,
    )

    # Run decoder in one parallel pass — cross-attention to memory has grad
    rope = gen._rope_freqs[:seq_len] if gen._rope_freqs is not None else None
    hidden = gen.decoder(
        tok_emb, memory, tgt_mask=causal_mask, rope_freqs=rope,
    )

    return hidden


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    steps: int = 2000,
    sender_lr: float = 1e-4,
    receiver_lr: float = 1e-3,
    batch_size: int = 64,
    num_distractors: int = 15,
    embedding_dim: int = 384,
    device: str = "cuda",
    log_every: int = 50,
    checkpoint_every: int = 100,
    output_dir: str = "data/referential_game",
    curriculum: bool = True,
    curriculum_warmup: int = 500,
    curriculum_start: float = 0.0,
    curriculum_end: float = 1.0,
    resume: str | None = None,
    vq_codebook_path: str | None = None,
    vq_residual_alpha: float = 1.0,
    num_statements: int = 1,
    max_output_len: int = 96,
    num_memory_tokens: int = 8,
    encoder_layers: int = 2,
    encoder_heads: int = 8,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run the referential game with direct backprop.

    Single optimizer trains _input_proj, message encoder, and receiver
    end-to-end via the receiver's cross-entropy loss.  Gradients flow
    through the frozen decoder's cross-attention to the latent memory.
    """
    torch_device = torch.device(device)

    # Load embeddings
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Store: %d passages, dim=%d, %d clusters",
        store.num_passages, store.embedding_dim, store.num_clusters,
    )

    # Faculty with frozen generator — eval mode for full AR decode
    faculty_config = FacultyConfig(
        dim=embedding_dim,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
            max_output_len=max_output_len,
            num_statements=num_statements,
            vq_codebook_path=vq_codebook_path,
            vq_residual_alpha=vq_residual_alpha,
            num_memory_tokens=num_memory_tokens,
        ),
    )

    faculty = LanguageFaculty(faculty_config).to(torch_device)
    gen = faculty.generator
    gen.eval()

    # Trigger lazy init of _input_proj
    with torch.no_grad():
        dummy = torch.randn(1, embedding_dim).to(torch_device)
        faculty(dummy)

    # Message encoder: attention over decoder hidden states
    decoder_hidden = faculty_config.generator.decoder_hidden_dim
    msg_encoder = MessageEncoder(
        decoder_hidden, embedding_dim,
        num_heads=encoder_heads, num_layers=encoder_layers,
    ).to(torch_device)
    receiver = Receiver(embedding_dim).to(torch_device)

    # Collect trainable params
    sender_params = [p for p in gen.parameters() if p.requires_grad]
    encoder_params = list(msg_encoder.parameters())
    receiver_params = list(receiver.parameters())
    all_params = sender_params + encoder_params + receiver_params

    # Single optimizer with per-group learning rates
    optimizer = torch.optim.Adam([
        {"params": sender_params, "lr": sender_lr},
        {"params": encoder_params, "lr": receiver_lr},
        {"params": receiver_params, "lr": receiver_lr},
    ])

    logger.info(
        "Sender params: %d, Encoder params: %d, Receiver params: %d",
        sum(p.numel() for p in sender_params),
        sum(p.numel() for p in encoder_params),
        sum(p.numel() for p in receiver_params),
    )

    # Resume from checkpoint
    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=torch_device, weights_only=False)
        gen._input_proj.load_state_dict(ckpt["input_proj"])
        gen._input_refine.load_state_dict(ckpt["input_refine"])
        msg_encoder.load_state_dict(ckpt["msg_encoder"])
        receiver.load_state_dict(ckpt["receiver"])
        start_step = ckpt.get("step", 0)
        logger.info("Resumed from %s at step %d", resume, start_step)

    embeddings_np = store._embeddings
    cluster_labels = store._cluster_labels
    n = embeddings_np.shape[0]
    num_candidates = num_distractors + 1
    chance = 1.0 / num_candidates
    rng = np.random.default_rng(42)

    # Output directory
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    logger.info(
        "Backprop referential game: %d distractors, chance=%.1f%%, curriculum=%s",
        num_distractors, chance * 100, curriculum,
    )
    if curriculum:
        logger.info(
            "  curriculum: %.0f%% → %.0f%% hard negatives over %d steps",
            curriculum_start * 100, curriculum_end * 100, curriculum_warmup,
        )

    for step in range(start_step, steps):
        # --- Curriculum difficulty ---
        if curriculum:
            frac = min(step / max(curriculum_warmup, 1), 1.0)
            hard_ratio = curriculum_start + frac * (curriculum_end - curriculum_start)
        else:
            hard_ratio = 0.0

        # --- Sample batch ---
        idx = rng.integers(0, n, size=batch_size)
        anchor = torch.tensor(
            embeddings_np[idx], dtype=torch.float32,
        ).to(torch_device)

        # Sample distractors: mix of hard (same cluster) and easy (random)
        dist_indices = np.empty((batch_size, num_distractors), dtype=np.intp)
        for i in range(batch_size):
            n_hard = int(hard_ratio * num_distractors)
            n_easy = num_distractors - n_hard

            hard_idx = np.empty(0, dtype=np.intp)
            if n_hard > 0:
                anchor_cluster = int(cluster_labels[idx[i]])
                hard_idx = store.sample_from_cluster(
                    anchor_cluster, n_hard, rng=rng,
                )
            easy_idx = rng.integers(0, n, size=n_easy)
            dist_indices[i] = np.concatenate([hard_idx, easy_idx])

        distractors = torch.tensor(
            embeddings_np[dist_indices], dtype=torch.float32,
        ).to(torch_device)

        # =============================================================
        # Phase 1: Generate tokens (no_grad, fast KV-cached decode)
        # =============================================================
        with torch.no_grad():
            lfm_outputs = faculty(anchor)

        tokens = lfm_outputs["generator.tokens"]        # (B, S)
        gen_mask = lfm_outputs["generator.mask"]         # (B, S)

        # =============================================================
        # Phase 2: Re-run decoder WITH gradients (parallel, one pass)
        # =============================================================
        # Recompute z from _input_proj with gradients
        gen._ensure_input_proj(anchor.size(-1))
        embeddings_in = anchor.unsqueeze(1)  # (B, 1, dim)
        mask_in = torch.ones(
            batch_size, 1, dtype=torch.bool, device=torch_device,
        )
        pooled = gen._pool(embeddings_in, mask_in)
        h = gen._input_proj(pooled) + gen._input_refine(pooled)
        n_stmt = gen.config.num_statements
        h = h.view(batch_size, n_stmt, gen._latent_dim * 2)
        mu, _ = h.chunk(2, dim=-1)  # (B, N, latent_dim)
        z = mu.reshape(batch_size * n_stmt, gen._latent_dim)

        # Re-run decoder: cross-attention gradients flow to z → _input_proj
        hidden = rerun_decoder_with_grad(gen, z, tokens, gen_mask)

        # =============================================================
        # Encode message and score candidates
        # =============================================================
        message = msg_encoder(hidden, gen_mask)

        candidates = torch.cat(
            [anchor.unsqueeze(1), distractors], dim=1,
        )
        perm = torch.stack([
            torch.randperm(num_candidates, device=torch_device)
            for _ in range(batch_size)
        ])
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = receiver(message, candidates)

        # =============================================================
        # Single loss, single backward
        # =============================================================
        loss = F.cross_entropy(logits, target_idx)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        # --- Logging ---
        if step % log_every == 0:
            with torch.no_grad():
                accuracy = (logits.argmax(1) == target_idx).float().mean().item()
            total_len = gen_mask.float().sum(dim=1).mean().item()

            stmt_lens_t = lfm_outputs.get("generator.statement_lengths")
            if stmt_lens_t is not None:
                avg_per = stmt_lens_t.float().mean(dim=0)
                stmt_str = "/".join(f"{v:.0f}" for v in avg_per.tolist())
                len_info = f"stmt_lens={stmt_str} total={total_len:.0f}"
            else:
                len_info = f"avg_msg_len={total_len:.1f}"

            logger.info(
                "step=%d  loss=%.3f  acc=%.1f%%  "
                "%s  hard=%.0f%%  "
                "(chance=%.1f%%)",
                step,
                loss.item(),
                accuracy * 100,
                len_info,
                hard_ratio * 100,
                chance * 100,
            )

        # --- Periodic checkpoint: latest + best ---
        if step > 0 and step % checkpoint_every == 0:
            with torch.no_grad():
                step_acc = (logits.argmax(1) == target_idx).float().mean().item()
            ckpt = {
                "input_proj": gen._input_proj.state_dict(),
                "input_refine": gen._input_refine.state_dict(),
                "msg_encoder": msg_encoder.state_dict(),
                "receiver": receiver.state_dict(),
                "step": step,
                "accuracy": step_acc,
            }
            latest_path = str(Path(output_dir) / "latest.pt")
            best_path = str(Path(output_dir) / "best.pt")
            torch.save(ckpt, latest_path)
            if step_acc > best_acc:
                best_acc = step_acc
                torch.save(ckpt, best_path)
                logger.info("Checkpoint step %d — new best acc=%.1f%%", step, step_acc * 100)
            else:
                logger.info("Checkpoint step %d — acc=%.1f%% (best=%.1f%%)", step, step_acc * 100, best_acc * 100)

    # --- Save final checkpoint ---
    torch.save(
        {
            "input_proj": gen._input_proj.state_dict(),
            "input_refine": gen._input_refine.state_dict(),
            "msg_encoder": msg_encoder.state_dict(),
            "receiver": receiver.state_dict(),
            "step": steps,
        },
        str(Path(output_dir) / "latest.pt"),
    )
    logger.info("Saved final checkpoint to %s/latest.pt", output_dir)

    with torch.no_grad():
        final_acc = (logits.argmax(1) == target_idx).float().mean().item()
    results = {
        "final_accuracy": final_acc,
        "final_loss": loss.item(),
        "chance": chance,
    }
    logger.info("Final: %s", results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Referential game (backprop)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vq-codebook", default=None, help="Path to VQ codebook")
    parser.add_argument("--vq-alpha", type=float, default=1.0, help="VQ residual alpha")
    parser.add_argument("--num-statements", type=int, default=1)
    parser.add_argument("--max-output-len", type=int, default=96)
    parser.add_argument("--curriculum-warmup", type=int, default=500)
    parser.add_argument("--num-memory-tokens", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--sender-lr", type=float, default=1e-4)
    parser.add_argument("--receiver-lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="data/referential_game")
    args = parser.parse_args()
    main(
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        device=args.device, vq_codebook_path=args.vq_codebook,
        vq_residual_alpha=args.vq_alpha,
        num_statements=args.num_statements, max_output_len=args.max_output_len,
        curriculum_warmup=args.curriculum_warmup,
        num_memory_tokens=args.num_memory_tokens,
        encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads,
        sender_lr=args.sender_lr, receiver_lr=args.receiver_lr,
        output_dir=args.output_dir,
    )
