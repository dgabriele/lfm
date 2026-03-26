"""Referential game with REINFORCE through the linguistic bottleneck.

The sender's embedding is projected to z, the frozen decoder generates
variable-length IPA text (the "message"), the message is re-encoded
into a fixed vector, and the receiver scores candidates against it.

The generated text IS the bottleneck — not z. Complex embeddings should
produce longer, more detailed utterances; simple ones produce shorter.

Since text generation is non-differentiable, we use REINFORCE:
  - Action: the generated token sequence (from frozen decoder)
  - Reward: whether the receiver correctly identified the target
  - Policy: _input_proj (maps embedding → z, which conditions the decoder)

The receiver and message re-encoder are trained with standard backprop
(they operate on the discrete text, not through the decoder).

Usage::

    python scripts/generate_synthetic_embeddings.py  # or real embeddings
    python scripts/run_referential_reinforce.py
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class MessageEncoder(nn.Module):
    """Re-encode N variable-length statement hidden states into a fixed vector.

    For each statement, mean-pools the decoder hidden states into a
    statement-level embedding, then uses learned cross-statement attention
    to aggregate N statement embeddings into one message vector.

    When num_statements=1, degrades to the original single-pool behavior.
    """

    def __init__(
        self, hidden_dim: int, output_dim: int, num_statements: int = 1,
    ) -> None:
        super().__init__()
        self.num_statements = num_statements
        self.statement_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        if num_statements > 1:
            self.agg_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
            self.agg_attn = nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=4, batch_first=True,
            )
        else:
            self.agg_query = None
            self.agg_attn = None

    def forward(
        self,
        statement_embeddings: Tensor,
        statement_mask: Tensor,
    ) -> Tensor:
        """Pool N statement hidden states into a single message vector.

        Args:
            statement_embeddings: (batch, N, S, H) or (batch, S, H) for N=1.
            statement_mask: (batch, N, S) or (batch, S) for N=1.

        Returns:
            (batch, output_dim) message vector.
        """
        if statement_embeddings.dim() == 3:
            # Single statement — backward compat: (B, S, H) → (B, 1, S, H)
            statement_embeddings = statement_embeddings.unsqueeze(1)
            statement_mask = statement_mask.unsqueeze(1)

        b, n, s, h = statement_embeddings.shape

        # 1. Mean-pool each statement independently
        mask_f = statement_mask.unsqueeze(-1).float()  # (B, N, S, 1)
        pooled = (
            (statement_embeddings * mask_f).sum(dim=2)
            / mask_f.sum(dim=2).clamp(min=1)
        )  # (B, N, H)

        # 2. Project each statement
        stmt_vecs = self.statement_proj(pooled)  # (B, N, output_dim)

        # 3. Aggregate across statements
        if n == 1 or self.agg_attn is None:
            return stmt_vecs.squeeze(1)

        query = self.agg_query.expand(b, -1, -1)  # (B, 1, output_dim)
        agg, _ = self.agg_attn(query, stmt_vecs, stmt_vecs)
        return agg.squeeze(1)  # (B, output_dim)


class Receiver(nn.Module):
    """Score candidates against the message via learned dot-product."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        """Score candidates.

        Args:
            message: (batch, dim) message vector.
            candidates: (batch, K, dim) candidate embeddings.

        Returns:
            (batch, K) logits.
        """
        projected = self.proj(message)  # (B, dim)
        return torch.bmm(
            projected.unsqueeze(1), candidates.transpose(1, 2)
        ).squeeze(1)


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    steps: int = 2000,
    sender_lr: float = 1e-4,
    receiver_lr: float = 1e-3,
    batch_size: int = 128,
    num_distractors: int = 15,
    embedding_dim: int = 384,
    device: str = "cuda",
    log_every: int = 50,
    checkpoint_every: int = 100,
    baseline_decay: float = 0.99,
    curriculum: bool = True,
    curriculum_warmup: int = 500,
    curriculum_start: float = 0.0,
    curriculum_end: float = 1.0,
    resume: str | None = None,
    vq_codebook_path: str | None = None,
    vq_residual_alpha: float = 1.0,
    length_cost: float = 0.0,
    num_statements: int = 1,
    max_output_len: int = 96,
) -> dict[str, float]:
    """Run the REINFORCE referential game.

    Two optimizers:
    - Sender (REINFORCE): _input_proj learns which z produces messages
      that help the receiver succeed.
    - Receiver (backprop): message_encoder + receiver learn to extract
      discriminative info from the generated text.
    """
    torch_device = torch.device(device)

    # Load embeddings
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Store: %d passages, dim=%d, %d clusters",
        store.num_passages, store.embedding_dim, store.num_clusters,
    )

    # Faculty with frozen generator — full AR decode (eval mode for decoder)
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
        ),
    )

    faculty = LanguageFaculty(faculty_config).to(torch_device)

    # We need the generator to do full AR decode, not the direct z path.
    # Set eval mode so freeze_decoder + training doesn't trigger the shortcut.
    faculty.generator.eval()

    # Trigger lazy init of _input_proj by running one forward pass
    with torch.no_grad():
        dummy = torch.randn(1, embedding_dim).to(torch_device)
        faculty(dummy)

    # Message encoder: decoder_hidden_dim → embedding_dim
    decoder_hidden = faculty_config.generator.decoder_hidden_dim
    msg_encoder = MessageEncoder(
        decoder_hidden, embedding_dim, num_statements,
    ).to(torch_device)
    receiver = Receiver(embedding_dim).to(torch_device)

    # Sender params (REINFORCE — only _input_proj)
    sender_params = [
        p for p in faculty.generator.parameters() if p.requires_grad
    ]
    sender_optimizer = torch.optim.Adam(sender_params, lr=sender_lr)

    # Receiver params (backprop)
    receiver_params = list(msg_encoder.parameters()) + list(receiver.parameters())
    receiver_optimizer = torch.optim.Adam(receiver_params, lr=receiver_lr)

    logger.info(
        "Sender params: %d, Receiver params: %d",
        sum(p.numel() for p in sender_params),
        sum(p.numel() for p in receiver_params),
    )

    # Resume from checkpoint
    start_step = 0
    baseline = 0.0
    if resume is not None:
        ckpt = torch.load(resume, map_location=torch_device, weights_only=False)
        faculty.generator._input_proj.load_state_dict(ckpt["input_proj"])
        faculty.generator._input_refine.load_state_dict(ckpt["input_refine"])
        msg_encoder.load_state_dict(ckpt["msg_encoder"])
        receiver.load_state_dict(ckpt["receiver"])
        start_step = ckpt.get("step", 0)
        baseline = ckpt.get("baseline", 0.0)
        logger.info("Resumed from %s at step %d (baseline=%.3f)", resume, start_step, baseline)

    embeddings_np = store._embeddings
    cluster_labels = store._cluster_labels
    n = embeddings_np.shape[0]
    num_candidates = num_distractors + 1
    chance = 1.0 / num_candidates
    rng = np.random.default_rng(42)

    logger.info(
        "REINFORCE referential game: %d distractors, chance=%.1f%%, curriculum=%s",
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
            embeddings_np[idx], dtype=torch.float32
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
            embeddings_np[dist_indices], dtype=torch.float32
        ).to(torch_device)

        # --- Sender: generate message via frozen decoder ---
        # Need to capture log_prob of the generated tokens for REINFORCE
        with torch.no_grad():
            lfm_outputs = faculty(anchor)

        # Extract decoder hidden states and mask
        gen_embeddings = lfm_outputs["generator.embeddings"]  # (B, N*S, H)
        gen_mask = lfm_outputs["generator.mask"]  # (B, N*S)
        # Statement-level data for multi-statement MessageEncoder
        gen_stmt_emb = lfm_outputs.get("generator.statement_embeddings")  # (B, N, S, H)
        gen_stmt_mask = lfm_outputs.get("generator.statement_mask")  # (B, N, S)

        # Also get the log probability of the generated sequence
        # from the generator's token_probs
        token_probs = lfm_outputs["generator.token_probs"]  # (B, S, V)
        tokens = lfm_outputs["generator.tokens"]  # (B, S)

        # Log prob of each generated token under the policy
        log_probs_all = torch.log(
            token_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1) + 1e-10
        )  # (B, S)
        # Mask and sum for sequence log prob
        (log_probs_all * gen_mask.float()).sum(dim=1)  # (B,)

        # --- Receiver: score candidates ---
        if gen_stmt_emb is not None:
            message = msg_encoder(gen_stmt_emb.detach(), gen_stmt_mask.detach())
        else:
            message = msg_encoder(gen_embeddings.detach(), gen_mask.detach())

        candidates = torch.cat(
            [anchor.unsqueeze(1), distractors], dim=1
        )
        perm = torch.stack(
            [torch.randperm(num_candidates, device=torch_device)
             for _ in range(batch_size)]
        )
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = receiver(message, candidates)

        # --- Receiver loss (backprop) ---
        receiver_loss = F.cross_entropy(logits, target_idx)
        receiver_optimizer.zero_grad()
        receiver_loss.backward()
        nn.utils.clip_grad_norm_(receiver_params, 1.0)
        receiver_optimizer.step()

        # --- Sender loss (REINFORCE) ---
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == target_idx).float()  # 1 if correct, 0 if not
            # Length-penalized reward: correct predictions are discounted by
            # message length.  Incorrect predictions always get 0.  This
            # incentivizes the shortest message that still discriminates.
            msg_lengths = gen_mask.float().sum(dim=1)  # (B,)
            max_len = gen_mask.size(1)
            reward = correct * (1.0 - length_cost * msg_lengths / max_len)
            # Update baseline
            baseline = baseline_decay * baseline + (1 - baseline_decay) * reward.mean().item()
            advantage = reward - baseline

        # REINFORCE: ∇J = E[advantage * ∇log π(a|s)]
        # We need gradients of _input_proj w.r.t. the log probability
        # of the generated sequence. But the generation was done with
        # no_grad. We need to recompute the z → log_prob path with grad.

        # Recompute z with gradients
        gen = faculty.generator
        n_stmt = gen.config.num_statements
        gen._ensure_input_proj(anchor.size(-1))
        embeddings_in = anchor.unsqueeze(1)  # (B, 1, dim)
        mask_in = torch.ones(
            batch_size, 1, dtype=torch.bool, device=torch_device
        )
        pooled = gen._pool(embeddings_in, mask_in)
        h = gen._input_proj(pooled) + gen._input_refine(pooled)
        h = h.view(batch_size, n_stmt, gen._latent_dim * 2)
        mu, logvar = h.chunk(2, dim=-1)  # each (B, N, latent_dim)

        # Surrogate: push all N mu vectors toward z values that got high reward.
        # Advantage (B,) broadcasts to all N statements for the same sample.
        sender_loss = -(
            advantage.detach().unsqueeze(-1).unsqueeze(-1) * mu
        ).sum(dim=(-2, -1)).mean()

        sender_optimizer.zero_grad()
        sender_loss.backward()
        nn.utils.clip_grad_norm_(sender_params, 1.0)
        sender_optimizer.step()

        # --- Logging ---
        if step % log_every == 0:
            accuracy = reward.mean().item()
            total_len = gen_mask.float().sum(dim=1).mean().item()
            # Per-statement length breakdown
            stmt_lens_t = lfm_outputs.get("generator.statement_lengths")
            if stmt_lens_t is not None:
                avg_per = stmt_lens_t.float().mean(dim=0)  # (N,)
                stmt_str = "/".join(f"{v:.0f}" for v in avg_per.tolist())
                len_info = f"stmt_lens={stmt_str} total={total_len:.0f}"
            else:
                len_info = f"avg_msg_len={total_len:.1f}"
            logger.info(
                "step=%d  recv_loss=%.3f  acc=%.1f%%  "
                "%s  baseline=%.3f  hard=%.0f%%  "
                "(chance=%.1f%%)",
                step,
                receiver_loss.item(),
                accuracy * 100,
                len_info,
                baseline,
                hard_ratio * 100,
                chance * 100,
            )

        # --- Periodic checkpoint ---
        if step > 0 and step % checkpoint_every == 0:
            ckpt = {
                "input_proj": faculty.generator._input_proj.state_dict(),
                "input_refine": faculty.generator._input_refine.state_dict(),
                "msg_encoder": msg_encoder.state_dict(),
                "receiver": receiver.state_dict(),
                "step": step,
                "baseline": baseline,
            }
            torch.save(ckpt, f"data/input_proj_step{step}.pt")
            torch.save(ckpt, "data/input_proj.pt")  # always keep latest
            logger.info("Checkpoint saved at step %d", step)

    # --- Save final checkpoint ---
    torch.save(
        {
            "input_proj": faculty.generator._input_proj.state_dict(),
            "input_refine": faculty.generator._input_refine.state_dict(),
            "msg_encoder": msg_encoder.state_dict(),
            "receiver": receiver.state_dict(),
            "step": steps,
            "baseline": baseline,
        },
        "data/input_proj.pt",
    )
    logger.info("Saved final checkpoint to data/input_proj.pt")

    final_acc = reward.mean().item()
    results = {
        "final_accuracy": final_acc,
        "final_receiver_loss": receiver_loss.item(),
        "baseline": baseline,
        "chance": chance,
    }
    logger.info("Final: %s", results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="REINFORCE referential game")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vq-codebook", default=None, help="Path to VQ codebook")
    parser.add_argument("--vq-alpha", type=float, default=1.0, help="VQ residual alpha")
    parser.add_argument("--length-cost", type=float, default=0.0, help="Length penalty")
    parser.add_argument("--num-statements", type=int, default=1, help="Statements per input")
    parser.add_argument("--max-output-len", type=int, default=96, help="Max tokens per statement")
    args = parser.parse_args()
    main(
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        device=args.device, vq_codebook_path=args.vq_codebook,
        vq_residual_alpha=args.vq_alpha, length_cost=args.length_cost,
        num_statements=args.num_statements, max_output_len=args.max_output_len,
    )
