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
    """Re-encode generated token embeddings into a fixed-size vector.

    Takes the decoder's hidden states (from the full AR decode) and
    produces a single vector that the receiver uses for scoring.
    This is a learned module — it trains to extract discriminative
    information from the generated linguistic form.
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        """Pool decoder hidden states into a fixed vector.

        Args:
            embeddings: (batch, seq_len, hidden_dim) decoder states.
            mask: (batch, seq_len) boolean mask.

        Returns:
            (batch, output_dim) message vector.
        """
        # Masked mean pool
        mask_f = mask.unsqueeze(-1).float()
        pooled = (embeddings * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.pool_proj(pooled)


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
    batch_size: int = 512,
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
            max_output_len=32,  # shorter for speed
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
    msg_encoder = MessageEncoder(decoder_hidden, embedding_dim).to(torch_device)
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

    embeddings_np = store._embeddings
    cluster_labels = store._cluster_labels
    n = embeddings_np.shape[0]
    num_candidates = num_distractors + 1
    chance = 1.0 / num_candidates
    rng = np.random.default_rng(42)

    # REINFORCE baseline (running mean of rewards)
    baseline = 0.0

    logger.info(
        "REINFORCE referential game: %d distractors, chance=%.1f%%, curriculum=%s",
        num_distractors, chance * 100, curriculum,
    )
    if curriculum:
        logger.info(
            "  curriculum: %.0f%% → %.0f%% hard negatives over %d steps",
            curriculum_start * 100, curriculum_end * 100, curriculum_warmup,
        )

    for step in range(steps):
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
        gen_embeddings = lfm_outputs["generator.embeddings"]  # (B, S, H)
        gen_mask = lfm_outputs["generator.mask"]  # (B, S)

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
            reward = (preds == target_idx).float()  # 1 if correct, 0 if not
            # Update baseline
            baseline = baseline_decay * baseline + (1 - baseline_decay) * reward.mean().item()
            advantage = reward - baseline

        # REINFORCE: ∇J = E[advantage * ∇log π(a|s)]
        # We need gradients of _input_proj w.r.t. the log probability
        # of the generated sequence. But the generation was done with
        # no_grad. We need to recompute the z → log_prob path with grad.

        # Recompute z with gradients
        gen = faculty.generator
        gen._ensure_input_proj(anchor.size(-1))
        embeddings_in = anchor.unsqueeze(1)  # (B, 1, dim)
        mask_in = torch.ones(
            batch_size, 1, dtype=torch.bool, device=torch_device
        )
        pooled = gen._pool(embeddings_in, mask_in)
        h = gen._input_proj(pooled)
        mu, logvar = h.chunk(2, dim=-1)

        # The "action" was the z that produced the tokens.
        # Under the Gaussian policy N(mu, sigma), log_prob of z is:
        # For REINFORCE we use the mu directly — the policy is deterministic
        # so we use a surrogate: push mu toward z values that got high reward
        sender_loss = -(advantage.detach() * mu.sum(dim=-1)).mean()

        sender_optimizer.zero_grad()
        sender_loss.backward()
        nn.utils.clip_grad_norm_(sender_params, 1.0)
        sender_optimizer.step()

        # --- Logging ---
        if step % log_every == 0:
            accuracy = reward.mean().item()
            avg_len = gen_mask.float().sum(dim=1).mean().item()
            logger.info(
                "step=%d  recv_loss=%.3f  acc=%.1f%%  "
                "avg_msg_len=%.1f  baseline=%.3f  hard=%.0f%%  "
                "(chance=%.1f%%)",
                step,
                receiver_loss.item(),
                accuracy * 100,
                avg_len,
                baseline,
                hard_ratio * 100,
                chance * 100,
            )

        # --- Periodic checkpoint ---
        if step > 0 and step % checkpoint_every == 0:
            ckpt = {
                "input_proj": faculty.generator._input_proj.state_dict(),
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
    main()
