"""MUSE-style unsupervised cross-lingual embedding alignment.

Implements the Lample/Conneau 2018 recipe for aligning two monolingual
embedding spaces with zero paired data:

1. **Adversarial Procrustes**: a discriminator MLP tries to classify
   whether a vector is from space X (mapped via rotation ``W``) or
   space Y (unmodified).  ``W`` is trained adversarially to fool the
   discriminator.  After each gradient step ``W`` is projected back
   onto the manifold of orthogonal matrices via a simple Newton
   iteration, preserving the "rotation" constraint.

2. **CSLS-refined Procrustes**: after the adversarial phase produces a
   rough rotation, construct a pseudo-dictionary from the top-k mutual
   nearest neighbors under the Cross-Domain Similarity Local Scaling
   metric (which corrects for the hubness problem of raw cosine).
   Solve the orthogonal Procrustes problem in closed form on that
   pseudo-dictionary to get an improved ``W``.  Iterate a few rounds.

The output is a rotation matrix ``W`` such that ``source @ W.T ≈
target`` in semantic terms: for any source embedding, multiplying by
``W`` moves it into the target language's coordinate frame, where
cosine similarity can be used as a translation signal in the Stage 3
training loop.

Reference: github.com/facebookresearch/MUSE  (we do not import it;
the math is re-implemented directly).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lfm.unmt.config import UNMTConfig
from lfm.unmt.embeddings.skipgram import load_embeddings

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Output of MUSE alignment.

    Attributes:
        W: Rotation matrix ``(D, D)`` mapping source → target.
        source_lang: Source language code.
        target_lang: Target language code.
        adversarial_loss: Final adversarial loss before refinement.
        refinement_rounds: Number of Procrustes refinement rounds applied.
        final_csls_mean: Mean CSLS score across the pseudo-dictionary
            at the end of refinement.
    """

    W: torch.Tensor
    source_lang: str
    target_lang: str
    adversarial_loss: float
    refinement_rounds: int
    final_csls_mean: float


class Discriminator(nn.Module):
    """Two-hidden-layer MLP that classifies source-mapped vs target vectors.

    Architecture follows the original MUSE implementation: two hidden
    layers of size ``hidden``, leaky ReLU activations, dropout, and a
    single sigmoid-headed output.  Dropout is also applied to the
    input layer.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden: int = 2048,
        layers: int = 2,
        input_dropout: float = 0.1,
        hidden_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Dropout(input_dropout)]
        prev = embed_dim
        for _ in range(layers):
            modules.append(nn.Linear(prev, hidden))
            modules.append(nn.LeakyReLU(0.2))
            if hidden_dropout > 0:
                modules.append(nn.Dropout(hidden_dropout))
            prev = hidden
        modules.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _orthogonalize(W: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """Push ``W`` toward the nearest orthogonal matrix in-place-safe form.

    Uses a single Newton step: ``W ← (1 + β) W - β W Wᵀ W``.  This is
    the update rule from Cisse et al. 2017 that preserves the spectral
    properties of W close to the orthogonal manifold without doing a
    full SVD at every gradient step.
    """
    return (1 + beta) * W - beta * W @ W.t() @ W


def adversarial_align(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    device: torch.device,
    *,
    dico_max_rank: int = 15_000,
    n_epochs: int = 5,
    epoch_size: int = 1_000_000,
    batch_size: int = 128,
    lr: float = 0.1,
    smoothing: float = 0.1,
    beta: float = 0.01,
    hidden: int = 2048,
    hidden_layers: int = 2,
) -> tuple[torch.Tensor, float]:
    """Adversarial Procrustes alignment, returns ``(W, final_loss)``.

    Trains the rotation matrix to map ``source_emb`` into the
    ``target_emb`` space such that a discriminator cannot distinguish
    mapped-source from target.  Only the top ``dico_max_rank`` most
    frequent tokens from each language are used during adversarial
    training — rare tokens have noisier embeddings and destabilize the
    GAN.

    Args:
        source_emb: ``(V_src, D)`` source-language embeddings.
        target_emb: ``(V_tgt, D)`` target-language embeddings.
        device: Compute device.
        dico_max_rank: Use only the first ``dico_max_rank`` rows of
            each embedding matrix for training.  Assumes the rows are
            ordered by frequency (sentencepiece does not guarantee
            this, but it holds approximately for BPE output after the
            special-token prefix).
        n_epochs: Number of adversarial epochs.
        epoch_size: Number of discriminator/W updates per epoch.
        batch_size: Vectors sampled per update on each side.
        lr: SGD learning rate for both D and W.
        smoothing: Label smoothing for the discriminator targets.
        beta: Orthogonalization step size for W after each update.
        hidden: Hidden size of the discriminator MLP.
        hidden_layers: Number of hidden layers in the discriminator.

    Returns:
        ``(W, final_loss)`` where ``W`` is a ``(D, D)`` rotation matrix
        on CPU and ``final_loss`` is the mean discriminator loss on
        the last epoch.
    """
    D = source_emb.size(1)
    assert target_emb.size(1) == D, "embed_dim mismatch"

    # Normalize the embeddings to the unit sphere — MUSE does this
    # because cosine distance is what we ultimately align on.
    source = F.normalize(source_emb[:dico_max_rank].to(device), dim=-1)
    target = F.normalize(target_emb[:dico_max_rank].to(device), dim=-1)

    W = torch.eye(D, device=device)
    W.requires_grad_(True)

    disc = Discriminator(D, hidden=hidden, layers=hidden_layers).to(device)

    opt_W = torch.optim.SGD([W], lr=lr)
    opt_D = torch.optim.SGD(disc.parameters(), lr=lr)

    final_loss = 0.0
    for epoch in range(n_epochs):
        epoch_d_loss = 0.0
        epoch_w_loss = 0.0
        n_updates = 0
        for step in range(epoch_size // batch_size):
            # Sample equal-sized batches of source and target.
            idx_s = torch.randint(0, source.size(0), (batch_size,), device=device)
            idx_t = torch.randint(0, target.size(0), (batch_size,), device=device)
            src_batch = source[idx_s]
            tgt_batch = target[idx_t]

            # ---- discriminator step ----
            src_mapped = src_batch @ W.t()
            x = torch.cat([src_mapped.detach(), tgt_batch], dim=0)
            # Label smoothing:
            #   real target → 1 - smoothing
            #   mapped src  → smoothing
            y = torch.cat([
                torch.full((batch_size,), smoothing, device=device),
                torch.full((batch_size,), 1.0 - smoothing, device=device),
            ])
            d_out = disc(x)
            d_loss = F.binary_cross_entropy_with_logits(d_out, y)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            epoch_d_loss += d_loss.item()

            # ---- W step (fool discriminator) ----
            src_mapped = src_batch @ W.t()
            y_fool = torch.full((batch_size,), 1.0 - smoothing, device=device)
            w_loss = F.binary_cross_entropy_with_logits(
                disc(src_mapped), y_fool,
            )
            opt_W.zero_grad()
            w_loss.backward()
            opt_W.step()
            epoch_w_loss += w_loss.item()

            with torch.no_grad():
                W.copy_(_orthogonalize(W, beta=beta))

            n_updates += 1

        avg_d = epoch_d_loss / max(n_updates, 1)
        avg_w = epoch_w_loss / max(n_updates, 1)
        logger.info(
            "  adv epoch %d/%d: D_loss=%.4f W_loss=%.4f updates=%d",
            epoch + 1, n_epochs, avg_d, avg_w, n_updates,
        )
        final_loss = avg_d

    return W.detach().cpu(), final_loss


def _csls_nearest(
    source: torch.Tensor,
    target: torch.Tensor,
    k_local: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each source vector, return its nearest-target index by CSLS.

    CSLS corrects cosine similarity for hubness: for a source ``s`` and
    candidate target ``t``, the CSLS score is
    ``2 * cos(s, t) - mean_k(cos(s, T_k)) - mean_k(cos(t, S_k))`` where
    the means are taken over the ``k_local`` nearest neighbors in the
    *other* language.

    Both ``source`` and ``target`` are assumed to be unit-normalized.
    Returns ``(nearest_indices, nearest_scores)``, one entry per row of
    source.
    """
    # r_target[i] = mean cosine between target[i] and its k nearest sources
    sim_t_to_s = target @ source.t()           # (V_tgt, V_src)
    top_t, _ = sim_t_to_s.topk(k_local, dim=1)  # (V_tgt, k)
    r_target = top_t.mean(dim=1)                # (V_tgt,)

    # r_source[j] = mean cosine between source[j] and its k nearest targets
    sim_s_to_t = source @ target.t()           # (V_src, V_tgt)
    top_s, _ = sim_s_to_t.topk(k_local, dim=1)  # (V_src, k)
    r_source = top_s.mean(dim=1)                # (V_src,)

    # CSLS(s, t) = 2 * cos(s, t) - r_source[s] - r_target[t]
    csls = 2 * sim_s_to_t - r_source.unsqueeze(1) - r_target.unsqueeze(0)
    nearest_scores, nearest_idx = csls.max(dim=1)
    return nearest_idx, nearest_scores


def _build_pseudo_dictionary(
    source: torch.Tensor,
    target: torch.Tensor,
    k_local: int = 10,
    mutual_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build a pseudo-dictionary by mutual-NN CSLS.

    Returns ``(src_ids, tgt_ids, mean_csls_of_selected)``.
    """
    src_to_tgt, src_scores = _csls_nearest(source, target, k_local=k_local)
    if not mutual_only:
        return (
            torch.arange(source.size(0), device=source.device),
            src_to_tgt,
            src_scores.mean().item(),
        )

    # Reverse direction — for each target, its nearest source by CSLS.
    tgt_to_src, _ = _csls_nearest(target, source, k_local=k_local)

    src_ids = torch.arange(source.size(0), device=source.device)
    reverse = tgt_to_src[src_to_tgt]  # what does *its* nearest point back to
    mutual = reverse == src_ids
    selected_src = src_ids[mutual]
    selected_tgt = src_to_tgt[mutual]
    selected_scores = src_scores[mutual]
    return selected_src, selected_tgt, selected_scores.mean().item() if selected_scores.numel() else 0.0


def procrustes_refine(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    W: torch.Tensor,
    device: torch.device,
    *,
    n_rounds: int = 5,
    dico_max_rank: int = 15_000,
    k_csls: int = 10,
) -> tuple[torch.Tensor, int, float]:
    """Refine ``W`` by iterative closed-form Procrustes on CSLS mutual-NN.

    Each round applies the current ``W`` to source embeddings, builds a
    pseudo-dictionary from top mutual CSLS pairs, and solves the
    orthogonal Procrustes problem in closed form on that dictionary.

    Returns ``(W_refined, rounds_completed, final_mean_csls)``.
    """
    source_all = F.normalize(source_emb[:dico_max_rank].to(device), dim=-1)
    target_all = F.normalize(target_emb[:dico_max_rank].to(device), dim=-1)
    W = W.to(device)

    last_mean_csls = 0.0
    completed = 0
    for round_idx in range(n_rounds):
        src_mapped = F.normalize(source_all @ W.t(), dim=-1)
        src_ids, tgt_ids, mean_csls = _build_pseudo_dictionary(
            src_mapped, target_all, k_local=k_csls, mutual_only=True,
        )
        n_pairs = src_ids.numel()
        logger.info(
            "  procrustes round %d: dict_size=%d mean_csls=%.4f",
            round_idx + 1, n_pairs, mean_csls,
        )
        if n_pairs < 10:
            logger.warning(
                "  refinement dictionary has only %d pairs — stopping early",
                n_pairs,
            )
            break

        X = source_all[src_ids]     # (N, D) — raw (not W-mapped) source
        Y = target_all[tgt_ids]     # (N, D) — target
        # W minimizes ||W X^T - Y^T|| s.t. W orthogonal.
        # Closed form: W = V U^T where U S V^T = Y^T X.
        M = Y.t() @ X               # (D, D)
        U, _, Vh = torch.linalg.svd(M)
        W = U @ Vh
        last_mean_csls = mean_csls
        completed = round_idx + 1

    return W.detach().cpu(), completed, last_mean_csls


def run_muse_alignment(
    config: UNMTConfig,
    source_lang: str = "ng",
    target_lang: str = "en",
    *,
    n_adv_epochs: int = 5,
    adv_epoch_size: int = 1_000_000,
    adv_batch_size: int = 128,
    adv_lr: float = 0.1,
    n_refine_rounds: int = 5,
    dico_max_rank: int = 15_000,
    k_csls: int = 10,
) -> AlignmentResult:
    """End-to-end MUSE alignment.

    Loads the monolingual embedding matrices produced by Stage 2a,
    runs adversarial alignment to get an initial rotation ``W``, then
    refines with iterative CSLS-based Procrustes.  Saves the result to
    ``<output_dir>/alignment_<src>2<tgt>.pt``.
    """
    output_dir = Path(config.output_dir)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    ng_path = output_dir / "embed_neuroglot.pt"
    en_path = output_dir / "embed_english.pt"
    ng_weights, ng_meta = load_embeddings(ng_path)
    en_weights, en_meta = load_embeddings(en_path)

    if source_lang == "ng" and target_lang == "en":
        src_emb, tgt_emb = ng_weights, en_weights
    elif source_lang == "en" and target_lang == "ng":
        src_emb, tgt_emb = en_weights, ng_weights
    else:
        raise ValueError(
            f"Unknown alignment direction: {source_lang}→{target_lang}"
        )

    logger.info(
        "MUSE alignment %s → %s: src=%s tgt=%s dim=%d",
        source_lang, target_lang,
        tuple(src_emb.shape), tuple(tgt_emb.shape),
        src_emb.size(1),
    )

    logger.info("Phase 1: adversarial Procrustes")
    W, adv_loss = adversarial_align(
        src_emb, tgt_emb, device,
        dico_max_rank=dico_max_rank,
        n_epochs=n_adv_epochs,
        epoch_size=adv_epoch_size,
        batch_size=adv_batch_size,
        lr=adv_lr,
    )

    logger.info("Phase 2: CSLS-refined Procrustes")
    W, rounds, mean_csls = procrustes_refine(
        src_emb, tgt_emb, W, device,
        n_rounds=n_refine_rounds,
        dico_max_rank=dico_max_rank,
        k_csls=k_csls,
    )

    result = AlignmentResult(
        W=W,
        source_lang=source_lang,
        target_lang=target_lang,
        adversarial_loss=adv_loss,
        refinement_rounds=rounds,
        final_csls_mean=mean_csls,
    )

    out_path = output_dir / f"alignment_{source_lang}2{target_lang}.pt"
    torch.save(
        {
            "W": W,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "adversarial_loss": adv_loss,
            "refinement_rounds": rounds,
            "final_csls_mean": mean_csls,
        },
        out_path,
    )
    logger.info("Saved alignment → %s", out_path)
    return result


def load_alignment(path: Path) -> AlignmentResult:
    """Load a saved alignment result from disk."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return AlignmentResult(
        W=blob["W"],
        source_lang=blob["source_lang"],
        target_lang=blob["target_lang"],
        adversarial_loss=blob["adversarial_loss"],
        refinement_rounds=blob["refinement_rounds"],
        final_csls_mean=blob["final_csls_mean"],
    )
