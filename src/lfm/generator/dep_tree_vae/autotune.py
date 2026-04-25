"""Empirical search over ``_greedy_decode`` hyperparameters.

The decode loop has three knobs that aren't worth retraining for but
that meaningfully affect surface quality:

  - ``eos_boost`` — pushes the EOS logit once length exceeds ``expected_len``
  - ``expected_len`` — target length the EOS pressure starts to grow past
  - ``ngram_block`` — tuple of n-gram widths whose completions are masked

This module evaluates a decoder under arbitrary settings on a fixed
batch of latent vectors and returns ranked results, so callers can
pick a Pareto-good config or sort by whichever metric they care about.

Typical use:

    tuner = DecodeAutotuner(model, sp, cfg, device, val_z=z_batch,
                            source_texts=src_texts, st_model=miniLM)
    results = tuner.grid_search()
    best_cfg, best_metrics = results[0]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.trainer import _greedy_decode


@dataclass(frozen=True)
class DecodeConfig:
    """One concrete setting of decode-time knobs."""

    eos_boost: float = 3.0
    expected_len: int = 13
    ngram_block: tuple[int, ...] = (3, 4)

    def short(self) -> str:
        return (
            f"eos={self.eos_boost:>4.1f} "
            f"len={self.expected_len:>2d} "
            f"ngram={self.ngram_block}"
        )


@dataclass
class DecodeMetrics:
    """Per-config quality summary on the validation batch."""

    mean_decoded_len: float
    length_mae: float | None  # mean abs error vs source length (None w/o sources)
    eos_rate: float
    uniqueness: float          # fraction of distinct decoded outputs
    repetition_rate: float     # fraction of tokens completing a prior trigram
    semantic_score: float | None  # mean cos(MiniLM(decoded), MiniLM(source))
    composite: float           # weighted aggregate, higher = better

    def short(self) -> str:
        sem = f"sem={self.semantic_score:.3f}" if self.semantic_score is not None else "sem=  -  "
        mae = f"mae={self.length_mae:>4.1f}" if self.length_mae is not None else "mae=  -  "
        return (
            f"comp={self.composite:+.3f} "
            f"{sem} {mae} "
            f"len={self.mean_decoded_len:>4.1f} "
            f"eos={self.eos_rate:.0%} "
            f"uniq={self.uniqueness:.0%} "
            f"rep={self.repetition_rate:.2%}"
        )


class DecodeAutotuner:
    """Grid-search decode-time hyperparameters on a fixed latent batch.

    Args:
        model: Frozen DepTreeVAE in eval mode.
        sp: SentencePieceProcessor.
        cfg: The model's DepTreeVAEConfig.
        device: Inference device.
        val_z: ``(B, total_dim)`` latent batch.
        source_texts: Optional reference texts (one per latent), enabling
            length-MAE and semantic-score metrics.
        st_model: Optional SentenceTransformer used to compute semantic
            similarity. Pass any model whose ``.encode`` returns tensors.

    The composite score is documented in :py:meth:`evaluate`.
    """

    # Composite weights — kept explicit so users can override or recompute.
    W_SEMANTIC = 1.0
    W_LENGTH_MAE = 0.02
    W_REPETITION = 0.5
    W_EOS_PENALTY = 0.10
    W_UNIQ_PENALTY = 0.10

    def __init__(
        self,
        model: DepTreeVAE,
        sp,
        cfg: DepTreeVAEConfig,
        device: torch.device,
        val_z: Tensor,
        source_texts: list[str] | None = None,
        st_model=None,
    ) -> None:
        self.model = model
        self.sp = sp
        self.cfg = cfg
        self.device = device
        self.val_z = val_z.to(device)
        self.source_texts = source_texts
        self.st_model = st_model

        self._cache: dict[DecodeConfig, DecodeMetrics] = {}
        self._source_embs: Tensor | None = None
        self._source_lens: np.ndarray | None = None

        if source_texts is not None:
            self._source_lens = np.array([len(t.split()) for t in source_texts])
            if st_model is not None:
                with torch.no_grad():
                    embs = st_model.encode(
                        source_texts,
                        convert_to_tensor=True,
                        device=str(device),
                    )
                self._source_embs = torch.nn.functional.normalize(embs, dim=-1)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _trigram_repetition_rate(texts: list[str]) -> float:
        """Fraction of tokens that complete a trigram seen earlier in the text."""
        rep = 0
        total = 0
        for t in texts:
            words = t.split()
            seen: set[tuple[str, str, str]] = set()
            for i in range(2, len(words)):
                tri = (words[i - 2], words[i - 1], words[i])
                if tri in seen:
                    rep += 1
                seen.add(tri)
                total += 1
        return rep / max(total, 1)

    def _semantic_score(self, decoded: list[str]) -> float | None:
        if self._source_embs is None or self.st_model is None:
            return None
        with torch.no_grad():
            embs = self.st_model.encode(
                decoded, convert_to_tensor=True, device=str(self.device),
            )
        embs = torch.nn.functional.normalize(embs, dim=-1)
        return float((embs * self._source_embs).sum(dim=-1).mean())

    # ------------------------------------------------------------------ public
    def evaluate(self, config: DecodeConfig) -> DecodeMetrics:
        """Decode the validation batch under ``config`` and score it.

        Composite score (higher = better):

            +W_SEMANTIC * semantic_score   (cos similarity to source via MiniLM)
            -W_LENGTH_MAE * length_mae     (raw words; small weight so length is a
                                            tiebreaker, not a dominator)
            -W_REPETITION * repetition_rate
            -W_EOS_PENALTY * (1 - eos_rate)
            -W_UNIQ_PENALTY * (1 - uniqueness)

        When sources are unavailable, semantic_score falls back to uniqueness.
        """
        if config in self._cache:
            return self._cache[config]

        with torch.no_grad():
            decoded = _greedy_decode(
                self.model, self.val_z, self.device, self.cfg, self.sp,
                ngram_block=config.ngram_block,
                eos_boost=config.eos_boost,
                expected_len=config.expected_len,
            )
        texts = [t for t, _ in decoded]
        eos_flags = [eos for _, eos in decoded]

        decoded_lens = np.array([len(t.split()) for t in texts])
        mean_decoded_len = float(decoded_lens.mean())
        length_mae = (
            float(np.abs(decoded_lens - self._source_lens).mean())
            if self._source_lens is not None else None
        )
        eos_rate = sum(eos_flags) / len(eos_flags)
        uniqueness = len(set(texts)) / len(texts)
        repetition_rate = self._trigram_repetition_rate(texts)
        semantic_score = self._semantic_score(texts)

        primary = semantic_score if semantic_score is not None else uniqueness
        composite = (
            self.W_SEMANTIC * primary
            - self.W_LENGTH_MAE * (length_mae or 0.0)
            - self.W_REPETITION * repetition_rate
            - self.W_EOS_PENALTY * (1.0 - eos_rate)
            - self.W_UNIQ_PENALTY * (1.0 - uniqueness)
        )

        metrics = DecodeMetrics(
            mean_decoded_len=mean_decoded_len,
            length_mae=length_mae,
            eos_rate=eos_rate,
            uniqueness=uniqueness,
            repetition_rate=repetition_rate,
            semantic_score=semantic_score,
            composite=composite,
        )
        self._cache[config] = metrics
        return metrics

    def grid_search(
        self,
        eos_boosts: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0),
        expected_lens: Sequence[int] = (10, 12, 13, 14, 15),
        ngram_blocks: Sequence[tuple[int, ...]] = ((3,), (3, 4), (2, 3, 4), (3, 4, 5)),
        verbose: bool = True,
    ) -> list[tuple[DecodeConfig, DecodeMetrics]]:
        """Evaluate every combination of the given grids; return ranked by composite."""
        combos = list(product(eos_boosts, expected_lens, ngram_blocks))
        results: list[tuple[DecodeConfig, DecodeMetrics]] = []
        for i, (eb, el, nb) in enumerate(combos):
            cfg = DecodeConfig(eos_boost=float(eb), expected_len=int(el), ngram_block=tuple(nb))
            m = self.evaluate(cfg)
            results.append((cfg, m))
            if verbose:
                print(f"[{i+1:>3}/{len(combos)}] {cfg.short()}  →  {m.short()}")
        results.sort(key=lambda x: x[1].composite, reverse=True)
        return results

    def top_by(
        self,
        results: list[tuple[DecodeConfig, DecodeMetrics]],
        metric: str,
        k: int = 5,
        higher_is_better: bool | None = None,
    ) -> list[tuple[DecodeConfig, DecodeMetrics]]:
        """Sort ``results`` by an arbitrary metric name and return the top ``k``.

        Default direction is inferred: ``length_mae`` and ``repetition_rate``
        prefer lower; everything else prefers higher.
        """
        lower_is_better = {"length_mae", "repetition_rate", "mean_decoded_len"}
        if higher_is_better is None:
            higher_is_better = metric not in lower_is_better
        keyed = [(cfg, m, getattr(m, metric)) for cfg, m in results]
        keyed = [t for t in keyed if t[2] is not None]
        keyed.sort(key=lambda t: t[2], reverse=higher_is_better)
        return [(cfg, m) for cfg, m, _ in keyed[:k]]
