"""Random-z decode quality — how well does the decoder handle arbitrary
latent vectors sampled from the prior.

For LFM's thesis (any continuous representation → valid constituent),
this is the core capability test.  We sample N vectors from
``N(0, z_std²)`` where ``z_std`` is the empirical per-dimension std
recorded in the checkpoint (falling back to 1.0), decode them, and
report:

  * tag-validity rate: fraction with a matched ``<TAG>…</TAG>`` pair
  * unique-output rate: unique decoded strings / N
  * length distribution of outputs
  * distribution of outer-tag types (S / NP / VP / PP / …)

Figures
-------
``tag_validity``  — bar chart of matched / mismatched / unmatched
``length_hist``   — histogram of decoded character lengths
``tag_types``     — bar chart of outer-tag frequencies
"""

from __future__ import annotations

import logging
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm_lib
import torch
from matplotlib.figure import Figure

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"^<(\w+)>\s+.+?\s+</(\w+)>$")
_OPEN_TAG_RE = re.compile(r"^<(\w+)>")


def _classify(line: str) -> tuple[str, str | None]:
    """Return ("matched" | "mismatched" | "unmatched", open_tag).

    "matched"    → <NP> … </NP>
    "mismatched" → <NP> … </VP>
    "unmatched"  → anything else (missing open/close, empty, etc.)
    """
    m = _TAG_RE.match(line.strip())
    if m:
        return ("matched" if m.group(1) == m.group(2) else "mismatched"), m.group(1)
    o = _OPEN_TAG_RE.match(line.strip())
    return "unmatched", (o.group(1) if o else None)


class RandomZQualityVisualization(BaseVisualization):
    """Report the decoder's behavior on vectors drawn from the prior."""

    name = "random_z_quality"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    def generate(self, data: dict) -> list[Figure]:
        model_data = data["model_data"]
        n = data.get("num_samples", 1000)
        seed = data.get("seed", 42)
        device = model_data["device"]

        z_std = model_data.get("z_std")
        cfg = model_data["cfg"]
        latent_dim = cfg.latent_dim
        if isinstance(z_std, torch.Tensor):
            z_std_t = z_std.to(device).float()
            if z_std_t.numel() == 1:
                z_std_t = z_std_t.expand(latent_dim)
        elif z_std is None:
            logger.warning("checkpoint has no z_std; falling back to unit Gaussian")
            z_std_t = torch.ones(latent_dim, device=device)
        else:
            z_std_t = torch.tensor(z_std, device=device).expand(latent_dim)

        # Sample z ~ N(0, diag(z_std²)).  Match the checkpoint's
        # observed per-dim scale rather than hitting the prior cold.
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(n, latent_dim, generator=g, device=device) * z_std_t

        # Decode via the standard nucleus sampler used throughout viz.
        token_ids_list = decode_z(z, model_data, self.config)

        sp = spm_lib.SentencePieceProcessor(model_file=self.config.spm_model)
        specials = {
            tid
            for tid in [sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()]
            if tid >= 0
        }
        texts: list[str] = []
        for toks in token_ids_list:
            toks = [t for t in toks if t not in specials and t < sp.vocab_size()]
            texts.append(sp.decode(toks).strip())

        # Stats ---------------------------------------------------------
        cats = [_classify(t) for t in texts]
        counts = {"matched": 0, "mismatched": 0, "unmatched": 0}
        tag_types: dict[str, int] = {}
        lengths: list[int] = []
        for (cat, tag), text in zip(cats, texts):
            counts[cat] += 1
            if tag:
                tag_types[tag] = tag_types.get(tag, 0) + 1
            lengths.append(len(text))
        uniq = len(set(texts))

        # Core numerical analysis — printed directly to stdout so the
        # figures are a visual aid, not the only way to read results.
        lens_arr = np.asarray(lengths, dtype=np.int32)
        pct_bounds = (0, 25, 50, 75, 90, 99, 100)
        pct_vals = np.percentile(lens_arr, pct_bounds)
        print(f"\n==== random-z decode quality (N={n}) ====")
        print(f"tag pair outcome      count     pct")
        print(f"  matched             {counts['matched']:>5}   {100*counts['matched']/n:>5.1f}%")
        print(f"  mismatched          {counts['mismatched']:>5}   {100*counts['mismatched']/n:>5.1f}%")
        print(f"  unmatched           {counts['unmatched']:>5}   {100*counts['unmatched']/n:>5.1f}%")
        print(f"unique outputs:       {uniq:>5}   {100*uniq/n:>5.1f}%")
        print()
        print(f"decoded length (chars): min={lens_arr.min()} max={lens_arr.max()} "
              f"mean={lens_arr.mean():.1f} std={lens_arr.std():.1f}")
        print("  percentile  length")
        for p, v in zip(pct_bounds, pct_vals):
            print(f"  p{p:<3}        {int(v)}")
        # Histogram buckets that mirror the saved figure so you can
        # scan the shape from the terminal.
        hist, bin_edges = np.histogram(lens_arr, bins=10)
        print("  length histogram (10 bins):")
        for i, c in enumerate(hist):
            lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
            bar = "#" * int(40 * c / max(hist))
            print(f"    [{lo:>4}-{hi:<4}]  {c:>4}  {bar}")
        print()
        print("outer-tag type frequency:")
        if tag_types:
            for tag, ct in sorted(tag_types.items(), key=lambda kv: -kv[1]):
                print(f"  <{tag}>  {ct:>5}   {100*ct/n:>5.1f}%")
        else:
            print("  (no recognizable opener)")
        print()
        print("examples (up to 3 per category):")
        for cat in ("matched", "mismatched", "unmatched"):
            ex = [t for (c, _), t in zip(cats, texts) if c == cat][:3]
            for t in ex:
                print(f"  [{cat}] {t}")
        print("==== end random-z ====\n")

        # Save a compact text report next to the figures.
        report = [
            f"random-z decode on {n} samples (seed={seed})",
            f"  matched    tags: {counts['matched']} ({100*counts['matched']/n:.1f}%)",
            f"  mismatched tags: {counts['mismatched']} ({100*counts['mismatched']/n:.1f}%)",
            f"  unmatched:       {counts['unmatched']} ({100*counts['unmatched']/n:.1f}%)",
            f"  unique outputs:  {uniq} ({100*uniq/n:.1f}%)",
            f"  length: min={min(lengths)} max={max(lengths)} "
            f"mean={float(np.mean(lengths)):.1f} p99={int(np.percentile(lengths, 99))}",
            "",
            "== 10 matched examples ==",
            *[t for (c, _), t in zip(cats, texts) if c == "matched"][:10],
            "",
            "== 10 mismatched examples ==",
            *[t for (c, _), t in zip(cats, texts) if c == "mismatched"][:10],
            "",
            "== 10 unmatched examples ==",
            *[t for (c, _), t in zip(cats, texts) if c == "unmatched"][:10],
        ]
        self._report_text = "\n".join(report)

        apply_style()
        # 1. Tag validity bar
        fig_tv, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        names = ["matched", "mismatched", "unmatched"]
        values = [counts[k] for k in names]
        colors = ["#4caf50", "#ff9800", "#f44336"]
        ax.bar(names, values, color=colors)
        ax.set_title(f"random-z tag validity (N={n})")
        ax.set_ylabel("count")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v}\n({100*v/n:.1f}%)", ha="center", va="bottom")
        ax.set_ylim(top=max(values) * 1.2)
        fig_tv.tight_layout()

        # 2. Length histogram
        fig_len, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.hist(lengths, bins=40, color="#2196f3", edgecolor="black")
        ax.set_title("decoded length (chars)")
        ax.set_xlabel("length")
        ax.set_ylabel("count")
        fig_len.tight_layout()

        # 3. Outer-tag distribution
        fig_tt, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        if tag_types:
            sorted_tags = sorted(tag_types.items(), key=lambda kv: -kv[1])
            tags = [t for t, _ in sorted_tags]
            cts = [c for _, c in sorted_tags]
            ax.bar(tags, cts, color="#673ab7")
            ax.set_title("outer-tag type frequency")
            ax.set_ylabel("count")
            plt.xticks(rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "no recognizable outer tags", ha="center", transform=ax.transAxes)
        fig_tt.tight_layout()

        return [fig_tv, fig_len, fig_tt]

    @staticmethod
    def _log_examples(cats, texts, k: int = 3) -> None:
        for cat in ("matched", "mismatched", "unmatched"):
            examples = [t for (c, _), t in zip(cats, texts) if c == cat][:k]
            for t in examples:
                logger.info("  [%s] %s", cat, t)

    def save(self, figures, suffixes=None):  # type: ignore[override]
        # Write the text report alongside the figures.
        from pathlib import Path
        paths = super().save(figures, suffixes)
        if getattr(self, "_report_text", None):
            txt = Path(self.config.output_dir) / f"{self.name}_report.txt"
            txt.write_text(self._report_text)
            paths.append(txt)
        return paths
