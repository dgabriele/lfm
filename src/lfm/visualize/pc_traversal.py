"""Principal-component traversal of the latent space.

Encodes the training corpus, runs PCA on the posterior means, and
sweeps each of the top-K principal components from −3σ to +3σ around
the centroid.  Decodes at every step so the user can read off what
each PC encodes — length? tag? topic?  If the model has disentangled
anything interpretable, traversals should reveal it.

Saves a figure showing the explained-variance curve (which PCs to
trust) and a text report with decoded samples per PC axis.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm_lib
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, apply_style

logger = logging.getLogger(__name__)


class PCTraversalVisualization(BaseVisualization):
    name = "pc_traversal"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    def generate(self, data: dict) -> list[Figure]:
        z_encoded = data["z"]  # (N, latent_dim) from encode_labeled_corpus
        model_data = data["model_data"]
        top_k = data.get("top_k", 8)
        n_steps = data.get("n_steps", 7)
        span_sigma = data.get("span_sigma", 3.0)

        z_np = z_encoded.detach().cpu().numpy().astype(np.float64)
        mean = z_np.mean(axis=0)
        logger.info("running PCA on %d encoded z's (dim=%d)", z_np.shape[0], z_np.shape[1])
        pca = PCA(n_components=min(top_k, z_np.shape[1]))
        pca.fit(z_np - mean)
        # σ along each PC = sqrt of the component's eigenvalue
        stds = np.sqrt(pca.explained_variance_)

        # Build a traversal batch: for each of top_k PCs, walk along the
        # component at n_steps points from −span to +span sigma.
        sweep = np.linspace(-span_sigma, span_sigma, n_steps)
        device = model_data["device"]
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)

        traversals = {}
        sp = spm_lib.SentencePieceProcessor(model_file=self.config.spm_model)
        specials = {
            tid
            for tid in [sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()]
            if tid >= 0
        }
        for k in range(pca.n_components_):
            direction = torch.tensor(pca.components_[k], dtype=torch.float32, device=device)
            scale = float(stds[k])
            z_batch = torch.stack(
                [mean_t + float(t) * scale * direction for t in sweep],
                dim=0,
            )
            tok_lists = decode_z(z_batch, model_data, self.config)
            texts = []
            for toks in tok_lists:
                toks = [t for t in toks if t not in specials and t < sp.vocab_size()]
                texts.append(sp.decode(toks).strip())
            traversals[k] = (scale, texts)

        # Core numerical analysis to stdout before the traversal text
        # dump — gives a scannable summary of the latent geometry
        # without needing to open the saved figure.
        ev_ratio = pca.explained_variance_ratio_
        cum_ev = np.cumsum(ev_ratio)
        # Participation ratio: (Σλ)² / Σ(λ²) — effective dimensionality.
        eigvals = pca.explained_variance_
        participation = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
        # Sorted list + thresholds for "dims to explain 50 / 90 / 99%".
        # Uses the full fit (top_k components); clamps if top_k is low.
        def _dims_to_reach(threshold: float) -> int:
            return int((cum_ev < threshold).sum()) + 1
        print(
            f"\n==== PC traversal ({z_np.shape[0]} encoded samples, "
            f"top_k={pca.n_components_}) ===="
        )
        print(f"participation ratio (effective dim): {participation:.2f}")
        print(f"dims to reach 50% variance: {_dims_to_reach(0.50)}")
        print(f"dims to reach 90% variance: {_dims_to_reach(0.90)}")
        print(f"dims to reach 99% variance: {_dims_to_reach(0.99)}")
        print()
        print("PC   σ          var%    cum%")
        for k in range(pca.n_components_):
            print(
                f"PC{k:<2}  {float(np.sqrt(eigvals[k])):>6.3f}   "
                f"{100*ev_ratio[k]:>5.2f}%  {100*cum_ev[k]:>5.2f}%"
            )
        print()
        print("traversal samples per PC:")

        # Log + report ---------------------------------------------------
        report_lines = [
            f"PC traversal on {z_np.shape[0]} encoded samples, top_k={pca.n_components_}, "
            f"steps={n_steps}, span=±{span_sigma:g}σ",
            f"participation ratio (effective dim): {participation:.2f}",
            f"dims to reach 50/90/99%: "
            f"{_dims_to_reach(0.50)}/{_dims_to_reach(0.90)}/{_dims_to_reach(0.99)}",
            "",
        ]
        for k, (scale, texts) in traversals.items():
            ev = pca.explained_variance_ratio_[k]
            header = f"PC{k}  (σ={scale:.3f}, explains {100*ev:.2f}% variance)"
            report_lines.append(header)
            print(header)
            for t, text in zip(sweep, texts):
                line = f"  t={t:+.2f}σ  {text}"
                report_lines.append(line)
                print(line)
            report_lines.append("")
            print()
        print("==== end PC traversal ====\n")

        self._report_text = "\n".join(report_lines)

        # Figure: explained variance of top components
        apply_style()
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ks = np.arange(1, pca.n_components_ + 1)
        ax.bar(ks, pca.explained_variance_ratio_ * 100, color="#3f51b5")
        ax.set_xlabel("principal component")
        ax.set_ylabel("explained variance (%)")
        ax.set_title(f"top {pca.n_components_} PC explained variance")
        fig.tight_layout()

        return [fig]

    def save(self, figures, suffixes=None):  # type: ignore[override]
        from pathlib import Path
        paths = super().save(figures, suffixes)
        if getattr(self, "_report_text", None):
            txt = Path(self.config.output_dir) / f"{self.name}_report.txt"
            txt.write_text(self._report_text)
            paths.append(txt)
        return paths
