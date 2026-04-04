"""Multi-scale attention pattern visualization for the PhraseDecoder.

Captures attention weights from every decoder layer via forward hooks,
then generates:

1. Per-head attention heatmap grid (first layer, averaged across sentences).
2. Per-head attention heatmap grid (averaged across all layers and sentences).
3. Attention entropy per head per position (line plot).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.figure import Figure
from torch import Tensor, nn

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.style import FIGSIZE_GRID, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)


class AttentionVisualization(BaseVisualization):
    """Visualize multi-scale attention patterns from the PhraseDecoder."""

    name = "attention"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict[str, Any]) -> list[Figure]:
        """Generate attention visualizations from teacher-forced decoding.

        Args:
            data: Dict containing ``z``, ``modules``, ``device``, ``cfg``,
                ``rope_freqs``, ``cached_mask``, ``token_ids_list``,
                ``full_vocab``, ``bos_id``, ``eos_id``.

        Returns:
            Three figures: [head_heatmaps_layer0, head_heatmaps_all,
            entropy_lines].
        """
        apply_style()

        head_indices = self._parse_heads(data)
        n_heads = data["cfg"].decoder_num_heads
        head_windows = data["cfg"].attention_head_windows

        # Capture attention weights via teacher-forced decoding
        # all_attn shape: (n_sentences, n_layer_applications, H, S, S)
        all_attn = self._capture_attention(data)

        if len(all_attn) == 0:
            logger.warning("No attention weights captured; returning empty list")
            return []

        # Stack into (N, L, H, S, S)
        attn_tensor = torch.stack(all_attn, dim=0)

        fig1 = self._make_head_heatmaps(
            attn_tensor, head_indices, head_windows, n_heads,
            layer_idx=0,
            title_prefix="Layer 0",
        )
        fig2 = self._make_head_heatmaps(
            attn_tensor, head_indices, head_windows, n_heads,
            layer_idx=None,
            title_prefix="All Layers",
        )
        fig3 = self._make_entropy_plot(
            attn_tensor, head_indices, head_windows, n_heads,
        )

        return [fig1, fig2, fig3]

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures,
            suffixes=["head_heatmaps_layer0", "head_heatmaps_all", "entropy"],
        )

    # ------------------------------------------------------------------
    # Head selection
    # ------------------------------------------------------------------

    def _parse_heads(self, data: dict) -> list[int]:
        """Parse ``self.config.heads`` (comma-separated) into head indices.

        Returns all heads if the config string is empty.
        """
        n_heads = data["cfg"].decoder_num_heads
        heads_str = self.config.heads.strip()
        if not heads_str:
            return list(range(n_heads))
        indices = []
        for tok in heads_str.split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < n_heads:
                    indices.append(idx)
        return indices if indices else list(range(n_heads))

    # ------------------------------------------------------------------
    # Attention capture via forward hooks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _capture_attention(self, data: dict) -> list[Tensor]:
        """Run teacher-forced decoding and capture attention weights.

        For each sentence, hooks on every ``PhraseDecoderLayer``
        recompute the self-attention weights (pre-dropout) from the
        layer's QKV projection weights and the input hidden states.

        Returns:
            List of tensors, each of shape ``(L, H, S, S)`` where
            ``L`` is the number of layer applications.
        """
        from lfm.generator.layers import PhraseDecoder, apply_rope

        device = data["device"]
        modules = data["modules"]
        cfg = data["cfg"]
        rope_freqs = data.get("rope_freqs")
        cached_mask = data.get("cached_mask")
        bos_id = data["bos_id"]
        eos_id = data["eos_id"]
        token_ids_list = data["token_ids_list"]

        latent_to_decoder = modules["latent_to_decoder"]
        dec_tok = modules["dec_token_embedding"]
        decoder: PhraseDecoder = modules["decoder"]
        output_head = modules["output_head"]

        for m in modules.values():
            if isinstance(m, nn.Module):
                m.eval()

        z = data["z"]
        n_sentences = min(self.config.n_sentences, z.size(0))
        z_subset = z[:n_sentences].to(device)

        all_sentence_attentions: list[Tensor] = []

        for i in range(n_sentences):
            z_i = z_subset[i : i + 1]  # (1, latent_dim)

            # Build teacher-forced input: [BOS] + token_ids (truncated)
            raw_ids = token_ids_list[i]
            max_len = cfg.max_seq_len - 1  # room for BOS
            ids = raw_ids[:max_len]
            input_ids = [bos_id] + ids
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=device
            )  # (1, S)

            seq_len = input_tensor.size(1)
            _n_mem = latent_to_decoder.out_features // dec_tok.embedding_dim
            memory = latent_to_decoder(z_i).reshape(1, _n_mem, -1)  # (1, K, D)
            tgt = dec_tok(input_tensor)  # (1, S, D)

            # Prepare mask
            tgt_mask = None
            if cached_mask is not None:
                tgt_mask = cached_mask[:, :seq_len, :seq_len]

            # Register hooks on each unique decoder layer
            captured_attentions: list[Tensor] = []

            def _make_hook(
                _rope_freqs: Tensor | None,
                _tgt_mask: Tensor | None,
                _seq_len: int,
            ):
                """Create a hook closure that captures attention weights."""

                def _hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
                    tgt_in = inputs[0]  # (B, S, D) input to this layer
                    x = module.norm1(tgt_in)
                    b, s, d = x.shape
                    qkv = module.self_attn_qkv(x).reshape(
                        b, s, 3, module.nhead, module.head_dim
                    )
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
                    q, k = qkv[0], qkv[1]

                    if _rope_freqs is not None:
                        q = apply_rope(q, _rope_freqs[:s])
                        k = apply_rope(k, _rope_freqs[:s])

                    scale = math.sqrt(module.head_dim)
                    attn = torch.matmul(q, k.transpose(-2, -1)) / scale

                    if _tgt_mask is not None:
                        mask = _tgt_mask[:, :s, :s]
                        if mask.dim() == 2:
                            attn = attn + mask.unsqueeze(0).unsqueeze(0)
                        elif mask.dim() == 3:
                            attn = attn + mask.unsqueeze(0)

                    attn = F.softmax(attn, dim=-1)
                    captured_attentions.append(attn.detach().cpu())

                return _hook

            # Attach hooks to unique layers (they may be applied multiple
            # times due to weight sharing, so we hook each unique layer)
            handles = []
            for layer in decoder.layers:
                h = layer.register_forward_hook(
                    _make_hook(rope_freqs, tgt_mask, seq_len)
                )
                handles.append(h)

            # Forward pass triggers hooks once per layer application
            _ = decoder(tgt, memory, tgt_mask=tgt_mask, rope_freqs=rope_freqs)

            # Remove hooks
            for h in handles:
                h.remove()

            # captured_attentions has one entry per layer *application*
            # (may be > len(decoder.layers) if weight sharing is on).
            # Each entry is (1, H, S, S) — squeeze batch dim.
            layer_attns = torch.stack(
                [a.squeeze(0) for a in captured_attentions], dim=0
            )  # (L, H, S, S)

            # Pad to max_seq_len so all sentences have the same tensor size
            max_s = cfg.max_seq_len
            if layer_attns.size(-1) < max_s:
                pad_s = max_s - layer_attns.size(-1)
                layer_attns = F.pad(layer_attns, (0, pad_s, 0, pad_s), value=0.0)

            all_sentence_attentions.append(layer_attns)

        return all_sentence_attentions

    # ------------------------------------------------------------------
    # Figure 1 & 2: Per-head attention heatmap grids
    # ------------------------------------------------------------------

    def _make_head_heatmaps(
        self,
        attn_tensor: Tensor,
        head_indices: list[int],
        head_windows: tuple[int, ...],
        n_heads: int,
        layer_idx: int | None,
        title_prefix: str,
    ) -> Figure:
        """Create a 2x4 grid of per-head average attention heatmaps.

        Args:
            attn_tensor: Shape ``(N, L, H, S, S)``.
            head_indices: Which heads to plot.
            head_windows: Window size per head from config.
            n_heads: Total number of heads.
            layer_idx: If ``int``, plot only that layer; if ``None``,
                average across all layers.
            title_prefix: Prefix for the suptitle.

        Returns:
            Matplotlib Figure.
        """
        if layer_idx is not None:
            # Average across sentences for one layer: (H, S, S)
            avg_attn = attn_tensor[:, layer_idx, :, :, :].mean(dim=0).numpy()
        else:
            # Average across sentences and layers: (H, S, S)
            avg_attn = attn_tensor.mean(dim=(0, 1)).numpy()

        n_plot = len(head_indices)
        ncols = min(4, n_plot)
        nrows = math.ceil(n_plot / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=FIGSIZE_GRID, squeeze=False
        )

        for plot_i, h in enumerate(head_indices):
            row, col = divmod(plot_i, ncols)
            ax = axes[row][col]

            attn_h = avg_attn[h]
            # Find the effective sequence length (non-zero region)
            effective_len = self._effective_length(attn_h)
            attn_crop = attn_h[:effective_len, :effective_len]

            window = head_windows[h] if h < len(head_windows) else 0
            window_label = "full" if window == 0 else str(window)

            im = ax.imshow(
                attn_crop,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                vmin=0.0,
                vmax=min(1.0, attn_crop.max() * 1.2) if attn_crop.max() > 0 else 1.0,
            )
            ax.set_title(f"Head {h} (w={window_label})")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for plot_i in range(n_plot, nrows * ncols):
            row, col = divmod(plot_i, ncols)
            axes[row][col].set_visible(False)

        scope = f"{title_prefix}" if layer_idx is not None else "All Layers"
        fig.suptitle(
            f"Average Attention Patterns — {scope} (n={int(attn_tensor.size(0))})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 3: Attention entropy per head per position
    # ------------------------------------------------------------------

    def _make_entropy_plot(
        self,
        attn_tensor: Tensor,
        head_indices: list[int],
        head_windows: tuple[int, ...],
        n_heads: int,
    ) -> Figure:
        """Create a line plot of attention entropy per position per head.

        Entropy is computed per query position as
        ``-sum(p * log(p))`` over the key dimension, then averaged
        across sentences and layers.

        Args:
            attn_tensor: Shape ``(N, L, H, S, S)``.
            head_indices: Which heads to plot.
            head_windows: Window size per head from config.
            n_heads: Total number of heads.

        Returns:
            Matplotlib Figure.
        """
        # Average across sentences and layers: (H, S, S)
        avg_attn = attn_tensor.mean(dim=(0, 1))  # still a Tensor for log

        # Compute entropy: for each query position, -sum(p * log(p)) over keys
        # Clamp to avoid log(0)
        eps = 1e-10
        p = avg_attn.clamp(min=eps)
        entropy = -(p * p.log()).sum(dim=-1)  # (H, S)
        entropy = entropy.numpy()

        # Find effective length (from the first head, they share the same padding)
        effective_len = self._effective_length(avg_attn[0].numpy())

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        cmap = plt.get_cmap("tab10")
        positions = np.arange(effective_len)

        for plot_i, h in enumerate(head_indices):
            window = head_windows[h] if h < len(head_windows) else 0
            window_label = "full" if window == 0 else str(window)
            color = cmap(plot_i % 10)
            ax.plot(
                positions,
                entropy[h, :effective_len],
                label=f"Head {h} (w={window_label})",
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )

        ax.set_xlabel("Position in sequence")
        ax.set_ylabel("Attention entropy (nats)")
        ax.set_title(
            "Attention Entropy per Head per Position "
            f"(avg over {int(attn_tensor.size(0))} sentences, "
            f"{int(attn_tensor.size(1))} layers)"
        )
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        ax.set_xlim(0, effective_len - 1)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_length(attn_matrix: np.ndarray) -> int:
        """Find the effective (non-padded) sequence length from an attention matrix.

        Padded rows/cols are all zeros. Returns the index of the last
        non-zero row + 1, clamped to at least 1.
        """
        row_sums = attn_matrix.sum(axis=-1)
        nonzero = np.nonzero(row_sums)[0]
        if len(nonzero) == 0:
            return max(1, attn_matrix.shape[0])
        return int(nonzero[-1]) + 1
