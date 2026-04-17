"""Attractor-basin diagnostics for the VAE latent space.

Quantifies geometry properties that raw CE/PPL miss:

* **basin radius** — smallest σ at which a random direction causes the
  decoded text to differ from the anchor's decode
* **tag-stability radius** — same criterion, but only requiring the
  opening tag to stay the same
* **degeneration radius** — smallest σ at which the decode becomes
  pathological (repetition cycles, unclosed tags)
* **transition width** — along A→B interpolation, the α-width of the
  region where the decode is neither A's text nor B's text
* **attractor count** — distinct decoded texts drawn from the prior

Figures
-------
``basin_density``  Per-anchor curves of fraction-exact-match vs σ.
``stability``      Tag-stability and degeneration curves vs σ.
``radii``          Per-anchor bar chart of r_basin / r_tag / r_deg.
``transitions``    Per-pair transition widths and midpoints.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm_lib
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn

from lfm.generator.layers import PhraseDecoder
from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)


def load_via_build_model(
    checkpoint_path: str,
    config_yaml_path: str,
    device: torch.device,
    spm_model_path: str | None = None,
) -> dict:
    """Alternative loader that mirrors ``scripts/decode_checkpoint.py``.

    Builds modules via the canonical ``build_model`` factory and loads
    state from a training resume checkpoint.  Returns a dict shaped like
    ``lfm.visualize.loader.load_checkpoint`` so downstream code is
    interchangeable.
    """
    import sentencepiece as _spm
    import yaml as _yaml

    from lfm.generator.pretrain.config import VAEPretrainConfig
    from lfm.generator.pretrain.model import build_model

    cfg_dict = _yaml.unsafe_load(open(config_yaml_path).read())
    ckpt_peek = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "max_seq_len" in ckpt_peek:
        cfg_dict["max_seq_len"] = int(ckpt_peek["max_seq_len"])
    cfg = VAEPretrainConfig(**cfg_dict)

    spm_file = spm_model_path or cfg_dict.get("spm_model_path") or cfg_dict.get("spm_path")
    sp = _spm.SentencePieceProcessor(model_file=spm_file)
    vocab_size = sp.vocab_size()
    full_vocab = vocab_size + 2

    modules = build_model(cfg, cfg.decoder_hidden_dim, full_vocab, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = ckpt.get("modules", ckpt)
    for name, mod in modules.items():
        if name.startswith("_"):
            continue
        state = module_states.get(name)
        if state is None:
            continue
        mod.load_state_dict(state)
        if isinstance(mod, nn.Module):
            mod.eval()

    return {
        "modules": modules,
        "rope_freqs": modules.get("_rope_freqs"),
        "cached_mask": modules.get("_cached_mask"),
        "cfg": cfg,
        "device": device,
        "full_vocab": full_vocab,
        "bos_id": vocab_size,
        "eos_id": vocab_size + 1,
        "z_mean": ckpt.get("z_mean"),
        "z_std": ckpt.get("z_std"),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }


# Default anchor prompts — mirror scripts/decode_checkpoint.py so results
# are directly comparable across the ad-hoc diagnostic and this viz.
DEFAULT_ANCHORS: tuple[str, ...] = (
    "<NP> seeds grapes peaches barrels or bottles </NP>",
    "<NP> the convergence on small angular scales </NP>",
    "<S> she entered a popular restaurant </S>",
    "<VP> jumped over the lazy dog </VP>",
    "<PP> in the atlantic </PP>",
)

_OPEN_TAG_RE = re.compile(r"^\s*<(\w+)>")
_CLOSE_TAG_RE = re.compile(r"</(\w+)>\s*$")


def _open_tag(text: str) -> str | None:
    m = _OPEN_TAG_RE.match(text)
    return m.group(1) if m else None


def _close_tag(text: str) -> str | None:
    m = _CLOSE_TAG_RE.search(text)
    return m.group(1) if m else None


def _is_degenerate(text: str) -> bool:
    """Heuristic: repetition-dominated OR unbalanced tags OR empty."""
    if not text.strip():
        return True
    words = text.split()
    if len(words) >= 6:
        # If the top-1 token occupies >= 35 % of content words, call it a
        # repetition cascade.
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) >= 0.35:
            return True
    # Multiple opening tags without matching closes == cascade
    n_open = len(re.findall(r"<\w+>", text))
    n_close = len(re.findall(r"</\w+>", text))
    if n_open > n_close + 1:
        return True
    # Open/close tag mismatch
    ot, ct = _open_tag(text), _close_tag(text)
    if ot and ct and ot != ct:
        return True
    return False


@torch.no_grad()
def _greedy_decode(
    z: Tensor,
    *,
    modules: dict[str, nn.Module],
    cfg,
    rope_freqs,
    cached_mask,
    bos_id: int,
    eos_id: int,
    specials: set[int],
    vocab_size: int,
    device: torch.device,
    sp: spm_lib.SentencePieceProcessor,
    max_len: int | None = None,
) -> list[str]:
    """Greedy argmax decode — deterministic given z, no sampling noise."""
    latent_to_decoder = modules["latent_to_decoder"]
    dec_tok = modules["dec_token_embedding"]
    decoder = modules["decoder"]
    output_head = modules["output_head"]

    n = z.size(0)
    n_mem = getattr(cfg, "num_memory_tokens", 1)
    mem = latent_to_decoder(z).reshape(n, n_mem, -1)
    ids = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    max_len = max_len or cfg.max_seq_len
    finished = torch.zeros(n, dtype=torch.bool, device=device)

    is_phrase = isinstance(decoder, PhraseDecoder) and cached_mask is not None

    for _ in range(max_len - 1):
        if is_phrase:
            tgt = dec_tok(ids)
            cm = cached_mask[:, : ids.size(1), : ids.size(1)]
            out = decoder(tgt, mem, tgt_mask=cm, rope_freqs=rope_freqs)
        else:
            pos = torch.arange(ids.size(1), device=device).unsqueeze(0)
            tgt = dec_tok(ids) + modules["dec_pos_embedding"](pos)
            cm = nn.Transformer.generate_square_subsequent_mask(
                ids.size(1), device=device,
            )
            out = decoder(tgt=tgt, memory=mem, tgt_mask=cm)

        logits = output_head(out[:, -1])
        for tid in specials:
            logits[:, tid] = float("-inf")
        logits[:, bos_id] = float("-inf")
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        finished |= nxt.squeeze(-1).eq(eos_id)
        if finished.all():
            break

    filt = specials | {bos_id, eos_id}
    texts: list[str] = []
    for j in range(n):
        toks = ids[j, 1:].cpu().tolist()
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        toks = [x for x in toks if x < vocab_size and x not in filt]
        texts.append(sp.decode(toks).strip())
    return texts


@torch.no_grad()
def _encode_prompts(
    prompts: list[str],
    *,
    modules: dict[str, nn.Module],
    cfg,
    sp: spm_lib.SentencePieceProcessor,
    specials: set[int],
    device: torch.device,
) -> Tensor:
    """Tokenize prompts and run the VAE encoder to get posterior means."""
    from lfm.generator.pretrain.diagnostics import encode_text

    batch: list[list[int]] = []
    for p in prompts:
        ids = sp.encode(p, out_type=int)
        ids = [x for x in ids if x not in specials]
        batch.append(ids)
    lengths = torch.tensor([len(b) for b in batch], device=device)
    max_len = int(lengths.max().item())
    src = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    for i, b in enumerate(batch):
        src[i, : len(b)] = torch.tensor(b, device=device)
    return encode_text(src, lengths, modules=modules, cfg=cfg, device=device)


class BasinAnalysisVisualization(BaseVisualization):
    name = "basin_analysis"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    def generate(self, data: dict) -> list[Figure]:  # noqa: C901
        model_data = data["model_data"]
        anchors: list[str] = list(data.get("anchors", DEFAULT_ANCHORS))
        sigmas: list[float] = list(
            data.get("sigmas", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50])
        )
        n_directions: int = int(data.get("n_directions", 32))
        alpha_resolution: int = int(data.get("alpha_resolution", 21))
        n_prior: int = int(data.get("n_prior", 512))
        seed: int = int(data.get("seed", 42))

        device = model_data["device"]
        modules = model_data["modules"]
        cfg = model_data["cfg"]
        rope_freqs = model_data.get("rope_freqs")
        cached_mask = model_data.get("cached_mask")
        bos_id = model_data["bos_id"]
        eos_id = model_data["eos_id"]
        vocab_size = bos_id  # bos_id == vocab_size by construction in loader

        for m in modules.values():
            if isinstance(m, nn.Module):
                m.eval()

        sp = spm_lib.SentencePieceProcessor(model_file=self.config.spm_model)
        specials = {
            tid
            for tid in [sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()]
            if tid >= 0
        }

        decode_kw = dict(
            modules=modules, cfg=cfg, rope_freqs=rope_freqs,
            cached_mask=cached_mask, bos_id=bos_id, eos_id=eos_id,
            specials=specials, vocab_size=vocab_size, device=device, sp=sp,
        )

        # -- 1. Encode anchors + take their greedy decode as reference text
        z_anchors = _encode_prompts(
            anchors, modules=modules, cfg=cfg, sp=sp,
            specials=specials, device=device,
        )  # (A, D)
        anchor_texts = _greedy_decode(z_anchors, **decode_kw)

        print("\n==== basin analysis ====")
        print(f"checkpoint: {self.config.checkpoint}")
        print(f"n_anchors={len(anchors)}  n_directions={n_directions}  "
              f"sigma_grid={sigmas}  alpha_res={alpha_resolution}  "
              f"n_prior={n_prior}  seed={seed}")
        print()
        print("anchors — original vs greedy decode:")
        for p, d in zip(anchors, anchor_texts):
            match = "✓" if p.strip() == d.strip() else "≠"
            print(f"  {match}  orig: {p}")
            print(f"     dec : {d}")
        print()

        # -- 2. Basin density: per-anchor fraction matching anchor text
        gen = torch.Generator(device=device).manual_seed(seed)
        n_anchors, latent_dim = z_anchors.shape
        noise = torch.randn(
            n_directions, latent_dim, device=device, generator=gen,
        )  # shared directions across anchors for comparability

        # density[a, s] = fraction matching anchor_texts[a]
        density_exact = np.zeros((n_anchors, len(sigmas)))
        density_tag = np.zeros((n_anchors, len(sigmas)))
        density_degen = np.zeros((n_anchors, len(sigmas)))

        for a in range(n_anchors):
            z_a = z_anchors[a]  # (D,)
            tag_a = _open_tag(anchor_texts[a])
            for si, sigma in enumerate(sigmas):
                if sigma == 0.0:
                    perturbed = z_a.unsqueeze(0).expand(n_directions, -1)
                else:
                    perturbed = z_a.unsqueeze(0) + sigma * noise
                texts = _greedy_decode(perturbed, **decode_kw)
                exact = sum(1 for t in texts if t == anchor_texts[a])
                tag_same = sum(
                    1 for t in texts if _open_tag(t) == tag_a and tag_a
                )
                degen = sum(1 for t in texts if _is_degenerate(t))
                density_exact[a, si] = exact / n_directions
                density_tag[a, si] = tag_same / n_directions
                density_degen[a, si] = degen / n_directions

        print("per-anchor basin density (fraction of K random directions "
              f"that keep the output identical / tag-identical / degenerate):")
        print(f"{'anchor':<42}" + "  ".join(f"σ={s:<4.2f}" for s in sigmas))
        for a in range(n_anchors):
            label = (anchor_texts[a][:40] + "…") if len(anchor_texts[a]) > 40 else anchor_texts[a]
            print(f"  exact:  {label:<40}" + "  ".join(f"{density_exact[a, si]:>5.2f}" for si in range(len(sigmas))))
            print(f"  tag:    {label:<40}" + "  ".join(f"{density_tag[a, si]:>5.2f}" for si in range(len(sigmas))))
            print(f"  degen:  {label:<40}" + "  ".join(f"{density_degen[a, si]:>5.2f}" for si in range(len(sigmas))))
            print()

        # -- 3. Radii (first σ where metric crosses 0.5)
        def _first_cross(curve: np.ndarray, up: bool) -> float:
            """Return first σ where curve crosses 0.5 (linear interp)."""
            for i in range(len(sigmas) - 1):
                a, b = curve[i], curve[i + 1]
                if up and a < 0.5 <= b:
                    frac = (0.5 - a) / (b - a) if b != a else 0.0
                    return sigmas[i] + frac * (sigmas[i + 1] - sigmas[i])
                if not up and a >= 0.5 > b:
                    frac = (a - 0.5) / (a - b) if a != b else 0.0
                    return sigmas[i] + frac * (sigmas[i + 1] - sigmas[i])
            return sigmas[-1] if (up and curve[-1] < 0.5) or (not up and curve[-1] >= 0.5) else sigmas[-1]

        r_basin = np.array([_first_cross(density_exact[a], up=False) for a in range(n_anchors)])
        r_tag = np.array([_first_cross(density_tag[a], up=False) for a in range(n_anchors)])
        r_deg = np.array([_first_cross(density_degen[a], up=True) for a in range(n_anchors)])

        print("half-basin radii (σ at which the curve crosses 0.5):")
        print(f"{'anchor':<42}  r_basin  r_tag   r_deg")
        for a in range(n_anchors):
            label = (anchor_texts[a][:40] + "…") if len(anchor_texts[a]) > 40 else anchor_texts[a]
            print(f"  {label:<42}  {r_basin[a]:>6.3f}  {r_tag[a]:>6.3f}  {r_deg[a]:>6.3f}")
        print(f"  {'mean':<42}  {r_basin.mean():>6.3f}  {r_tag.mean():>6.3f}  {r_deg.mean():>6.3f}")
        print()

        # -- 4. Transition width for each consecutive anchor pair
        alphas = np.linspace(0.0, 1.0, alpha_resolution)
        pairs = [(i, (i + 1) % n_anchors) for i in range(n_anchors)]
        transition_rows: list[tuple[int, int, float, float, float, list[str]]] = []
        print("interpolation transition widths:")
        print(f"  {'pair':<10}  α_exit  α_enter  width  α=0.5 decode")
        for a, b in pairs:
            z_interp = torch.stack(
                [(1 - t) * z_anchors[a] + t * z_anchors[b] for t in alphas],
                dim=0,
            )
            texts = _greedy_decode(z_interp, **decode_kw)
            # α_exit: first α where output != anchor_texts[a]
            alpha_exit = 1.0
            for i, t in enumerate(texts):
                if t != anchor_texts[a]:
                    alpha_exit = alphas[i]
                    break
            # α_enter: first α where output == anchor_texts[b]
            alpha_enter = 1.0
            for i, t in enumerate(texts):
                if t == anchor_texts[b]:
                    alpha_enter = alphas[i]
                    break
            width = max(0.0, alpha_enter - alpha_exit)
            mid_text = texts[len(texts) // 2]
            transition_rows.append((a, b, alpha_exit, alpha_enter, width, texts))
            print(f"  {a}→{b:<8}  {alpha_exit:>5.2f}   {alpha_enter:>5.2f}    {width:>5.2f}  {mid_text[:80]}")
        print()

        # Echo a few sample trajectories for visual inspection
        print("full trajectories (a→b, α-indexed):")
        for a, b, _, _, _, texts in transition_rows[:2]:  # first 2 pairs
            print(f"  pair {a}→{b}")
            for i, t in enumerate(texts):
                print(f"    α={alphas[i]:.2f}  {t}")
            print()

        # -- 5. Attractor count on prior samples
        z_std_value = 0.4
        if model_data.get("z_std") is not None:
            z_std_t = model_data["z_std"]
            if isinstance(z_std_t, torch.Tensor):
                z_std_value = float(z_std_t.mean().item())
            else:
                z_std_value = float(z_std_t)
        gen_prior = torch.Generator(device=device).manual_seed(seed + 1)
        z_prior = torch.randn(
            n_prior, latent_dim, device=device, generator=gen_prior,
        ) * z_std_value

        # Decode in batches of 128 to keep VRAM bounded
        prior_texts: list[str] = []
        for start in range(0, n_prior, 128):
            chunk = z_prior[start : start + 128]
            prior_texts.extend(_greedy_decode(chunk, **decode_kw))

        unique_texts = set(prior_texts)
        n_unique = len(unique_texts)
        n_degen_prior = sum(1 for t in prior_texts if _is_degenerate(t))
        tag_counts = Counter(_open_tag(t) or "∅" for t in prior_texts)
        print(f"prior-sample attractor analysis (N={n_prior}, "
              f"z_std={z_std_value:.3f}):")
        print(f"  unique decoded texts: {n_unique} ({100 * n_unique / n_prior:.1f}%)")
        print(f"  degenerate outputs:   {n_degen_prior} ({100 * n_degen_prior / n_prior:.1f}%)")
        print("  outer-tag distribution:")
        for tag, ct in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            print(f"    <{tag}>  {ct:>5}  ({100 * ct / n_prior:>5.1f}%)")
        print()

        # Store text report for save()
        self._report_text = self._build_report(
            anchors, anchor_texts, sigmas, density_exact, density_tag,
            density_degen, r_basin, r_tag, r_deg, transition_rows, alphas,
            n_prior, n_unique, n_degen_prior, tag_counts, z_std_value,
        )

        # ---------------- Figures ------------------
        apply_style()

        # Fig 1: basin density (exact-match) vs σ
        fig_density, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for a in range(n_anchors):
            label = anchor_texts[a][:30]
            ax.plot(sigmas, density_exact[a], marker="o", label=label)
        ax.axhline(0.5, color="black", linestyle=":", alpha=0.4, linewidth=1)
        ax.set_xlabel("σ (perturbation std)")
        ax.set_ylabel("fraction decoding == anchor")
        ax.set_title(f"basin density (K={n_directions} directions/σ)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")

        # Fig 2: tag-stability & degeneration curves
        fig_stab, (ax_t, ax_d) = plt.subplots(
            1, 2, figsize=(FIGSIZE_WIDE[0], FIGSIZE_SINGLE[1])
        )
        for a in range(n_anchors):
            label = anchor_texts[a][:25]
            ax_t.plot(sigmas, density_tag[a], marker="o", label=label)
            ax_d.plot(sigmas, density_degen[a], marker="o", label=label)
        for ax, title, ylabel in [
            (ax_t, "tag stability", "frac same opening tag"),
            (ax_d, "degeneration", "frac degenerate"),
        ]:
            ax.axhline(0.5, color="black", linestyle=":", alpha=0.4, linewidth=1)
            ax.set_xlabel("σ (perturbation std)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(-0.05, 1.05)
        ax_t.legend(fontsize=6, loc="upper right")

        # Fig 3: radii bar chart
        fig_r, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        idx = np.arange(n_anchors)
        w = 0.25
        ax.bar(idx - w, r_basin, width=w, label="r_basin", color="#4caf50")
        ax.bar(idx, r_tag, width=w, label="r_tag", color="#2196f3")
        ax.bar(idx + w, r_deg, width=w, label="r_deg", color="#f44336")
        ax.set_xticks(idx)
        ax.set_xticklabels(
            [anchor_texts[a][:18] + ("…" if len(anchor_texts[a]) > 18 else "") for a in range(n_anchors)],
            rotation=30, ha="right", fontsize=7,
        )
        ax.set_ylabel("σ")
        ax.set_title("half-basin radii per anchor")
        ax.legend(fontsize=8)

        # Fig 4: transitions
        fig_tr, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        idx = np.arange(len(transition_rows))
        exits = np.array([r[2] for r in transition_rows])
        enters = np.array([r[3] for r in transition_rows])
        ax.bar(idx, enters - exits, bottom=exits, color="#9c27b0", alpha=0.6, label="transition zone")
        ax.scatter(idx, exits, marker="v", color="#4caf50", label="α_exit", zorder=3)
        ax.scatter(idx, enters, marker="^", color="#f44336", label="α_enter", zorder=3)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{a}→{b}" for a, b, *_ in transition_rows], fontsize=8)
        ax.set_ylabel("α")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("A→B interpolation transition zones")
        ax.legend(fontsize=8)

        for fig in (fig_density, fig_stab, fig_r, fig_tr):
            fig.tight_layout()

        return [fig_density, fig_stab, fig_r, fig_tr]

    @staticmethod
    def _build_report(
        anchors, anchor_texts, sigmas, density_exact, density_tag,
        density_degen, r_basin, r_tag, r_deg, transition_rows, alphas,
        n_prior, n_unique, n_degen_prior, tag_counts, z_std_value,
    ) -> str:
        lines: list[str] = []
        lines.append(f"Basin analysis — {len(anchors)} anchors, σ grid: {sigmas}")
        lines.append("")
        lines.append("== Anchor reconstructions ==")
        for p, d in zip(anchors, anchor_texts):
            match = "MATCH" if p.strip() == d.strip() else "DIFF "
            lines.append(f"  [{match}] {p}")
            lines.append(f"          {d}")
        lines.append("")
        lines.append("== Half-basin radii ==")
        for a, (p, rb, rt, rd) in enumerate(zip(anchor_texts, r_basin, r_tag, r_deg)):
            lines.append(f"  anchor {a}: {p}")
            lines.append(f"    r_basin={rb:.3f}  r_tag={rt:.3f}  r_deg={rd:.3f}")
        lines.append(f"  mean: r_basin={r_basin.mean():.3f} "
                     f"r_tag={r_tag.mean():.3f} r_deg={r_deg.mean():.3f}")
        lines.append("")
        lines.append("== Transition widths ==")
        for a, b, ex, en, w, texts in transition_rows:
            lines.append(f"  {a}→{b}: α_exit={ex:.2f}  α_enter={en:.2f}  width={w:.2f}")
            lines.append(f"    A: {anchor_texts[a]}")
            lines.append(f"    B: {anchor_texts[b]}")
            for i, t in enumerate(texts):
                lines.append(f"    α={alphas[i]:.2f}  {t}")
            lines.append("")
        lines.append(f"== Prior attractor count (N={n_prior}, z_std={z_std_value:.3f}) ==")
        lines.append(f"  unique texts: {n_unique} ({100*n_unique/n_prior:.1f}%)")
        lines.append(f"  degenerate:   {n_degen_prior} ({100*n_degen_prior/n_prior:.1f}%)")
        lines.append("  outer-tag distribution:")
        for tag, ct in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    <{tag}>  {ct}  ({100*ct/n_prior:.1f}%)")
        return "\n".join(lines)

    def save(self, figures, suffixes=None):  # type: ignore[override]
        from pathlib import Path
        paths = super().save(
            figures,
            suffixes or ["density", "stability", "radii", "transitions"],
        )
        if getattr(self, "_report_text", None):
            txt = Path(self.config.output_dir) / f"{self.name}_report.txt"
            txt.write_text(self._report_text)
            paths.append(txt)
        return paths
