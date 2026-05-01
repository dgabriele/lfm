"""Game-agnostic training orchestrator for agent communication games."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lfm.embeddings.store import EmbeddingStore
from lfm.utils.oom import shrink_on_oom

logger = logging.getLogger(__name__)

def _respell_ipa(ipa_text: str) -> str | None:
    """Respell IPA as diacritical Latin for diagnostic logs."""
    try:
        from lfm.translator.romanize import respell
        return respell(ipa_text)
    except Exception:
        return None


class AgentTrainer:
    """Train an agent game with curriculum learning, logging, and checkpointing.

    The trainer is game-agnostic: any game module that implements
    ``forward(anchor, distractors) -> dict`` and ``trainable_param_groups()``
    can be plugged in.

    Args:
        game: A game module (e.g. ``ReferentialGame``).
        config: The game's config (must have training/curriculum/checkpoint fields).
    """

    def __init__(self, game: nn.Module, config) -> None:
        self.game = game
        self.config = config
        self.device = torch.device(config.device)
        self._batch_size = config.batch_size  # mutable for OOM recovery

        # Load embedding store
        self.store = EmbeddingStore(config.embedding_store_dir)
        self.store.load()
        logger.info(
            "Store: %d passages, dim=%d, %d clusters",
            self.store.num_passages, self.store.embedding_dim,
            self.store.num_clusters,
        )

        self._embeddings = self.store._embeddings
        self._cluster_labels = self.store._cluster_labels
        self._n = self._embeddings.shape[0]
        self._rng = np.random.default_rng(config.seed)

        # Optimizer
        param_groups = game.trainable_param_groups()
        self._all_params = [p for g in param_groups for p in g["params"]]
        self.optimizer = torch.optim.Adam(param_groups)

        total_params = sum(
            sum(p.numel() for p in g["params"]) for g in param_groups
        )
        logger.info("Total trainable params: %d", total_params)

        # Output directory
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Online hard-negative mining (ANCE-style). When enabled by the
        # game config, periodically re-encode all passages and cache each
        # passage's top-K nearest neighbors in the model's current alien
        # space. Sampling falls back to KMeans-cluster hard negatives until
        # the first refresh fires.
        self._hard_neg_indices: np.ndarray | None = None
        self._hard_neg_refresh_every = int(getattr(config, "hard_neg_refresh_every", 0))
        self._hard_neg_topk = int(getattr(config, "hard_neg_topk", 100))
        self._hard_neg_warmup = int(getattr(config, "hard_neg_warmup", 0))

        from lfm.agents.training_history import TrainingHistory
        self._history = TrainingHistory()

    def _sample_batch(
        self, step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
        """Sample anchor + distractors with curriculum difficulty.

        Vectorized: hard-negative sampling groups batch items by cluster and
        bulk-samples all distractors for each group at once, avoiding the
        per-item Python loop that starves the GPU between steps.
        """
        cfg = self.config
        curriculum = cfg.curriculum

        if curriculum.enabled:
            frac = min(step / max(curriculum.warmup_steps, 1), 1.0)
            hard_ratio = (
                curriculum.start_hard_ratio
                + frac * (curriculum.end_hard_ratio - curriculum.start_hard_ratio)
            )
        else:
            hard_ratio = 1.0

        n_hard = int(hard_ratio * cfg.num_distractors)
        medium_ratio = getattr(curriculum, "medium_ratio", 0.0) if curriculum.enabled else 0.0
        n_medium = int(medium_ratio * cfg.num_distractors)
        n_easy = cfg.num_distractors - n_hard - n_medium

        idx = self._rng.integers(0, self._n, size=self._batch_size)
        clusters = self._cluster_labels[idx]  # (B,)

        dist_indices = np.empty(
            (self._batch_size, cfg.num_distractors), dtype=np.intp,
        )

        # Easy distractors: fully vectorized
        if n_easy > 0:
            dist_indices[:, n_hard + n_medium:] = self._rng.integers(
                0, self._n, size=(self._batch_size, n_easy),
            )

        # Hard distractors: prefer online-mined index (model's current
        # confusion neighbors) when available; fall back to KMeans cluster
        # hard negatives otherwise.
        if n_hard > 0:
            if self._hard_neg_indices is not None:
                # Online-mined: each anchor's top-K nearest non-self in
                # alien-emb space. Sample n_hard from each anchor's top-K.
                hard_pool = self._hard_neg_indices[idx]   # (B, K)
                K = hard_pool.shape[1]
                rand_choice = self._rng.integers(
                    0, K, size=(self._batch_size, n_hard),
                )
                dist_indices[:, :n_hard] = np.take_along_axis(
                    hard_pool, rand_choice, axis=1,
                )
            else:
                # Static KMeans-cluster: same cluster — group by cluster,
                # bulk sample per group.
                unique_clusters, inv_idx = np.unique(clusters, return_inverse=True)
                cluster_arrays = self.store._cluster_arrays
                for ci, cid in enumerate(unique_clusters):
                    mask = inv_idx == ci
                    count = int(mask.sum())
                    members = cluster_arrays[cid]
                    total = count * n_hard
                    drawn = self._rng.choice(members, size=total, replace=len(members) < total)
                    dist_indices[mask, :n_hard] = drawn.reshape(count, n_hard)

        # Medium distractors: different cluster — one Python loop, but n_medium is
        # typically 0; kept for completeness.
        if n_medium > 0:
            eligible_ids = [cid for cid in self.store._cluster_index]
            cluster_arrays = self.store._cluster_arrays
            for i in range(self._batch_size):
                anchor_cluster = int(clusters[i])
                chosen_clusters = self._rng.choice(
                    [c for c in eligible_ids if c != anchor_cluster],
                    size=n_medium, replace=True,
                )
                for j, cid in enumerate(chosen_clusters):
                    members = cluster_arrays[cid]
                    dist_indices[i, n_hard + j] = self._rng.choice(members)

        anchor = torch.tensor(
            self._embeddings[idx], dtype=torch.float32,
        ).to(self.device)
        distractors = torch.tensor(
            self._embeddings[dist_indices], dtype=torch.float32,
        ).to(self.device)

        # Build candidate indices for IPA receiver (anchor + distractors)
        all_indices = np.concatenate(
            [idx.reshape(-1, 1), dist_indices], axis=1,
        )  # (batch, 1 + num_distractors)
        candidate_indices = torch.tensor(all_indices, dtype=torch.long)

        return anchor, distractors, hard_ratio, candidate_indices

    def _maybe_refresh_hard_negatives(self, step: int, force: bool = False) -> None:
        """Refresh the online hard-negative kNN index when due.

        Skips silently if mining is disabled, the warmup hasn't passed, or
        the game doesn't expose ``encode_for_hard_neg_mining``. Otherwise
        runs the model over the full embedding store (no_grad), pools into
        a per-passage signature, and rebuilds a top-K nearest-non-self
        index in alien-emb space using cosine similarity.

        ``force=True`` skips the step%refresh_every modulo check — used at
        startup-on-resume so the index exists before the first training
        step (otherwise resuming at e.g. step 5100 with refresh_every=500
        would wait until step 5500 to first build the index).
        """
        if self._hard_neg_refresh_every <= 0:
            return
        if step < self._hard_neg_warmup:
            return
        if not force and step % self._hard_neg_refresh_every != 0:
            return
        if not hasattr(self.game, "encode_for_hard_neg_mining"):
            return

        import time

        n = self.store.num_passages
        bs = max(8, self._batch_size * 4)   # bigger batches OK in pure inference
        logger.info(
            "Refreshing online hard-negative index at step %d (n=%d, batch=%d) ...",
            step, n, bs,
        )

        # 1) Encode all signatures. The game's signature is per-position
        #    L2-normalised + flattened, so dot products between rows equal
        #    the sum of per-position cosines = InfoNCE confusion score.
        t0 = time.time()
        sigs_list = []
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            anchor = torch.tensor(
                self._embeddings[idx], dtype=torch.float32,
            ).to(self.device)
            sig = self.game.encode_for_hard_neg_mining(anchor)   # (b, D_flat)
            sigs_list.append(sig)
        sigs = torch.cat(sigs_list, dim=0)                       # (n, D_flat)
        dt_enc = time.time() - t0

        # 2) GPU chunked top-K kNN on InfoNCE-geometry similarity.
        #    sim[i, j] = sigs[i] @ sigs[j].T (vectors are per-position
        #    L2-normed; this is the exact InfoNCE-summed-over-positions score).
        #    Self is masked with -inf so all top-K are non-self neighbors.
        t0 = time.time()
        K = self._hard_neg_topk
        topk_idx = torch.empty((n, K), dtype=torch.long, device=sigs.device)
        chunk = 1024
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            sim = sigs[i:end] @ sigs.t()                         # (chunk, n)
            row_idx = torch.arange(i, end, device=sigs.device)
            sim[row_idx - i, row_idx] = float("-inf")            # mask self
            _, top = sim.topk(K, dim=1)
            topk_idx[i:end] = top
        self._hard_neg_indices = topk_idx.cpu().numpy().astype(np.intp)
        dt_knn = time.time() - t0
        del sigs

        logger.info(
            "  encoded %d sigs in %.1fs; built kNN(top-%d) in %.1fs; index ready",
            n, dt_enc, self._hard_neg_topk, dt_knn,
        )

    def _save_checkpoint(
        self, step: int, accuracy: float, hard_ratio: float = 1.0,
    ) -> None:
        """Overwrite latest.pt with current state. No "best" tracking — at
        B=32 per-batch acc has stddev ~5-8%, so the max single-batch accuracy
        is just the luckiest batch, not a meaningful global best. If you want
        a snapshot of a particular run state, ``cp latest.pt my_label.pt``.
        """
        ckpt = self.game.checkpoint_state()
        ckpt["step"] = step
        ckpt["accuracy"] = accuracy
        ckpt["hard_ratio"] = hard_ratio
        ckpt["batch_size"] = self._batch_size  # persist actual (post-OOM) batch size

        torch.save(ckpt, str(self._output_dir / "latest.pt"))
        self._history.save(str(self._output_dir / "history.parquet"))
        logger.info(
            "Checkpoint step %d — acc=%.1f%%",
            step, accuracy * 100,
        )

    def train(self, resume: str | None = None) -> dict[str, float]:
        """Run the training loop.

        Args:
            resume: Optional path to a checkpoint to resume from.

        Returns:
            Dict with final metrics.
        """
        cfg = self.config
        game = self.game
        num_candidates = cfg.num_distractors + 1
        chance = 1.0 / num_candidates

        start_step = 0
        if resume is not None:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            game.load_checkpoint_state(ckpt)
            start_step = ckpt.get("step", 0)
            logger.info("Resumed from %s at step %d", resume, start_step)

        contrastive = getattr(cfg, "contrastive_scoring", False)
        if contrastive:
            num_candidates = self._batch_size
            chance = 1.0 / num_candidates
            logger.info(
                "Contrastive scoring: %d-way (batch-wide InfoNCE), chance=%.1f%%",
                num_candidates, chance * 100,
            )
        else:
            logger.info(
                "Receiver scoring: %d distractors, %d-way, chance=%.1f%%, "
                "curriculum=%s",
                cfg.num_distractors, num_candidates, chance * 100,
                cfg.curriculum.enabled,
            )
        if cfg.curriculum.enabled:
            logger.info(
                "  curriculum: %.0f%% → %.0f%% hard negatives over %d steps",
                cfg.curriculum.start_hard_ratio * 100,
                cfg.curriculum.end_hard_ratio * 100,
                cfg.curriculum.warmup_steps,
            )

        # Per-step log-line legend — printed right before the loop so the
        # log file is self-documenting for whoever's reading it later.
        for line in (
            "step log legend:",
            "  loss      = total weighted training loss",
            "  acc       = top-1 accuracy on the contrastive pool",
            "  nce       = per-sentence InfoNCE  (mean across K paragraphs)",
            "  nce_agg   = aggregate-of-K InfoNCE  (mean alien_emb vs source)",
            "  topo      = per-sentence  1 - cos(msg_alien, msg_source)",
            "  topo_agg  = aggregate     1 - cos(msg_alien_agg, msg_source)",
            "  nkl       = sum_n weight_n * NgramKL_n  (weighted total)",
            "  kl2/kl3/kl4 = raw KL per n-gram order against corpus reference",
            "  adj       = adjacency-diversity hinge on adjacent prob distributions",
            "  ttr       = type-token ratio over batch's valid generated tokens",
            "  coh       = cross-paragraph diversity hinge  (cos > target ⇒ penalty)",
            "  res       = residual loss  (sub-aggregate must improve by ≥ margin)",
            "  tok_rec   = token-recurrence hinge loss  (footprint cosine below target ⇒ penalty)",
            "  tok_sim   = raw within-doc footprint cosine similarity (diagnostic, no loss)",
            "  lex_coh   = within-doc / across-doc Jaccard ratio  (>1 = topical coherence at token level)",
            "  pos_div   = unique-first-tokens at last sentence pos / B  (<<1 = positional collapse)",
            "  K         = n_paragraphs (sentences per anchor)",
            "  ndist     = n distractors used (curriculum-ramped)",
            "  hard      = within-cluster hard-negative ratio (curriculum-ramped)",
            "  vram      = peak GPU memory in MB",
        ):
            logger.info(line)

        results: dict[str, float] = {}
        accum = getattr(cfg, "gradient_accumulation_steps", 1)

        # Build or load IPA cache if using IPA receiver
        use_ipa = getattr(cfg, "use_ipa_receiver", False)
        if use_ipa and hasattr(game, "build_ipa_cache"):
            from pathlib import Path as _Path
            cache_path = _Path(cfg.output_dir) / "ipa_cache.pt"
            if cache_path.exists():
                _cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                game._ipa_cache_tokens = _cache["tokens"]
                game._ipa_cache_masks = _cache["masks"]
                logger.info("Loaded IPA cache from %s (%d sequences)", cache_path, game._ipa_cache_tokens.size(0))
            else:
                all_embs = torch.tensor(self._embeddings, dtype=torch.float32).to(self.device)
                game.build_ipa_cache(all_embs, batch_size=256)
                del all_embs
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        ipa_refresh = getattr(cfg, "ipa_cache_refresh", 0)

        # Build the initial hard-negative index if we're already past warmup
        # (e.g. resuming from a saturated KMeans-cluster run). Force=True
        # so it fires regardless of whether start_step is a refresh-modulo
        # multiple — the index doesn't yet exist at startup.
        self._maybe_refresh_hard_negatives(start_step, force=True)

        for step in range(start_step, cfg.steps):
            # Refresh IPA cache periodically
            if use_ipa and ipa_refresh > 0 and step > 0 and step % ipa_refresh == 0:
                all_embs = torch.tensor(self._embeddings, dtype=torch.float32).to(self.device)
                game.build_ipa_cache(all_embs, batch_size=256)

            # Refresh the online hard-negative kNN index periodically
            if step > start_step and self._hard_neg_refresh_every > 0:
                self._maybe_refresh_hard_negatives(step)

            # Full step wrapped in OOM recovery.  On OOM, reduce batch
            # size by 10%, clear VRAM, and retry the step.
            try:
                self.optimizer.zero_grad()
                acc_sum = 0.0
                for micro in range(accum):
                    anchor, distractors, hard_ratio, cand_idx = self._sample_batch(step)
                    out = game(
                        anchor, distractors, step=step,
                        candidate_indices=cand_idx.to(self.device) if use_ipa else None,
                    )
                    loss = out["loss"] / accum
                    loss.backward()
                    acc_sum += out["accuracy"].item()
                nn.utils.clip_grad_norm_(self._all_params, cfg.max_grad_norm)
                self.optimizer.step()
            except RuntimeError as e:
                # Free Python references to any partial forward state so
                # the OOM handler can reclaim their buffers.
                if "out" in locals():
                    del out
                if "loss" in locals():
                    del loss
                self._batch_size = shrink_on_oom(
                    e, self._batch_size,
                    label=f"at step {step}", optimizer=self.optimizer,
                )
                continue
            # Release cached allocator blocks to prevent reserved VRAM
            # from growing unboundedly over long training runs.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Use accumulated accuracy for logging
            out["accuracy"] = torch.tensor(acc_sum / accum)
            step_acc = acc_sum / accum

            # Record all scalar metrics for post-hoc analysis
            record = {"step": step, "hard_ratio": hard_ratio}
            for k, v in out.items():
                if k.startswith("_"):
                    continue  # skip internal tensors (tokens, masks)
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    record[k] = v.item()
            if torch.cuda.is_available():
                record["vram_mb"] = torch.cuda.max_memory_allocated() // (1024 * 1024)
            self._history.record(**record)

            # Logging
            if step % cfg.log_every == 0:
                extra = ""
                if "num_phrases" in out:
                    n_phr = out['num_phrases'].item()
                    total_tok = out['msg_lengths'].item()
                    is_dialogue = "_dialogue_tokens" in out
                    if is_dialogue:
                        num_turns = len(out["_dialogue_tokens"])
                        extra += f"  phr/turn={n_phr:.1f}"
                        extra += f"  tok/turn={total_tok / num_turns:.0f}"
                        extra += f"  total_tok={total_tok:.0f}"
                    else:
                        extra += f"  phrases={n_phr:.1f}"
                        tok_per_phr = total_tok / max(n_phr, 1)
                        extra += f"  total_tok={total_tok:.0f}  tok/phr={tok_per_phr:.0f}"
                if "z_intra_sim" in out:
                    extra += f"  z_sim={out['z_intra_sim'].item():.3f}"
                if "halt_cost" in out:
                    extra += f"  halt={out['halt_cost'].item():.3f}"
                # Loss breakdown — printed when present in output dict.
                for _term, _label in (
                    ("reconstruction", "recon"),
                    ("phrase_length", "plen"),
                    ("hidden_nce", "hnce"),
                    ("surface_nce", "snce"),
                    ("info_nce", "nce"),
                    ("info_nce_agg", "nce_agg"),
                    ("topology", "topo"),
                    ("topology_agg", "topo_agg"),
                    ("z_div_loss", "div"),
                    ("bigram_kl", "bkl"),
                    ("ngram_kl", "nkl"),
                    ("ngram_kl_n2", "kl2"),
                    ("ngram_kl_n3", "kl3"),
                    ("ngram_kl_n4", "kl4"),
                    ("adj_div", "adj"),
                    ("adj_diversity", "adj"),
                    ("ttr", "ttr"),
                    ("coherence", "coh"),
                    ("residual", "res"),
                    ("token_recurrence", "tok_rec"),
                    ("tok_rec_sim", "tok_sim"),
                    ("lex_coh", "lex_coh"),
                    ("pos_div_last", "pos_div"),
                    ("n_paragraphs", "K"),
                    ("tok_conc", "tconc"),
                    ("uniq_tok", "uniq"),
                    ("dec_fluency", "dflu"),
                    ("z_prior", "zprior"),
                ):
                    if _term in out:
                        extra += f"  {_label}={out[_term].item():.3f}"
                if "n_distractors" in out:
                    extra += f"  ndist={int(out['n_distractors'].item())}"
                if "z_coverage" in out and out["z_coverage"].item() > 0:
                    extra += f"  zcov={out['z_coverage'].item():.2f}"
                if "hard_overlap" in out:
                    extra += f"  hovl={out['hard_overlap'].item():.3f}"
                if "surface_loss" in out and out["surface_loss"].item() != 0:
                    extra += f"  surf={out['surface_loss'].item():.3f}"
                if "ce_loss" in out and out["ce_loss"].item() != 0:
                    extra += f"  ce={out['ce_loss'].item():.3f}"
                if "vocab_entropy" in out and out["vocab_entropy"].item() != 0:
                    extra += f"  vent={out['vocab_entropy'].item():.2f}"
                if "llm_pressure" in out and out["llm_pressure"].item() != 0:
                    extra += f"  llm={out['llm_pressure'].item():.3f}"
                if "qwen_roundtrip" in out and out["qwen_roundtrip"].item() != 0:
                    extra += f"  rt_cos={out['qwen_roundtrip'].item():.4f}"
                if "topo_loss" in out and out["topo_loss"].item() != 0:
                    extra += f"  topo={out['topo_loss'].item():.4f}"
                if "hs_weight" in out:
                    extra += f"  hs={out['hs_weight'].item():.2f}"
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
                    extra += f"  vram={vram_mb}MB"
                    torch.cuda.reset_peak_memory_stats()
                # Flush VRAM trace to disk if the game has a monitor
                if hasattr(game, 'vram_monitor'):
                    from pathlib import Path as _P
                    game.vram_monitor.save(str(_P(cfg.output_dir) / "vram_trace.npz"))
                logger.info(
                    "step=%d  loss=%.3f  acc=%.1f%%  "
                    "%s  hard=%.0f%%",
                    step,
                    loss.item(),
                    out["accuracy"].item() * 100,
                    extra if extra else f"expr_len={out['msg_lengths'].item():.0f}",
                    hard_ratio * 100,
                )

                if "_tokens" in out and step % cfg.checkpoint_every == 0:
                    try:
                        # Resolve renderer: game.gen.render_surface or game.render_surface
                        _renderer = None
                        if hasattr(game, "gen") and hasattr(game.gen, "render_surface"):
                            _renderer = game.gen.render_surface
                        elif hasattr(game, "render_surface"):
                            _renderer = game.render_surface

                        if _renderer and "_dialogue_tokens" in out:
                            logger.info("  --- monologue (sample 0) ---")
                            for turn_i, (toks, mask) in enumerate(
                                zip(out["_dialogue_tokens"], out["_dialogue_masks"]),
                            ):
                                surface = _renderer(
                                    toks[0:1], mask=mask[0:1],
                                )[0]
                                eng = _respell_ipa(surface)
                                if eng:
                                    logger.info(
                                        "  [T%d] %s  (%d tok)\n        → %s",
                                        turn_i, surface, int(mask[0].sum().item()), eng,
                                    )
                                else:
                                    logger.info(
                                        "  [T%d] %s  (%d tok)",
                                        turn_i, surface, int(mask[0].sum().item()),
                                    )
                        elif _renderer and "_gen_mask" in out:
                            toks = out["_tokens"]
                            mask = out["_gen_mask"]
                            n = min(5, toks.size(0))
                            phrase_len = getattr(cfg, "max_tokens_per_phrase", 0)
                            K = toks.size(1) // phrase_len if phrase_len > 0 else 1
                            for j in range(n):
                                parts = []
                                for k in range(K):
                                    sl = slice(k * phrase_len, (k + 1) * phrase_len) if phrase_len > 0 else slice(None)
                                    pt = toks[j:j+1, sl]
                                    pm = mask[j:j+1, sl]
                                    surface = _renderer(pt, mask=pm)[0]
                                    eng = _respell_ipa(surface)
                                    if eng and eng.strip():
                                        parts.append(eng.strip().capitalize())
                                if parts:
                                    logger.info("  [%d] %s.", j, ". ".join(parts))
                    except Exception:
                        pass

            # Checkpoint
            if step > 0 and step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step, out["accuracy"].item(), hard_ratio)

        # Final checkpoint
        if cfg.steps > 0:
            with torch.no_grad():
                final_acc = out["accuracy"].item()
            self._save_checkpoint(cfg.steps, final_acc, hard_ratio)

            results = {
                "final_accuracy": final_acc,
                "final_loss": loss.item(),
                "chance": chance,
            }
            logger.info("Final: %s", results)
            self._history.save(str(self._output_dir / "history.parquet"))

        return results
