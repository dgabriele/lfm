"""Game-agnostic training orchestrator for agent communication games."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lfm.embeddings.store import EmbeddingStore

logger = logging.getLogger(__name__)


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
        self._best_acc = 0.0

        from lfm.agents.training_history import TrainingHistory
        self._history = TrainingHistory()

    def _sample_batch(
        self, step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]:
        """Sample anchor + distractors with curriculum difficulty."""
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

        idx = self._rng.integers(0, self._n, size=cfg.batch_size)
        anchor = torch.tensor(
            self._embeddings[idx], dtype=torch.float32,
        ).to(self.device)

        dist_indices = np.empty(
            (cfg.batch_size, cfg.num_distractors), dtype=np.intp,
        )
        medium_ratio = getattr(curriculum, "medium_ratio", 0.0) if curriculum.enabled else 0.0

        for i in range(cfg.batch_size):
            n_hard = int(hard_ratio * cfg.num_distractors)
            n_medium = int(medium_ratio * cfg.num_distractors)
            n_easy = cfg.num_distractors - n_hard - n_medium

            cluster = int(self._cluster_labels[idx[i]])

            hard_idx = np.empty(0, dtype=np.intp)
            if n_hard > 0:
                hard_idx = self.store.sample_from_cluster(
                    cluster, n_hard, rng=self._rng,
                )

            medium_idx = np.empty(0, dtype=np.intp)
            if n_medium > 0:
                medium_idx = self.store.sample_from_different_cluster(
                    cluster, n_medium, rng=self._rng,
                )

            easy_idx = self._rng.integers(0, self._n, size=max(n_easy, 0))
            dist_indices[i] = np.concatenate([hard_idx, medium_idx, easy_idx])

        distractors = torch.tensor(
            self._embeddings[dist_indices], dtype=torch.float32,
        ).to(self.device)

        # Build candidate indices for IPA receiver (anchor + distractors)
        all_indices = np.concatenate(
            [idx.reshape(-1, 1), dist_indices], axis=1,
        )  # (batch, 1 + num_distractors)
        candidate_indices = torch.tensor(all_indices, dtype=torch.long)

        return anchor, distractors, hard_ratio, candidate_indices

    def _save_checkpoint(
        self, step: int, accuracy: float, hard_ratio: float = 1.0,
    ) -> None:
        """Save latest checkpoint, and best if peak accuracy improved.

        Tracks peak accuracy across ALL steps (not just checkpoint steps)
        via ``_peak_acc``.  Best checkpoint is only updated once the
        curriculum has reached full difficulty (hard_ratio >= 1.0).
        """
        ckpt = self.game.checkpoint_state()
        ckpt["step"] = step
        ckpt["accuracy"] = accuracy
        ckpt["hard_ratio"] = hard_ratio

        torch.save(ckpt, str(self._output_dir / "latest.pt"))
        self._history.save(str(self._output_dir / "history.parquet"))
        logger.info(
            "Checkpoint step %d — acc=%.1f%% (best=%.1f%%)",
            step, accuracy * 100, self._best_acc * 100,
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

        logger.info(
            "Backprop referential game: %d distractors, chance=%.1f%%, "
            "curriculum=%s",
            cfg.num_distractors, chance * 100, cfg.curriculum.enabled,
        )
        if cfg.curriculum.enabled:
            logger.info(
                "  curriculum: %.0f%% → %.0f%% hard negatives over %d steps",
                cfg.curriculum.start_hard_ratio * 100,
                cfg.curriculum.end_hard_ratio * 100,
                cfg.curriculum.warmup_steps,
            )

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

        for step in range(start_step, cfg.steps):
            # Refresh IPA cache periodically
            if use_ipa and ipa_refresh > 0 and step > 0 and step % ipa_refresh == 0:
                all_embs = torch.tensor(self._embeddings, dtype=torch.float32).to(self.device)
                game.build_ipa_cache(all_embs, batch_size=256)

            # Accumulate gradients over multiple micro-batches
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
            if hard_ratio >= 1.0 and step_acc > self._best_acc:
                self._best_acc = step_acc
                # Save best checkpoint immediately at the peak
                ckpt = game.checkpoint_state()
                ckpt["step"] = step
                ckpt["accuracy"] = step_acc
                ckpt["hard_ratio"] = hard_ratio
                torch.save(ckpt, str(self._output_dir / "best.pt"))
                logger.info(
                    "  New best acc=%.1f%% at step %d — saved best.pt",
                    step_acc * 100, step,
                )

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
                if "z_div_loss" in out and out["z_div_loss"].item() > 0:
                    extra += f"  div={out['z_div_loss'].item():.3f}"
                if "z_coverage" in out and out["z_coverage"].item() > 0:
                    extra += f"  zcov={out['z_coverage'].item():.2f}"
                if "turn_sim" in out and out["turn_sim"].item() != 0:
                    extra += f"  tsim={out['turn_sim'].item():.3f}"
                if "surface_loss" in out and out["surface_loss"].item() != 0:
                    extra += f"  surf={out['surface_loss'].item():.3f}"
                if "ce_loss" in out and out["ce_loss"].item() != 0:
                    extra += f"  ce={out['ce_loss'].item():.3f}"
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

                # Print IPA samples every checkpoint
                if "_tokens" in out and step % cfg.checkpoint_every == 0:
                    try:
                        sp = game.gen._tokenizer._sp if hasattr(game.gen, '_tokenizer') else None
                        if sp is None:
                            import sentencepiece as _spm
                            sp = _spm.SentencePieceProcessor(model_file=cfg.spm_path)
                        vocab_size = sp.vocab_size()
                        eos_id = vocab_size + 1

                        from lfm.translator.romanize import syllable_hyphenate

                        if "_dialogue_tokens" in out:
                            # Dialogue game: print one full monologue
                            logger.info("  --- monologue (sample 0) ---")
                            for turn_i, (toks, mask) in enumerate(
                                zip(out["_dialogue_tokens"], out["_dialogue_masks"]),
                            ):
                                ids = [t.item() for t, m in zip(toks[0], mask[0])
                                       if m and t.item() != eos_id and t.item() < vocab_size]
                                ipa = syllable_hyphenate(sp.decode(ids))
                                logger.info("  [T%d] %s  (%d tok)", turn_i, ipa[:120], len(ids))
                        else:
                            toks = out["_tokens"]
                            mask = out["_gen_mask"]
                            for j in range(min(5, toks.size(0))):
                                ids = [t.item() for t, m in zip(toks[j], mask[j])
                                       if m and t.item() != eos_id and t.item() < vocab_size]
                                ipa = syllable_hyphenate(sp.decode(ids))
                                logger.info("  sample[%d]: %s  (%d tok)", j, ipa[:100], len(ids))
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
