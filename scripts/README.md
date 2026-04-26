# scripts/

Loose Python entry points used outside the `lfm` CLI: data-prep one-offs, ad-hoc diagnostics on trained checkpoints, decode-time hyperparameter sweeps, and quick agent-game-style probes. Anything reusable lives in `src/lfm/`; the files here are stand-alone runnables. Each script self-bootstraps `sys.path` to the repo's `src/` so it can be invoked directly with `python scripts/<subdir>/<name>.py` (or `poetry run python ...`).

The subdirectory layout reflects the role each script plays in the pipeline:

- `diagnostics/` — large-N empirical health checks of trained checkpoints.
- `preprocessing/` — one-time data preparation that has to run before training.
- `tuning/` — decode-time hyperparameter sweeps over a trained model.
- `probes/` — agent-game-style discriminability probes on a trained model.

---

## diagnostics/

Run after a training checkpoint lands (best.pt or resume.pt) to characterize what the model is actually doing. None of these mutate the model; all read a checkpoint, decode, and emit metrics + JSON.

### `diagnostics/lingdiag.py`
Large-N linguistic analysis of a DepTreeVAE checkpoint. Generates `--n` posterior samples (encode from the dep-tree cache, decode) and `--n` prior samples (z ~ N(0,1), decode), then reports syllable distribution, char-length distribution, top-20 word frequency, TTR / distinct-1/2/3 / hapax, Zipf exponent, bigram coverage against the training corpus, and posterior chrF vs. ground truth — all side-by-side with a training-data baseline.
Applies to: any DepTreeVAE checkpoint (`data/models/dep_tree_vae_v1/{best,resume}.pt` style).
Run: `python scripts/diagnostics/lingdiag.py --ckpt PATH --n 512 --out /tmp/lingdiag.json`.
Outputs: stdout report + JSON at `--out` (default `/tmp/lingdiag.json`).

### `diagnostics/diagnose_v3.py`
Root-cause diagnostic for reconstruction failures: stratifies a held-out sample by token rarity, runs encode→decode with instrumentation, and quantifies cycling rate, length error, length-head error, longest verbatim recovery, and z-stats (norm, logvar). Reports correlations between rarity / cycling / length error / verbatim and a failure-mode count breakdown.
Applies to: any DepTreeVAE checkpoint; specifically built around v3.3-era diagnostics.
Run: `python scripts/diagnostics/diagnose_v3.py --ckpt PATH --n 1024 --bins 5`.
Outputs: stdout tables + per-sample JSON at `/tmp/diagnose_v3_results.json` (override with `--out`).

### `diagnostics/disentangle_metrics.py`
Full disentanglement diagnostic. For a fixed batch of z samples, perturbs the struct dims and the content dims independently and measures: skeleton change rate, word-Jaccard similarity to base, length change, and a cross-recombination test (struct from A + content from B → does the output keep A's skeleton and B's words?). Loads `data/models/dep_tree_vae_v1/resume.pt` by default.
Applies to: DepTreeVAE checkpoints with the split-latent (struct/content) projector.
Run: `python scripts/diagnostics/disentangle_metrics.py` (paths hard-coded to vast layout — edit before running locally).
Outputs: stdout report only.

### `diagnostics/disentangle_v2.py`
Stripped-down version of `disentangle_metrics.py`: just the word-Jaccard struct-vs-content gap, swept over minimum-word-length thresholds (0 / 4 / 6 chars). Quick "is the gap positive yet?" check. Use this during training to track disentanglement progress without paying for the full skeleton + recombination measurement.
Applies to: DepTreeVAE checkpoints with split-latent projector.
Run: `python scripts/diagnostics/disentangle_v2.py`.
Outputs: stdout only.

## preprocessing/

One-time data prep that happens before pretraining starts.

### `preprocessing/compute_corpus_unigram.py`
Precomputes the training-corpus unigram distribution over the decoder vocab (SPM + BOS + EOS) from the dep-tree cache, applies add-k smoothing, and writes a `(V,)` `.npy` of probabilities. The DepTreeVAE model loads this at training start and uses it as the target of a KL regularizer on its batch-marginal output distribution (the well-formedness pressure described in `configs/dep_tree_vae_vast.yaml` as `corpus_kl_weight`).
When: once, after the dep-tree cache has been built and before kicking off pretraining (or any time the cache is rebuilt). Re-run if the SPM model or vocab changes.
Run: `python scripts/preprocessing/compute_corpus_unigram.py --cache data/datasets/english-dep-trees-v16/cache_depth4 --out <cache>/unigram.npy --smoothing 1.0`.
Outputs: `unigram.npy` of shape `(decoder_vocab,)` inside the cache directory by default.

## tuning/

Decode-time hyperparameter sweeps. Run after a checkpoint has stabilized to pick the best `eos_boost` / `expected_len` / `ngram_block` for inference.

### `tuning/run_autotune.py`
Loads `data/models/dep_tree_vae_v1/best.pt`, encodes a 512-sample held-out batch from the dep-tree cache to posterior mu, and grid-searches `DepTreeVAE.autotune(...)` over `eos_boosts × expected_lens × ngram_blocks`. Scores each combo with MiniLM-L6 semantic similarity to the source, plus length MAE, repetition, uniqueness, and EOS rate. Prints top-8 by composite, the winner per individual metric, and a delta vs. the current trainer baseline.
When: after a checkpoint reaches stable reconstruction quality, before locking in the decode defaults that ship in the config.
Run: `python scripts/tuning/run_autotune.py` (paths hard-coded; edit constants in `main` if pointing at a different checkpoint).
Outputs: stdout ranked tables; no files written.

## probes/

Agent-game-style discriminability probes — quick "would this checkpoint plausibly play a referential game?" sanity checks without standing up a real game trainer.

### `probes/probe_game.py`
Samples N=512 passages from the dep-tree cache, encodes through the DepTreeVAE posterior to mu, decodes with `_greedy_decode`, romanizes both source and decoded text with MiniLM-L6, and runs 16-way contrastive discrimination at random distractors (upper bound) and hard distractors (top-15 nearest in source-embedding space). Also reports full N-way ranking (top-1, top-3, median rank), uniqueness, mean pairwise cosine, and 8 qualitative sample reconstructions.
When: interactively after a DepTreeVAE checkpoint reaches non-trivial reconstruction CE, to estimate whether a downstream agent game would have any signal to learn from.
Run: `python scripts/probes/probe_game.py` (hard-coded to `data/models/dep_tree_vae_v1/best.pt`).
Outputs: stdout discrimination tables + sample reconstructions; no files written.
