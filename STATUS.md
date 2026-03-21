# LFM Project Status

Last updated: 2026-03-20

## What Exists

### Project Setup
- Poetry project at `/home/daniel/projects/lfm/` (src layout)
- Python >=3.11, PyTorch >=2.0, Pydantic >=2.0
- Virtualenv managed by Poetry (NOT in project root)
- ruff for linting/formatting, pytest for tests
- All lint passing, all 15 tests passing

### Architecture Scaffold (81 Python files, 14 subpackages)

**Fully implemented:**
- `_registry.py` — `@register` / `create()` / `list_registered()` plugin system
- `_types.py` — Semantic tensor type aliases
- `config/base.py` — `LFMBaseConfig` (frozen Pydantic, extra="forbid")
- `config/experiment.py` — `ExperimentConfig` (top-level composition)
- `core/module.py` — `LFMModule` ABC (dict-return protocol, extra_losses)
- `core/loss.py` — `LFMLoss`, `CompositeLoss` with `from_config()` classmethod
- `utils/` — tensor helpers, Gumbel-softmax/straight-through sampling, logging
- `faculty/model.py` — `LanguageFaculty` compositor (wires pipeline, dim projections, output merging)
- `faculty/config.py` — `FacultyConfig` (aggregates all sub-module configs)
- `training/loop.py` — `TrainingLoop` (multi-phase, optimizer, scheduler, checkpointing)
- `training/phase.py` — `TrainingPhase` ABC (freeze/unfreeze, default_step, loss building)
- `training/callbacks.py` — `Callback` protocol, `LoggingCallback`, `CheckpointCallback`
- `training/config.py` — `TrainingConfig`, `PhaseConfig`, `OptimizerConfig`, `SchedulerConfig`

**Fully defined interfaces (ABCs with typed signatures):**
- `quantization/base.py` — `Quantizer` (forward, lookup, codebook_size/dim)
- `phonology/base.py` — `PhonologyModule` (forward, to_phonemes)
- `morphology/base.py` — `MorphologyModule` (forward, segment)
- `syntax/base.py` — `SyntaxModule` (forward, constrain_attention)
- `sentence/base.py` — `SentenceModule` (forward)
- `channel/base.py` — `Channel` (forward, decode)
- `metrics/base.py` — `Metric` (compute, update, result, reset)
- All configs for each module

**Registered stubs (correct class hierarchy, @register, raise NotImplementedError):**
- Quantizers: `vqvae`, `fsq`, `lfq`
- Phonology: `pronounceable`, `constraints`, `inventory`
- Morphology: `mdl_segmenter`, `composer`, `hierarchical`
- Syntax: `structural_agreement`, `ordering_pressure`
- Sentence: `type_head`, `boundary_detector`
- Channel: `gumbel`, `straight_through`, `noisy`
- Losses: `agreement_consistency`, `morphological_well_formedness`, `ordering_regularity`, `case_marking_pressure`, `morpheme_reuse`, `productive_combination`, `codebook_utilization`, `entropy_regularization`, `paraphrastic_diversity`, `anti_collapse`, `segmentation_coherence`, `pronounceability`
- Metrics: 11 metric classes across compositionality, structural, information, expressivity, non-isomorphism
- Data: `CorpusDataset`, `MultilingualCorpusDataset`, `AgentDataset`, `variable_length_collate`
- Training phases: `structural_priors`, `corruption`, `morphological_emergence`, `paraphrastic`, `agent_integration`

### Tests
- `test_registry.py` — register, create, list, auto-naming (4 tests)
- `test_config.py` — frozen configs, extra field rejection, defaults, composition (6 tests)
- `test_faculty.py` — instantiation, quantizer wiring, projections, registry population (5 tests)

### Documentation
- `CLAUDE.md` — project context, design principles, architecture overview, commands
- `README.md` — high-level project description
- Literature review files at `~/.claude/plans/splendid-seeking-plum-agent-*.md` (60+ papers)

### Games subpackage (`lfm/games/`) — FULLY IMPLEMENTED
- `config.py` — SceneConfig, SceneEncoderConfig, SceneDecoderConfig, ReconstructionGameConfig, ReferentialGameConfig
- `scenes.py` — SceneGenerator (GPU tensor generation, configurable objects/attributes/relations)
- `encoder.py` — SceneEncoder (one-hot + MLP), SceneDecoder (MLP + per-attribute heads), MessagePooler (lazy projection)
- `reconstruction.py` — ReconstructionGame (self-play autoencoder)
- `referential.py` — ReferentialGame + ReceiverModule (Lewis signaling game with separate sender/receiver encoders)
- `losses.py` — SceneReconstructionLoss, ReferentialLoss (registered)
- `metrics.py` — AttributeAccuracy, RelationAccuracy, ReferentialAccuracy, MessageEntropy, MessageUniqueness
- `phases.py` — ReconstructionGamePhase, ReferentialGamePhase (registered, generate own data)
- `__init__.py` — re-exports + `run_reconstruction_game()` / `run_referential_game()` convenience runners
- 11 tests passing

### Embedding corpus pipeline (`lfm/embeddings/`) — FULLY IMPLEMENTED
- `config.py` — TextEncoderConfig, ChunkerConfig, ClusterConfig, EmbeddingStoreConfig, SamplerConfig, PrecomputePipelineConfig, EmbeddingGameConfig
- `encoder.py` — TextEncoder ABC + SentenceTransformersEncoder (stella_en_400M_v5 default, swappable)
- `chunker.py` — TextChunker (sliding window, token-based, handles .txt and .jsonl)
- `store.py` — EmbeddingStore (mmap .npy + cluster index JSON, O(1) random access)
- `sampler.py` — StratifiedSampler (round-robin clusters, curriculum difficulty ramp, hard negative selection)
- `prefetcher.py` — AsyncPrefetcher (background thread → pin_memory → bounded queue → non_blocking GPU transfer)
- `pipeline.py` — PrecomputePipeline (chunk → encode → cluster → store, supports resumption)
- `games.py` — EmbeddingReconstructionGame, EmbeddingReferentialGame
- `losses.py` — EmbeddingReconstructionLoss (cosine+MSE), EmbeddingReferentialLoss (CE) — registered
- `metrics.py` — EmbeddingReconstructionSimilarity, EmbeddingReferentialAccuracy, CurriculumDifficulty
- `phases.py` — EmbeddingReconstructionGamePhase, EmbeddingReferentialGamePhase — registered
- Optional deps: `sentence-transformers>=3.0`, `scikit-learn>=1.3` in `[tool.poetry.group.embeddings]`

### Tests
- `test_registry.py` — 4 tests
- `test_config.py` — 6 tests
- `test_faculty.py` — 5 tests
- `test_games.py` — 11 tests
- **26 total, all passing**

### Documentation
- `CLAUDE.md` — project context, design principles, architecture overview, commands
- `README.md` — high-level project description with agent games motivation
- `STATUS.md` — this file
- Literature review files at `~/.claude/plans/splendid-seeking-plum-agent-*.md` (60+ papers)

## What Does NOT Exist Yet

### No concrete neural implementations for LFM pipeline modules
Every module's `forward()` raises NotImplementedError. No actual VQ-VAE, no morphology algorithms, no agreement/ordering modules, no phonotactic scoring -- just interfaces. Both game systems (scene-based and embedding-based) are fully wired but can't run end-to-end until these are implemented.

### No precomputed embedding store
The pipeline code exists but hasn't been run on a real corpus yet. Needs: a text corpus, GPU time for encoding, and disk space for the store.

## Decided Next Steps

### Priority 1: Core module implementations (to make games runnable)
- VQ-VAE quantizer — discretize agent_state into tokens
- Gumbel-softmax channel — differentiable discrete communication
- Codebook utilization loss — basic training signal

### Priority 2: First end-to-end run
- Run reconstruction game (scene-based or embedding-based) with VQ-VAE + Gumbel channel
- Verify gradients flow, losses decrease, reconstruction accuracy improves
- For scenes: compare flat (1 object) vs relational (3+ objects)
- For embeddings: precompute a small store from a text sample, run reconstruction

### Priority 3: Remaining modules (driven by game results)
- Phonology (pronounceability)
- MDL morphological segmenter
- Syntax: agreement and ordering pressure modules
- Remaining losses and metrics

## Key Design Decisions Made

1. **Dict-return protocol** — all `forward()` returns `dict[str, Tensor]`, namespaced by output_prefix
2. **Registry/factory** — `@register("category", "name")` + `create()` for all swappable components
3. **Frozen Pydantic configs** — immutable, `extra="forbid"`, hierarchically composable
4. **Phase-based training** — each phase = different loss weights + frozen modules configuration
5. **Phonology enabled by default** — English-biased phonotactics; emergent words must be pronounceable
6. **Lazy registration** — concrete modules imported via `_ensure_registry()` only when LanguageFaculty is instantiated
7. **extra_losses** — intrinsic module losses (commitment, etc.) always active, separate from phase-dependent CompositeLoss
8. **Morphology-centric architecture** — structure emerges from morphological agreement, case marking, and information-theoretic ordering rather than being imposed by explicit grammars (e.g., Neural PCFG). Syntax provides agreement and ordering pressure; phrase-like organization is an emergent property of morphological constraints.
