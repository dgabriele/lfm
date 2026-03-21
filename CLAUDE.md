# LFM — Language Faculty Model

## What This Is

LFM is a framework for GPU-based multi-agent systems that need a natural language faculty. It provides a learnable system that imposes morphosyntactic and sentence-level constraints on sequences, enabling agents to express internal representations in structured and compositional form — without encoding predefined semantics.

LFM models the *faculty* of language, not any particular human language.

## Core Concepts

- **Language Faculty**: A constraint layer over communication — a bias toward language-like structure, a scaffold for emergent semantics. It does NOT define meaning, ontology, or enforce alignment with human concepts.
- **Agent Pipeline**: Grounded system -> representation (e.g. VQ-VAE codes) -> internal inference -> LFM (language faculty) -> emergent communication -> optional projection layer for human interpretation.
- **Structural Properties**: Compositionality, morphological structure, emergent structure -- phrase-like organization arising from morphological agreement, case marking, and information-theoretic ordering -- sentence type differentiation, paraphrastic capacity.
- **Phonotactic Constraints**: Emergent morphemes and words must be pronounceable. Phonotactic constraint is achieved via implicit surface-form smoothness — a GRU-based sequential predictor, energy contour regularization, and batch diversity pressure — without encoding explicit phonological categories (vowels, consonants, sonority). This is a form constraint, not a meaning constraint.

## Design Principles

1. **Flexible & Extensible** — Research framework. Every component is swappable, configurable, and composable via registry/factory pattern. No hard-coded single-configuration implementations.
2. **DRY & OOP** — Proper abstractions, base classes. Factor common behavior into reusable components.
3. **GPU-Centric** — All core compute paths designed for GPU execution. PyTorch tensors throughout. Batch everything.
4. **Good Configuration** — Pydantic models for all configs (frozen, extra="forbid"). Thoughtful, data-driven adaptive defaults.
5. **Constraint, Not Prescription** — LFM constrains outputs toward structured forms without dictating exact expressions.
6. **Non-Invertibility** — Mappings from internal representations to language must not be perfectly reversible.
7. **Structure Without Semantics** — Training rewards well-formedness, compositional reuse, structural consistency. NOT semantic correctness relative to human language.
8. **Typological Neutrality** — The architecture and losses must not bias toward any particular language typology. Whether the emergent language is isolating (Chinese-like), agglutinative (Turkish-like), polysynthetic (Mohawk-like), fusional (Latin-like), or a hybrid with no human analogue — that is determined by communication pressure, not architectural bias. Losses reward structural consistency and communicative success, not any particular kind of structure.

## What LFM Is NOT

- Not a traditional language model — it does not learn or reproduce human language semantics
- Not a fixed vocabulary/grammar system — structure must emerge and adapt
- Not a cipher or re-encoding of existing human languages
- Not a semantic embedding space tied to human concepts

## Architecture

### Pipeline (LanguageFaculty)

```
AgentState -> Quantizer -> Phonology -> Morphology -> Syntax -> Sentence -> Channel -> Message
```

Syntax provides structural agreement and ordering pressure -- phrase structure emerges from morphological constraints rather than being imposed by explicit grammars.

Each stage is optional (set config to None to skip). Phonology is enabled by default.

### Package Structure

```
src/lfm/
  _registry.py          # @register / create() / list_registered() — plugin system
  _types.py             # Tensor type aliases (AgentState, TokenIds, Mask, etc.)
  config/               # LFMBaseConfig, ExperimentConfig
  core/                 # LFMModule (ABC), LFMLoss, CompositeLoss
  quantization/         # Quantizer ABC + VQ-VAE, FSQ, LFQ implementations
  phonology/            # PhonologyModule ABC + pronounceability scorer, phonotactics
  morphology/           # MorphologyModule ABC + MDL segmenter, composer, tree tokenizer
  syntax/               # SyntaxModule ABC + structural agreement, ordering pressure
  sentence/             # SentenceModule ABC + type head, boundary detector
  channel/              # Channel ABC + straight-through, Gumbel-softmax, noisy channel
  losses/               # Structural, compositionality, information, diversity, morphological losses
  faculty/              # FacultyConfig + LanguageFaculty compositor
  training/             # TrainingLoop, TrainingPhase, Callbacks, 5 training phases
  data/                 # CorpusDataset, AgentDataset, collation
  metrics/              # Compositionality, structural, information, expressivity, non-isomorphism
  utils/                # Tensor helpers, sampling (Gumbel, straight-through), logging
```

### Key Patterns

- **Registry**: `@register("category", "name")` decorator + `create("category", "name", config)` factory
- **Dict-return protocol**: All `LFMModule.forward()` returns `dict[str, Tensor]`, namespaced by output_prefix
- **Frozen configs**: All Pydantic configs are immutable with `extra="forbid"`
- **Phase-based training**: Each phase configures which losses are active via `dict[str, float]`
- **extra_losses**: Modules with intrinsic losses (e.g. commitment loss) expose via `extra_losses()`, always active

### Training Phases

1. **structural_priors** — Learn structural priors from multilingual corpora
2. **corruption** — Structural corruption and recomposition
3. **morphological_emergence** — Morphological emergence pressure
4. **paraphrastic** — Paraphrastic generation diversity
5. **agent_integration** — Agent-integrated training

## Commands

- `poetry install` — Install dependencies
- `poetry run pytest` — Run tests
- `poetry run ruff check src/` — Lint
- `poetry run ruff format src/` — Format

## Tech Stack

- Python 3.11+
- PyTorch (GPU compute)
- Pydantic v2 (configuration & validation)
- Poetry (dependency management, virtualenv NOT in project root)
- pytest (testing)
- ruff (linting & formatting)

## Usage Example

```python
from lfm import LanguageFaculty, FacultyConfig, QuantizationConfig

faculty = LanguageFaculty(FacultyConfig(
    dim=128,
    quantizer=QuantizationConfig(name="vqvae", codebook_size=512),
))
```
