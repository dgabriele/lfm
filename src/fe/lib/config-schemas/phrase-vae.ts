import { z } from "zod";
import { ui, type SectionDef } from "./meta";

/**
 * Phrase-VAE training config.  Mirrors `configs/pretrain_vae_v13.yaml`
 * field-for-field, with UI metadata attached via `ui()` for each
 * entry.  Adding a field here lights it up in the editor form + YAML
 * preview automatically.
 *
 * One conceptual detour: the YAML has separate `dataset_path` and
 * `spm_model_path` fields, both pointing into a single dataset.  The
 * UI collapses those into one `corpus_id` selector driven by the
 * DuckDB corpora registry; YAML serialization expands it back out.
 * See `lib/config-schemas/phrase-vae-yaml.ts` for the transform.
 */

export const PHRASE_VAE_SECTIONS: SectionDef[] = [
  { key: "data", label: "Data & tokenizer" },
  { key: "arch", label: "Architecture" },
  { key: "reg", label: "KL & regularization" },
  { key: "aux", label: "Auxiliary objectives" },
  { key: "train", label: "Training schedule" },
  { key: "posterior", label: "Posterior shaping" },
  { key: "length", label: "Length boosting" },
  { key: "sampling", label: "Scheduled sampling & unlikelihood" },
  { key: "adversarial", label: "Adversarial & topology" },
  { key: "ops", label: "Ops" },
];

export const PhraseVAEConfig = z.object({
  // ── Data & tokenizer ────────────────────────────────────────────
  corpus_id: ui(
    z.string().default("english-constituents-v13"),
    {
      section: "data",
      label: "Corpus",
      caption:
        "Training dataset.  Select from registered corpora; dataset_path and spm_model_path in the emitted YAML derive from this choice.",
      inputKind: "corpus-select",
    },
  ),
  spm_vocab_size: ui(
    z.number().int().min(1000).max(64000).default(10000),
    {
      section: "data",
      label: "SPM vocab size",
      caption:
        "Subword vocabulary size.  The tokenizer is trained against the selected corpus.",
      inputKind: "number",
      min: 1000,
      max: 64000,
      step: 1000,
    },
  ),
  syllable_aligned_bpe: ui(
    z.boolean().default(false),
    {
      section: "data",
      label: "Syllable-aligned BPE",
      caption:
        "Train SPM with syllable-aligned pre-tokenization.  Only meaningful for syllable-hyphenated IPA corpora.",
      inputKind: "bool",
    },
  ),
  constituent_context: ui(
    z.boolean().default(false),
    {
      section: "data",
      label: "Constituent context",
      caption:
        "Prepend parent-phrase context tokens to each sample.  Off for v13-style phrase-tag corpora.",
      inputKind: "bool",
    },
  ),

  // ── Architecture ────────────────────────────────────────────────
  latent_dim: ui(z.number().int().default(256), {
    section: "arch",
    label: "Latent dim",
    caption: "Dimensionality of each z memory token.",
    inputKind: "number",
  }),
  encoder_num_layers: ui(z.number().int().default(2), {
    section: "arch",
    label: "Encoder layers",
    inputKind: "number",
  }),
  decoder_hidden_dim: ui(z.number().int().default(512), {
    section: "arch",
    label: "Decoder hidden dim",
    inputKind: "number",
  }),
  decoder_num_layers: ui(z.number().int().default(4), {
    section: "arch",
    label: "Decoder layers",
    caption:
      "Unique decoder layers; when share_decoder_layers is on, each is reused N times.",
    inputKind: "number",
  }),
  decoder_num_heads: ui(z.number().int().default(8), {
    section: "arch",
    label: "Decoder attention heads",
    inputKind: "number",
  }),
  decoder_dropout: ui(z.number().min(0).max(1).default(0.1), {
    section: "arch",
    label: "Decoder dropout",
    inputKind: "number",
    step: 0.05,
  }),
  num_memory_tokens: ui(z.number().int().default(8), {
    section: "arch",
    label: "Memory tokens",
    caption: "Number of z slots the encoder emits (K).",
    inputKind: "number",
  }),
  attention_head_windows: ui(
    z.array(z.number().int()).default([3, 3, 7, 7, 15, 15, 0, 0]),
    {
      section: "arch",
      label: "Per-head attention windows",
      caption:
        "Window radius per head; 0 means full (global).  Length should equal decoder_num_heads.",
      inputKind: "int-list",
    },
  ),
  attention_global_every: ui(z.number().int().default(7), {
    section: "arch",
    label: "Global-attention every N layers",
    inputKind: "number",
  }),
  use_rope: ui(z.boolean().default(true), {
    section: "arch",
    label: "RoPE positional encoding",
    inputKind: "bool",
  }),
  share_decoder_layers: ui(z.boolean().default(true), {
    section: "arch",
    label: "Share decoder layers",
    caption:
      "Weight-share unique decoder blocks across the full stack (4 unique × N).",
    inputKind: "bool",
  }),
  max_seq_len: ui(z.number().int().default(0), {
    section: "arch",
    label: "Max sequence length",
    caption: "0 = infer from dataset.",
    inputKind: "number",
  }),
  encoder_pooling: ui(
    z.enum(["mean", "max", "attention"]).default("mean"),
    {
      section: "arch",
      label: "Encoder pooling",
      inputKind: "select",
      options: [
        { value: "mean", label: "Mean" },
        { value: "max", label: "Max" },
        { value: "attention", label: "Attention" },
      ],
    },
  ),

  // ── KL & regularization ────────────────────────────────────────
  kl_weight: ui(z.number().default(0.0), {
    section: "reg",
    label: "KL weight",
    inputKind: "number",
    step: 0.01,
  }),
  kl_free_bits: ui(z.number().default(0.5), {
    section: "reg",
    label: "KL free bits",
    caption: "Per-dim KL below this threshold is ignored.",
    inputKind: "number",
    step: 0.1,
  }),
  kl_warmup_steps: ui(z.number().int().default(10000), {
    section: "reg",
    label: "KL warmup steps",
    inputKind: "number",
  }),
  kl_beta: ui(z.number().default(0.0), {
    section: "reg",
    label: "KL beta (target)",
    inputKind: "number",
    step: 0.01,
  }),

  // ── Auxiliary objectives ───────────────────────────────────────
  contrastive_weight: ui(z.number().default(0.0), {
    section: "aux",
    label: "Contrastive weight",
    inputKind: "number",
    step: 0.01,
  }),
  contrastive_temperature: ui(z.number().default(0.07), {
    section: "aux",
    label: "Contrastive temperature",
    inputKind: "number",
    step: 0.01,
  }),
  embeddings_path: ui(z.string().default(""), {
    section: "aux",
    label: "Embeddings path",
    caption:
      "Precomputed sentence embeddings (for contrastive / topo losses).  Leave empty to disable.",
    inputKind: "path",
  }),
  phonetic_init: ui(z.boolean().default(false), {
    section: "aux",
    label: "Phonetic init",
    inputKind: "bool",
  }),
  phonetic_init_scale: ui(z.number().default(0.5), {
    section: "aux",
    label: "Phonetic init scale",
    inputKind: "number",
    step: 0.1,
  }),
  phonetic_label_smoothing: ui(z.number().default(0.0), {
    section: "aux",
    label: "Phonetic label smoothing",
    inputKind: "number",
    step: 0.01,
  }),
  use_vq: ui(z.boolean().default(false), {
    section: "aux",
    label: "Vector quantization",
    inputKind: "bool",
  }),

  // ── Training schedule ──────────────────────────────────────────
  batch_size: ui(z.number().int().default(200), {
    section: "train",
    label: "Batch size",
    inputKind: "number",
  }),
  gradient_accumulation_steps: ui(z.number().int().default(2), {
    section: "train",
    label: "Gradient accumulation",
    inputKind: "number",
  }),
  use_amp: ui(z.boolean().default(true), {
    section: "train",
    label: "Mixed precision (AMP)",
    inputKind: "bool",
  }),
  lr: ui(z.number().default(0.0005), {
    section: "train",
    label: "Learning rate",
    inputKind: "number",
    step: 0.0001,
  }),
  lr_min: ui(z.number().default(0.0001), {
    section: "train",
    label: "LR floor (cosine min)",
    inputKind: "number",
    step: 0.00001,
  }),
  num_epochs: ui(z.number().int().default(5), {
    section: "train",
    label: "Epochs",
    inputKind: "number",
  }),

  // ── Posterior shaping ──────────────────────────────────────────
  z_var_weight: ui(z.number().default(0.01), {
    section: "posterior",
    label: "z-variance weight",
    inputKind: "number",
    step: 0.01,
  }),
  z_var_target: ui(z.number().default(0.03), {
    section: "posterior",
    label: "z-variance target",
    inputKind: "number",
    step: 0.01,
  }),
  z_var_floor: ui(z.number().default(0.01), {
    section: "posterior",
    label: "z-variance floor",
    inputKind: "number",
    step: 0.01,
  }),
  word_dropout: ui(z.number().default(0.0), {
    section: "posterior",
    label: "Word dropout",
    inputKind: "number",
    step: 0.05,
  }),
  word_dropout_min: ui(z.number().default(0.0), {
    section: "posterior",
    label: "Word dropout floor",
    inputKind: "number",
    step: 0.05,
  }),
  word_dropout_anneal_epochs: ui(z.number().int().default(0), {
    section: "posterior",
    label: "Word dropout anneal epochs",
    inputKind: "number",
  }),
  dip_weight: ui(z.number().default(0.1), {
    section: "posterior",
    label: "DIP-VAE weight",
    caption: "Penalizes off-diagonal covariance of z.",
    inputKind: "number",
    step: 0.05,
  }),
  bow_weight: ui(z.number().default(0.0), {
    section: "posterior",
    label: "Bag-of-words weight",
    inputKind: "number",
    step: 0.05,
  }),

  // ── Length boost ───────────────────────────────────────────────
  length_boost_threshold: ui(z.number().int().default(40), {
    section: "length",
    label: "Boost threshold (tokens)",
    inputKind: "number",
  }),
  length_boost_factor: ui(z.number().default(15.0), {
    section: "length",
    label: "Boost factor",
    inputKind: "number",
    step: 0.5,
  }),

  // ── Scheduled sampling & unlikelihood ──────────────────────────
  unlikelihood_weight: ui(z.number().default(0.15), {
    section: "sampling",
    label: "Unlikelihood weight",
    caption: "Anti-reduplication penalty over recent window.",
    inputKind: "number",
    step: 0.05,
  }),
  unlikelihood_window: ui(z.number().int().default(4), {
    section: "sampling",
    label: "Unlikelihood window",
    inputKind: "number",
  }),
  scheduled_sampling_target: ui(z.number().default(0.075), {
    section: "sampling",
    label: "SS target",
    caption: "Target probability of feeding model's own prediction at each step.",
    inputKind: "number",
    step: 0.005,
  }),
  scheduled_sampling_start_epoch: ui(z.number().int().default(0), {
    section: "sampling",
    label: "SS start epoch",
    inputKind: "number",
  }),
  scheduled_sampling_warmup_epochs: ui(z.number().int().default(1), {
    section: "sampling",
    label: "SS warmup epochs",
    inputKind: "number",
  }),

  // ── Adversarial & topology ─────────────────────────────────────
  topo_weight: ui(z.number().default(0.0), {
    section: "adversarial",
    label: "Topology weight",
    inputKind: "number",
    step: 0.01,
  }),
  topo_sample_pairs: ui(z.number().int().default(32), {
    section: "adversarial",
    label: "Topo sample pairs",
    inputKind: "number",
  }),
  use_adversarial: ui(z.boolean().default(false), {
    section: "adversarial",
    label: "Adversarial training",
    inputKind: "bool",
  }),
  adv_weight: ui(z.number().default(0.1), {
    section: "adversarial",
    label: "Adversarial weight",
    inputKind: "number",
    step: 0.05,
  }),
  adv_lr: ui(z.number().default(0.0001), {
    section: "adversarial",
    label: "Adversary LR",
    inputKind: "number",
    step: 0.00001,
  }),
  adv_disc_hidden: ui(z.number().int().default(256), {
    section: "adversarial",
    label: "Adversary hidden dim",
    inputKind: "number",
  }),
  adv_disc_embed_dim: ui(z.number().int().default(128), {
    section: "adversarial",
    label: "Adversary embed dim",
    inputKind: "number",
  }),
  adv_warmup_steps: ui(z.number().int().default(1000), {
    section: "adversarial",
    label: "Adversary warmup",
    inputKind: "number",
  }),
  adv_free_run_len: ui(z.number().int().default(32), {
    section: "adversarial",
    label: "Free-run length",
    inputKind: "number",
  }),
  adv_spectral_norm: ui(z.boolean().default(true), {
    section: "adversarial",
    label: "Adversary spectral norm",
    inputKind: "bool",
  }),

  // ── Ops ────────────────────────────────────────────────────────
  diagnostic_every: ui(z.number().int().default(5000), {
    section: "ops",
    label: "Diagnostic every N steps",
    inputKind: "number",
  }),
  checkpoint_every_steps: ui(z.number().int().default(5000), {
    section: "ops",
    label: "Checkpoint every N steps",
    inputKind: "number",
  }),
  val_fraction: ui(z.number().default(0.05), {
    section: "ops",
    label: "Validation fraction",
    inputKind: "number",
    step: 0.005,
  }),
  seed: ui(z.number().int().default(42), {
    section: "ops",
    label: "Random seed",
    inputKind: "number",
  }),
  device: ui(z.enum(["cuda", "cpu", "mps"]).default("cuda"), {
    section: "ops",
    label: "Device",
    inputKind: "select",
    options: [
      { value: "cuda", label: "CUDA" },
      { value: "cpu", label: "CPU" },
      { value: "mps", label: "MPS" },
    ],
  }),
  output_path: ui(
    z.string().default("/workspace/lfm/data/models/v13_english_ortho/vae_decoder.pt"),
    {
      section: "ops",
      label: "Output checkpoint path",
      inputKind: "path",
    },
  ),
});

export type PhraseVAEConfigShape = z.infer<typeof PhraseVAEConfig>;

/** Default config derived from Zod defaults. */
export function phraseVAEDefaults(): PhraseVAEConfigShape {
  return PhraseVAEConfig.parse({});
}
