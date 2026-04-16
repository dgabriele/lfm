import {
  pgTable,
  uuid,
  text,
  jsonb,
  timestamp,
  pgEnum,
} from "drizzle-orm/pg-core";

/**
 * Drizzle schema for LFM app state.
 *
 * Corpus samples live in DuckDB/parquet (see `lib/duckdb.ts`).  Postgres
 * holds small, mutable app state — model configs, instantiated models,
 * runs, annotations — that benefits from transactions and cross-row
 * queries.
 *
 * Domain model:
 *   PhraseVAEConfigPreset
 *     A reusable, editable config "template" — the recipe for a phrase
 *     VAE.  Lives in the right-rail "Config presets" picker.
 *
 *   PhraseVAE
 *     An instantiated model object.  When the user "Use"s a preset,
 *     we copy its config into a snapshot owned by the new PhraseVAE.
 *     Edits to the source preset never propagate to instantiated VAEs.
 *     Eventually each row carries training run state, checkpoints, etc.
 */

export const modelVariant = pgEnum("model_variant", ["phrase-vae"]);

// PhraseVAEConfigPreset.status — describes the *recipe's* lifecycle.
export const presetStatus = pgEnum("model_status", [
  "draft",
  "planned",
  "training",
  "trained",
  "archived",
]);

// PhraseVAE.status — describes an *instantiated* VAE's lifecycle.
export const phraseVaeStatus = pgEnum("phrase_vae_status", [
  "initialized",
  "training",
  "paused",
  "trained",
  "failed",
  "archived",
]);

// What the model architecturally consumes.  Different decoder + SPM
// regime per variant; primarily a UI / discovery cue today.
export const phraseVaeType = pgEnum("phrase_vae_type", [
  "ipa",
  "token_vocab",
]);

/** Reusable config templates — the recipe for a phrase VAE. */
export const phraseVaeConfigPresets = pgTable("phrase_vae_config_presets", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull().unique(),
  variant: modelVariant("variant").notNull(),
  status: presetStatus("status").notNull().default("draft"),
  // Full config as JSON — keeps the schema flexible while the Zod
  // schema evolves.  Validated against PhraseVAEConfig (Zod) at
  // read/write edges.
  config: jsonb("config").notNull(),
  description: text("description"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

/** Instantiated phrase VAEs — what the user actually trains and ships. */
export const phraseVaes = pgTable("phrase_vaes", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull().unique(),
  description: text("description"),
  status: phraseVaeStatus("status").notNull().default("initialized"),
  vaeType: phraseVaeType("vae_type").notNull().default("token_vocab"),
  // Stable string id from the corpora registry (DuckDB) — kept loose
  // (no FK) since corpora live in a separate engine.
  corpusId: text("corpus_id").notNull(),
  // Snapshot of the preset's config at instantiation time.  Never
  // updated by edits to the source preset.
  config: jsonb("config").notNull(),
  // Source preset reference — nullable so we can SET NULL if the
  // preset is deleted; the VAE keeps its snapshot regardless.
  presetId: uuid("preset_id").references(() => phraseVaeConfigPresets.id, {
    onDelete: "set null",
  }),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export type PhraseVaeConfigPresetRow = typeof phraseVaeConfigPresets.$inferSelect;
export type NewPhraseVaeConfigPresetRow = typeof phraseVaeConfigPresets.$inferInsert;
export type PhraseVaeRow = typeof phraseVaes.$inferSelect;
export type NewPhraseVaeRow = typeof phraseVaes.$inferInsert;
