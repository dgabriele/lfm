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
 * holds small, mutable app state — model configs, runs, annotations —
 * that benefits from transactions and cross-row queries.
 */

export const modelStatus = pgEnum("model_status", [
  "draft",
  "planned",
  "training",
  "trained",
  "archived",
]);

export const modelVariant = pgEnum("model_variant", [
  "phrase-vae",
]);

export const vaeModels = pgTable("vae_models", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull(),
  variant: modelVariant("variant").notNull(),
  status: modelStatus("status").notNull().default("draft"),
  // Full config as JSON — keeps schema flexible while the Zod schema
  // evolves.  Validated against `PhraseVAEConfig` at read/write edges.
  config: jsonb("config").notNull(),
  notes: text("notes"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export type VaeModelRow = typeof vaeModels.$inferSelect;
export type NewVaeModelRow = typeof vaeModels.$inferInsert;
