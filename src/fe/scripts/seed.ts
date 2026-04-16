/**
 * Seed initial phrase-VAE config from the v13 YAML that lives on disk.
 * Idempotent: re-running updates the existing row (by name) rather
 * than creating duplicates.
 *
 * Usage (inside the fe container): `pnpm db:seed`
 *   Reads   /workspace/lfm/configs/pretrain_vae_v13.yaml
 *   Writes  vae_models row { name: "v13-english-ortho", variant: phrase-vae }
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import yaml from "yaml";
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { eq, and } from "drizzle-orm";
import { PhraseVAEConfig } from "../lib/config-schemas/phrase-vae";
import { phraseVAEFromYaml } from "../lib/config-schemas/phrase-vae-yaml";
import * as schema from "../lib/db/schema";

const CONFIG_PATH =
  process.env.LFM_SEED_CONFIG ??
  resolve("/workspace/configs/pretrain_vae_v13.yaml");
const SEED_NAME = "v13-english-ortho";
const SEED_DESCRIPTION =
  "Multi-source English (Wikipedia + Gutenberg + arXiv) with phrase-type " +
  "brackets (<NP>, <VP>, <PP>, <S>, <SBAR>, ...), orthographic lowercased, " +
  "no standalone punctuation.  SPM 10k, KL-free decoder with DIP + length " +
  "boosting and v7-validated SS=7.5% + unlikelihood.  First phrase-aware " +
  "VAE in the LFM pipeline.";

async function main() {
  const raw = readFileSync(CONFIG_PATH, "utf8");
  const parsedYaml = yaml.parse(raw) as Record<string, unknown>;
  const canonical = phraseVAEFromYaml(parsedYaml);
  const config = PhraseVAEConfig.parse(canonical);

  const sql = postgres(
    process.env.DATABASE_URL ?? "postgres://lfm:lfm@localhost:5432/lfm",
    { max: 2 },
  );
  const db = drizzle(sql, { schema });

  const existing = await db
    .select({ id: schema.phraseVaeConfigPresets.id })
    .from(schema.phraseVaeConfigPresets)
    .where(
      and(
        eq(schema.phraseVaeConfigPresets.variant, "phrase-vae"),
        eq(schema.phraseVaeConfigPresets.name, SEED_NAME),
      ),
    )
    .limit(1);

  if (existing[0]) {
    await db
      .update(schema.phraseVaeConfigPresets)
      .set({
        config,
        description: SEED_DESCRIPTION,
        updatedAt: new Date(),
      })
      .where(eq(schema.phraseVaeConfigPresets.id, existing[0].id));
    console.log(`updated ${SEED_NAME} (${existing[0].id})`);
  } else {
    const [row] = await db
      .insert(schema.phraseVaeConfigPresets)
      .values({
        name: SEED_NAME,
        variant: "phrase-vae",
        status: "planned",
        config,
        description: SEED_DESCRIPTION,
      })
      .returning({ id: schema.phraseVaeConfigPresets.id });
    console.log(`inserted ${SEED_NAME} (${row?.id})`);
  }

  await sql.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
