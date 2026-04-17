"use server";

import { eq } from "drizzle-orm";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { db, schema } from "@/lib/db/client";
import {
  PhraseVAEConfig,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";

/**
 * Mutations split into "config preset" actions (the recipe templates)
 * and "phrase VAE" actions (instantiated models).
 *
 * Config validation happens at every write boundary so a misshapen
 * client state can't corrupt either table.
 */

// Postgres unique-violation SQLSTATE.
const PG_UNIQUE_VIOLATION = "23505";

function isUniqueViolation(err: unknown): err is { code: string } {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code?: string }).code === PG_UNIQUE_VIOLATION
  );
}

// ── Config preset actions (templates) ────────────────────────────

export async function createPhraseVAEConfigPreset(input: {
  name: string;
  description: string;
  config: PhraseVAEConfigShape;
}) {
  const parsed = PhraseVAEConfig.parse(input.config);
  let row;
  try {
    [row] = await db
      .insert(schema.phraseVaeConfigPresets)
      .values({
        name: input.name,
        description: input.description || null,
        variant: "phrase-vae",
        status: "draft",
        config: parsed,
      })
      .returning({ id: schema.phraseVaeConfigPresets.id });
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A preset named "${input.name}" already exists.`);
    }
    throw err;
  }
  revalidatePath("/models/phrase-vae");
  if (!row) throw new Error("insert returned no row");
  redirect(`/models/phrase-vae/presets/${row.id}/edit`);
}

export async function updatePhraseVAEConfigPreset(
  id: string,
  input: { name: string; description: string; config: PhraseVAEConfigShape },
) {
  const parsed = PhraseVAEConfig.parse(input.config);
  try {
    await db
      .update(schema.phraseVaeConfigPresets)
      .set({
        name: input.name,
        description: input.description || null,
        config: parsed,
        updatedAt: new Date(),
      })
      .where(eq(schema.phraseVaeConfigPresets.id, id));
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A preset named "${input.name}" already exists.`);
    }
    throw err;
  }
  revalidatePath("/models/phrase-vae");
  revalidatePath(`/models/phrase-vae/presets/${id}/edit`);
}

export async function touchPhraseVAEConfigPreset(id: string) {
  await db
    .update(schema.phraseVaeConfigPresets)
    .set({ updatedAt: new Date() })
    .where(eq(schema.phraseVaeConfigPresets.id, id));
  revalidatePath("/models/phrase-vae");
}

export async function deletePhraseVAEConfigPreset(id: string) {
  await db
    .delete(schema.phraseVaeConfigPresets)
    .where(eq(schema.phraseVaeConfigPresets.id, id));
  revalidatePath("/models/phrase-vae");
  redirect("/models/phrase-vae");
}

/**
 * Duplicate a preset into a new preset row — useful for forking a
 * known-good config and tweaking from there.  Auto-suffixes the new
 * name with " (copy)" / " (copy 2)" / … until unique.
 */
export async function duplicatePhraseVAEConfigPreset(id: string) {
  const sources = await db
    .select()
    .from(schema.phraseVaeConfigPresets)
    .where(eq(schema.phraseVaeConfigPresets.id, id))
    .limit(1);
  const src = sources[0];
  if (!src) throw new Error(`preset ${id} not found`);

  // Find an unused " (copy …)" name.
  const existing = new Set(
    (await db.select({ name: schema.phraseVaeConfigPresets.name })
      .from(schema.phraseVaeConfigPresets)).map((r) => r.name.toLowerCase()),
  );
  let candidate = `${src.name} (copy)`;
  let n = 2;
  while (existing.has(candidate.toLowerCase())) {
    candidate = `${src.name} (copy ${n})`;
    n += 1;
  }

  const [row] = await db
    .insert(schema.phraseVaeConfigPresets)
    .values({
      name: candidate,
      description: src.description,
      variant: src.variant,
      status: "draft",
      config: src.config,
    })
    .returning({ id: schema.phraseVaeConfigPresets.id });
  if (!row) throw new Error("insert returned no row");
  revalidatePath("/models/phrase-vae");
  redirect(`/models/phrase-vae/presets/${row.id}/edit`);
}

// ── Phrase VAE actions (instantiated models) ─────────────────────

/**
 * Instantiate a new PhraseVAE from a preset.  The preset's config
 * is snapshotted into the new VAE row — subsequent edits to the
 * preset never mutate this VAE.  The VAE's type is derived from the
 * selected corpus's `vae_type` (every corpus is intrinsically IPA or
 * token-vocab); the client doesn't get to choose.
 */
export async function createPhraseVAEFromPreset(input: {
  presetId: string;
  name: string;
  description: string;
  corpusId: string;
}) {
  // Re-fetch the preset to snapshot the latest version of its config.
  const presets = await db
    .select()
    .from(schema.phraseVaeConfigPresets)
    .where(eq(schema.phraseVaeConfigPresets.id, input.presetId))
    .limit(1);
  const preset = presets[0];
  if (!preset) throw new Error(`preset ${input.presetId} not found`);
  const snapshot = PhraseVAEConfig.parse(preset.config);

  // Authoritative vae_type comes from the corpus registry.
  const { getCorpus } = await import("@/lib/corpora");
  const corpus = await getCorpus(input.corpusId);
  if (!corpus) throw new Error(`corpus "${input.corpusId}" not found`);

  let row;
  try {
    [row] = await db
      .insert(schema.phraseVaes)
      .values({
        name: input.name,
        description: input.description || null,
        status: "initialized",
        vaeType: corpus.vaeType,
        corpusId: input.corpusId,
        config: snapshot,
        presetId: input.presetId,
      })
      .returning({ id: schema.phraseVaes.id });
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A phrase VAE named "${input.name}" already exists.`);
    }
    throw err;
  }
  if (!row) throw new Error("insert returned no row");
  revalidatePath("/models/phrase-vae");
  redirect(`/models/phrase-vae/${row.id}`);
}

export async function deletePhraseVAE(id: string) {
  await db.delete(schema.phraseVaes).where(eq(schema.phraseVaes.id, id));
  revalidatePath("/models/phrase-vae");
  redirect("/models/phrase-vae");
}

/**
 * Update a PhraseVAE instance — its snapshotted config, name, and
 * description.  The snapshot is the authoritative config for this VAE;
 * this action is how a user hand-tunes a VAE after instantiation
 * without touching the source preset.
 */
export async function updatePhraseVAE(
  id: string,
  input: { name: string; description: string; config: PhraseVAEConfigShape },
) {
  const parsed = PhraseVAEConfig.parse(input.config);
  try {
    await db
      .update(schema.phraseVaes)
      .set({
        name: input.name,
        description: input.description || null,
        config: parsed,
        updatedAt: new Date(),
      })
      .where(eq(schema.phraseVaes.id, id));
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A phrase VAE named "${input.name}" already exists.`);
    }
    throw err;
  }
  revalidatePath("/models/phrase-vae");
  revalidatePath(`/models/phrase-vae/${id}`);
  redirect(`/models/phrase-vae/${id}`);
}
