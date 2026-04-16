import "server-only";
import { desc, eq, ne } from "drizzle-orm";
import { db, schema } from "@/lib/db/client";

/**
 * Server-only read path.  Queries split between *config presets*
 * (reusable templates) and *phrase VAEs* (instantiated models).
 *
 * Mutations live in `lib/models/actions.ts` (server actions).
 */

// ── Config presets ────────────────────────────────────────────────

export async function listPhraseVAEConfigPresets() {
  return db
    .select()
    .from(schema.phraseVaeConfigPresets)
    .where(eq(schema.phraseVaeConfigPresets.variant, "phrase-vae"))
    .orderBy(desc(schema.phraseVaeConfigPresets.updatedAt));
}

export async function getPhraseVAEConfigPreset(id: string) {
  const rows = await db
    .select()
    .from(schema.phraseVaeConfigPresets)
    .where(eq(schema.phraseVaeConfigPresets.id, id))
    .limit(1);
  return rows[0] ?? null;
}

export async function listPhraseVAEConfigPresetNames(excludeId?: string) {
  const rows = await db
    .select({ name: schema.phraseVaeConfigPresets.name })
    .from(schema.phraseVaeConfigPresets)
    .where(
      excludeId
        ? ne(schema.phraseVaeConfigPresets.id, excludeId)
        : undefined,
    );
  return rows.map((r) => r.name);
}

// ── Phrase VAEs ───────────────────────────────────────────────────

export async function listPhraseVAEs() {
  return db
    .select()
    .from(schema.phraseVaes)
    .orderBy(desc(schema.phraseVaes.updatedAt));
}

export async function getPhraseVAE(id: string) {
  const rows = await db
    .select()
    .from(schema.phraseVaes)
    .where(eq(schema.phraseVaes.id, id))
    .limit(1);
  return rows[0] ?? null;
}

export async function listPhraseVAENames(excludeId?: string) {
  const rows = await db
    .select({ name: schema.phraseVaes.name })
    .from(schema.phraseVaes)
    .where(excludeId ? ne(schema.phraseVaes.id, excludeId) : undefined);
  return rows.map((r) => r.name);
}
