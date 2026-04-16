import "server-only";
import { desc, eq, ne } from "drizzle-orm";
import { db, schema } from "@/lib/db/client";

/**
 * Server-only read path for `vae_models`.  Mutations live in
 * `lib/models/actions.ts` (server actions) so client components don't
 * import this file.
 */

export async function listPhraseVAEModels() {
  return db
    .select()
    .from(schema.vaeModels)
    .where(eq(schema.vaeModels.variant, "phrase-vae"))
    .orderBy(desc(schema.vaeModels.updatedAt));
}

export async function getModel(id: string) {
  const rows = await db
    .select()
    .from(schema.vaeModels)
    .where(eq(schema.vaeModels.id, id))
    .limit(1);
  return rows[0] ?? null;
}

/**
 * Names of every existing model config, optionally excluding one row
 * by id (so the edit page can skip its own row when checking for
 * conflicts).  Used by the editor to block duplicate names before the
 * DB unique constraint trips.
 */
export async function listVaeModelNames(excludeId?: string) {
  const rows = await db
    .select({ name: schema.vaeModels.name })
    .from(schema.vaeModels)
    .where(excludeId ? ne(schema.vaeModels.id, excludeId) : undefined);
  return rows.map((r) => r.name);
}
