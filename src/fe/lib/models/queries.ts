import "server-only";
import { desc, eq } from "drizzle-orm";
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
