"use server";

import { eq } from "drizzle-orm";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { db, schema } from "@/lib/db/client";
import {
  PhraseVAEConfig,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";

// Postgres unique-violation SQLSTATE.  Mapped to a friendlier error so
// the client-side check is backstopped cleanly if it's ever bypassed.
const PG_UNIQUE_VIOLATION = "23505";

function isUniqueViolation(err: unknown): err is { code: string } {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code?: string }).code === PG_UNIQUE_VIOLATION
  );
}

/**
 * Server actions for phrase-VAE model configs.  Validation happens
 * against the Zod schema at the boundary so a misshapen client state
 * can't corrupt the DB.
 */

export async function createPhraseVAEModel(input: {
  name: string;
  description: string;
  config: PhraseVAEConfigShape;
}) {
  const parsed = PhraseVAEConfig.parse(input.config);
  let row;
  try {
    [row] = await db
      .insert(schema.vaeModels)
      .values({
        name: input.name,
        description: input.description || null,
        variant: "phrase-vae",
        status: "draft",
        config: parsed,
      })
      .returning({ id: schema.vaeModels.id });
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A config named "${input.name}" already exists.`);
    }
    throw err;
  }
  revalidatePath("/models/phrase-vae");
  if (!row) throw new Error("insert returned no row");
  redirect(`/models/phrase-vae/${row.id}/edit`);
}

export async function updatePhraseVAEModel(
  id: string,
  input: { name: string; description: string; config: PhraseVAEConfigShape },
) {
  const parsed = PhraseVAEConfig.parse(input.config);
  try {
    await db
      .update(schema.vaeModels)
      .set({
        name: input.name,
        description: input.description || null,
        config: parsed,
        updatedAt: new Date(),
      })
      .where(eq(schema.vaeModels.id, id));
  } catch (err) {
    if (isUniqueViolation(err)) {
      throw new Error(`A config named "${input.name}" already exists.`);
    }
    throw err;
  }
  revalidatePath("/models/phrase-vae");
  revalidatePath(`/models/phrase-vae/${id}/edit`);
}

/** Bump `updated_at` so the row rises to the top of the MRU list. */
export async function touchPhraseVAEModel(id: string) {
  await db
    .update(schema.vaeModels)
    .set({ updatedAt: new Date() })
    .where(eq(schema.vaeModels.id, id));
  revalidatePath("/models/phrase-vae");
}

export async function deletePhraseVAEModel(id: string) {
  await db.delete(schema.vaeModels).where(eq(schema.vaeModels.id, id));
  revalidatePath("/models/phrase-vae");
  redirect("/models/phrase-vae");
}
