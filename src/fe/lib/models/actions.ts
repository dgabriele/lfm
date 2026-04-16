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
 * Server actions for phrase-VAE model configs.  Validation happens
 * against the Zod schema at the boundary so a misshapen client state
 * can't corrupt the DB.
 */

export async function createPhraseVAEModel(input: {
  name: string;
  config: PhraseVAEConfigShape;
}) {
  const parsed = PhraseVAEConfig.parse(input.config);
  const [row] = await db
    .insert(schema.vaeModels)
    .values({
      name: input.name,
      variant: "phrase-vae",
      status: "draft",
      config: parsed,
    })
    .returning({ id: schema.vaeModels.id });
  revalidatePath("/models/phrase-vae");
  if (!row) throw new Error("insert returned no row");
  redirect(`/models/phrase-vae/${row.id}/edit`);
}

export async function updatePhraseVAEModel(
  id: string,
  input: { name: string; config: PhraseVAEConfigShape },
) {
  const parsed = PhraseVAEConfig.parse(input.config);
  await db
    .update(schema.vaeModels)
    .set({
      name: input.name,
      config: parsed,
      updatedAt: new Date(),
    })
    .where(eq(schema.vaeModels.id, id));
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
