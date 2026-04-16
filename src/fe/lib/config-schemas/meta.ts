import { z } from "zod";

/**
 * UI metadata attached to each Zod field via `.meta()`.  Consumed by
 * the generic `<SchemaForm />` to decide how to render a field, which
 * accordion section to put it under, and what help text to show.
 *
 * Keep this intentionally small.  When a schema needs something truly
 * custom (e.g. the corpus picker), we add a new `inputKind` and wire
 * a renderer for it — the metadata itself stays declarative.
 */

export type InputKind =
  | "text"
  | "number"
  | "bool"
  | "select"
  | "corpus-select"
  | "int-list"
  | "path";

export type FieldMeta = {
  section: string;
  label: string;
  caption?: string;
  inputKind: InputKind;
  options?: readonly { value: string; label: string }[];
  // Optional soft bounds for number inputs — purely UI hints.
  min?: number;
  max?: number;
  step?: number;
  // Hide from the form UI but keep in the config shape.  Used for
  // fields that are derived from another field (e.g. dataset_path
  // derived from corpus-select).
  hidden?: boolean;
};

export type SectionDef = {
  key: string;
  label: string;
  caption?: string;
};

/**
 * Typed helper: apply UI metadata to a Zod field.  The return type is
 * the same as the input so schemas remain fully typed.
 */
export function ui<T extends z.ZodType>(schema: T, meta: FieldMeta): T {
  return schema.meta(meta) as T;
}

/**
 * Pull the `FieldMeta` back off a Zod field, if present.
 */
export function getFieldMeta(schema: z.ZodType): FieldMeta | undefined {
  const m = schema.meta();
  if (m && typeof m === "object" && "section" in m && "label" in m) {
    return m as FieldMeta;
  }
  return undefined;
}
