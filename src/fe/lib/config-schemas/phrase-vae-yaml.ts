import yaml from "yaml";
import {
  PhraseVAEConfig,
  PHRASE_VAE_SECTIONS,
  type PhraseVAEConfigShape,
} from "./phrase-vae";
import { getFieldMeta } from "./meta";

/**
 * Serialize a PhraseVAEConfig form state to the YAML shape the trainer
 * expects, with comment section headers mirroring the form's accordion
 * groups — so the preview reads top-to-bottom exactly like the
 * editor sections, and hand-editing the emitted YAML is painless.
 *
 * Transform: the form's single `corpus_id` is expanded to
 * `dataset_path` + `spm_model_path` under
 * `/workspace/lfm/data/datasets/<id>/…` (vast container convention).
 * Both derived paths live under the "Data & tokenizer" section.
 */

const WORKSPACE_DATASETS = "/workspace/lfm/data/datasets";

/** Derived YAML-only fields injected into the "data" section. */
const DERIVED_DATA_FIELDS: readonly string[] = ["dataset_path", "spm_model_path"];

export function phraseVAEToYaml(cfg: PhraseVAEConfigShape): string {
  const { corpus_id, ...rest } = cfg;
  const datasetPath = `${WORKSPACE_DATASETS}/${corpus_id}`;
  const spmModelPath = `${datasetPath}/spm.model`;

  // Bucket fields by section, preserving schema declaration order
  // inside each bucket.  Non-corpus_id fields are grouped from the
  // schema's own `ui(...)` metadata so form and YAML stay in lockstep.
  const bySection: Record<string, string[]> = {};
  for (const [key, field] of Object.entries(PhraseVAEConfig.shape)) {
    if (key === "corpus_id") continue;
    const meta = getFieldMeta(field);
    if (!meta) continue;
    (bySection[meta.section] ??= []).push(key);
  }
  // Put the two derived path fields at the head of the "data" section.
  bySection.data = [...DERIVED_DATA_FIELDS, ...(bySection.data ?? [])];

  const valueFor = (key: string): unknown => {
    if (key === "dataset_path") return datasetPath;
    if (key === "spm_model_path") return spmModelPath;
    return (rest as Record<string, unknown>)[key];
  };

  // Emit each section as a YAML sub-document, prefixed with a comment
  // header.  Using per-section sub-documents lets us precisely control
  // the order and attach headers without patching the serializer.
  const chunks: string[] = [];
  for (const section of PHRASE_VAE_SECTIONS) {
    const fields = bySection[section.key];
    if (!fields || fields.length === 0) continue;
    const block: Record<string, unknown> = {};
    for (const key of fields) {
      block[key] = valueFor(key);
    }
    const body = yaml.stringify(block, {
      lineWidth: 100,
      defaultStringType: "PLAIN",
      defaultKeyType: "PLAIN",
    });
    chunks.push(
      `# ${section.label}\n${section.caption ? `# ${wrapComment(section.caption, 78)}\n` : ""}${body}`,
    );
  }
  return chunks.join("\n");
}

/**
 * Inverse transform: given a YAML document with dataset_path +
 * spm_model_path, derive the `corpus_id` that would re-emit it.
 */
export function phraseVAEFromYaml(
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const { dataset_path, spm_model_path: _spm, ...rest } = raw;
  let corpus_id = "";
  if (typeof dataset_path === "string") {
    const prefix = `${WORKSPACE_DATASETS}/`;
    corpus_id = dataset_path.startsWith(prefix)
      ? dataset_path.slice(prefix.length)
      : dataset_path;
  }
  return { corpus_id, ...rest };
}

/** Wrap a long caption across multiple comment lines. */
function wrapComment(text: string, width: number): string {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if (cur.length === 0) {
      cur = w;
    } else if (cur.length + 1 + w.length <= width) {
      cur += ` ${w}`;
    } else {
      lines.push(cur);
      cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines.join("\n# ");
}
