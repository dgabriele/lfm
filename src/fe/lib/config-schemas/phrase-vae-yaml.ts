import yaml from "yaml";
import type { PhraseVAEConfigShape } from "./phrase-vae";

/**
 * Serialize a PhraseVAEConfig form state to the exact YAML shape the
 * trainer expects.  One transform happens here: the form's single
 * `corpus_id` is expanded into `dataset_path` + `spm_model_path`
 * under `/workspace/lfm/data/datasets/<id>/...` — that's the path
 * convention the Python side uses inside the vast container.
 *
 * Keeps the in-UI schema shape small while the emitted YAML remains
 * identical to what we'd hand-write.
 */

const WORKSPACE_DATASETS = "/workspace/lfm/data/datasets";

export function phraseVAEToYaml(cfg: PhraseVAEConfigShape): string {
  const { corpus_id, ...rest } = cfg;
  const datasetPath = `${WORKSPACE_DATASETS}/${corpus_id}`;
  const spmModelPath = `${datasetPath}/spm.model`;

  // Emit fields in a stable, human-friendly order so the YAML preview
  // reads top-to-bottom the same way the source configs do.
  const ordered: Record<string, unknown> = {
    dataset_path: datasetPath,
    spm_model_path: spmModelPath,
    ...rest,
  };

  return yaml.stringify(ordered, {
    lineWidth: 100,
    defaultStringType: "PLAIN",
    defaultKeyType: "PLAIN",
  });
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
