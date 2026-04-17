"use client";

import { useId, useMemo, useState, useTransition } from "react";
import { Modal } from "@heroui/react";
import { unstable_rethrow } from "next/navigation";
import { Rocket, X } from "lucide-react";
import { createPhraseVAEFromPreset } from "@/lib/models/actions";
import type { PhraseVAEConfigShape } from "@/lib/config-schemas/phrase-vae";

/**
 * "Use" quick action: opens a modal that instantiates a new PhraseVAE
 * from this preset.  The preset's config is snapshotted into the new
 * VAE row server-side (subsequent edits to the preset don't propagate).
 *
 * Defaults:
 *   - name = "<preset-name>-1" (auto-incremented if taken).
 *   - description = blank.
 *   - corpus = preset's `corpus_id`.
 *   - vae_type = NOT user-selectable; derived from the chosen corpus
 *     (every corpus is intrinsically IPA or token-vocab).  Shown
 *     read-only so the user can see what they're getting.
 */

export type CorpusOption = {
  value: string;
  label: string;
  vaeType: "ipa" | "token_vocab";
};

const VAE_TYPE_LABEL: Record<string, string> = {
  ipa: "IPA (character / phoneme)",
  token_vocab: "Token vocab (BPE)",
};

export function UsePresetButton({
  presetId,
  presetName,
  presetConfig,
  corpora,
  existingVaeNames,
}: {
  presetId: string;
  presetName: string;
  presetConfig: PhraseVAEConfigShape;
  corpora: CorpusOption[];
  existingVaeNames: string[];
}) {
  const [open, setOpen] = useState(false);

  // Suggest an unused name based on the preset's name plus a counter.
  const taken = useMemo(
    () => new Set(existingVaeNames.map((n) => n.trim().toLowerCase())),
    [existingVaeNames],
  );
  const suggestedName = useMemo(() => {
    let i = 1;
    while (taken.has(`${presetName}-${i}`.toLowerCase())) i += 1;
    return `${presetName}-${i}`;
  }, [presetName, taken]);

  return (
    <>
      <button
        type="button"
        aria-label={`Use ${presetName} — instantiate a new phrase VAE`}
        title="Use as template for new phrase VAE"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 px-2 py-1 rounded-[calc(var(--radius)*0.5)] text-xs font-semibold text-accent bg-accent/10 hover:bg-accent/20 transition-colors"
      >
        <Rocket className="w-3.5 h-3.5" strokeWidth={2} />
        Use
      </button>
      {open && (
        <UsePresetModal
          presetId={presetId}
          presetName={presetName}
          defaultName={suggestedName}
          defaultCorpus={presetConfig.corpus_id}
          corpora={corpora}
          taken={taken}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function UsePresetModal({
  presetId,
  presetName,
  defaultName,
  defaultCorpus,
  corpora,
  taken,
  onClose,
}: {
  presetId: string;
  presetName: string;
  defaultName: string;
  defaultCorpus: string;
  corpora: CorpusOption[];
  taken: Set<string>;
  onClose: () => void;
}) {
  const nameId = useId();
  const descId = useId();
  const corpusFieldId = useId();

  const [name, setName] = useState(defaultName);
  const [description, setDescription] = useState("");
  const [corpusVal, setCorpusVal] = useState(defaultCorpus);
  const [pending, startTransition] = useTransition();

  // Resolve the chosen corpus's vae_type — shown read-only and sent
  // along on submit.  Server side trusts the corpus registry, not
  // this value, but we send it for symmetry + better error messages.
  const selectedCorpus = corpora.find((c) => c.value === corpusVal);
  const derivedVaeType = selectedCorpus?.vaeType ?? "token_vocab";

  const trimmed = name.trim();
  const nameError = !trimmed
    ? "Name is required."
    : taken.has(trimmed.toLowerCase())
      ? "A phrase VAE with this name already exists."
      : undefined;
  const isValid = !nameError && corpusVal;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid) return;
    startTransition(async () => {
      try {
        await createPhraseVAEFromPreset({
          presetId,
          name: trimmed,
          description: description.trim(),
          corpusId: corpusVal,
        });
        // server action redirects on success; component will unmount
      } catch (err) {
        unstable_rethrow(err);
        alert(err instanceof Error ? err.message : String(err));
      }
    });
  };

  const inputClass =
    "w-full px-3 rounded-[calc(var(--radius)*0.6)] bg-surface/60 border text-sm text-foreground focus:outline-none focus:bg-surface transition-colors";
  const okBorder = "border-separator focus:border-accent/60";

  return (
    <Modal isOpen onOpenChange={(o) => { if (!o) onClose(); }}>
      <Modal.Backdrop isDismissable>
        <Modal.Container placement="center" size="md">
          <Modal.Dialog className="rounded-[calc(var(--radius)*0.6)] w-full max-w-lg">
            <form onSubmit={onSubmit} className="flex flex-col">
            <Modal.Header className="flex flex-row flex-nowrap items-start justify-between gap-3 px-5 py-4 border-b border-separator">
              <div className="flex flex-col gap-1 flex-1 min-w-0">
                <Modal.Heading className="text-lg font-semibold">
                  Use preset
                </Modal.Heading>
                <p className="text-xs text-muted">
                  Instantiate a new phrase VAE from{" "}
                  <code className="text-foreground/90">{presetName}</code>.  The
                  preset&apos;s config is copied — edits to the preset
                  won&apos;t affect this VAE.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="shrink-0 p-1 rounded-full text-muted hover:text-foreground"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </Modal.Header>
            <Modal.Body className="flex flex-col gap-4 px-5 py-5">
              <label htmlFor={nameId} className="flex flex-col gap-1.5">
                <span className="text-sm text-foreground/90">Name</span>
                <input
                  id={nameId}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className={`h-10 ${inputClass} ${
                    nameError
                      ? "border-red-400/60 focus:border-red-400"
                      : okBorder
                  }`}
                  aria-invalid={nameError ? true : undefined}
                />
                <span className={`text-xs leading-snug ${nameError ? "text-red-400" : "text-muted"}`}>
                  {nameError ?? "Globally unique. Identifies this VAE in the index."}
                </span>
              </label>

              <label htmlFor={descId} className="flex flex-col gap-1.5">
                <span className="text-sm text-foreground/90">Description (optional)</span>
                <textarea
                  id={descId}
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Notes about this run, hypotheses, etc."
                  className={`py-2 resize-y ${inputClass} ${okBorder}`}
                />
              </label>

              <label htmlFor={corpusFieldId} className="flex flex-col gap-1.5">
                <span className="text-sm text-foreground/90">Training corpus</span>
                <select
                  id={corpusFieldId}
                  value={corpusVal}
                  onChange={(e) => setCorpusVal(e.target.value)}
                  className={`h-10 ${inputClass} ${okBorder}`}
                >
                  {corpora.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
                <span className="text-xs text-muted leading-snug">
                  Defaults to whatever the preset references.  VAE type
                  (below) is determined by the corpus you pick — every
                  corpus is intrinsically IPA or token-vocab.
                </span>
              </label>

              <div className="flex flex-col gap-1.5">
                <span className="text-sm text-foreground/90">VAE type</span>
                <div className="h-10 px-3 rounded-[calc(var(--radius)*0.6)] bg-default/30 border border-separator text-sm text-foreground/90 inline-flex items-center">
                  {VAE_TYPE_LABEL[derivedVaeType]}
                </div>
                <span className="text-xs text-muted leading-snug">
                  Auto-derived from the selected corpus.
                </span>
              </div>
            </Modal.Body>
            <Modal.Footer className="flex justify-end gap-2 px-5 py-4 border-t border-separator">
              <button
                type="button"
                onClick={onClose}
                className="h-9 px-4 rounded-[calc(var(--radius)*0.6)] text-sm text-muted hover:text-foreground border border-separator"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!isValid || pending}
                className="h-9 px-4 rounded-[calc(var(--radius)*0.6)] text-sm bg-accent text-accent-foreground font-semibold hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-1.5"
              >
                <Rocket className="w-4 h-4" />
                {pending ? "Creating…" : "Instantiate"}
              </button>
            </Modal.Footer>
            </form>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  );
}
