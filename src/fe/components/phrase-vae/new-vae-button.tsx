"use client";

import { useId, useMemo, useState, useTransition } from "react";
import { Modal } from "@heroui/react";
import { Boxes, X } from "lucide-react";
import { createPhraseVAEFromPreset } from "@/lib/models/actions";
import type { CorpusOption } from "./use-preset-modal";

/**
 * "New VAE" entry point on the phrase-VAE index page.  Opens a modal
 * shaped like UsePresetModal but with a preset picker at the top —
 * so the user starts from "I want a new VAE" rather than "I want to
 * use this specific preset".
 */

export type PresetOption = { value: string; label: string; corpusId: string };

const VAE_TYPE_LABEL: Record<string, string> = {
  ipa: "IPA (character / phoneme)",
  token_vocab: "Token vocab (BPE)",
};

export function NewVAEButton({
  presets,
  corpora,
  existingVaeNames,
}: {
  presets: PresetOption[];
  corpora: CorpusOption[];
  existingVaeNames: string[];
}) {
  const [open, setOpen] = useState(false);
  if (presets.length === 0) {
    return (
      <button
        type="button"
        disabled
        title="Create a config preset first"
        className="h-10 px-4 rounded-[calc(var(--radius)*0.6)] bg-default/40 text-muted text-sm font-semibold opacity-60 cursor-not-allowed flex items-center gap-2 shrink-0"
      >
        <Boxes className="w-4 h-4" strokeWidth={2.25} />
        New VAE
      </button>
    );
  }
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="h-10 px-4 rounded-[calc(var(--radius)*0.6)] bg-accent text-accent-foreground text-sm font-semibold hover:brightness-110 transition-all flex items-center gap-2 shrink-0"
      >
        <Boxes className="w-4 h-4" strokeWidth={2.25} />
        New VAE
      </button>
      {open && (
        <NewVAEModal
          presets={presets}
          corpora={corpora}
          existingVaeNames={existingVaeNames}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function NewVAEModal({
  presets,
  corpora,
  existingVaeNames,
  onClose,
}: {
  presets: PresetOption[];
  corpora: CorpusOption[];
  existingVaeNames: string[];
  onClose: () => void;
}) {
  const presetFieldId = useId();
  const nameId = useId();
  const descId = useId();
  const corpusFieldId = useId();

  const taken = useMemo(
    () => new Set(existingVaeNames.map((n) => n.trim().toLowerCase())),
    [existingVaeNames],
  );
  const suggestName = (presetName: string): string => {
    let i = 1;
    while (taken.has(`${presetName}-${i}`.toLowerCase())) i += 1;
    return `${presetName}-${i}`;
  };

  const [presetVal, setPresetVal] = useState(presets[0]!.value);
  const selectedPreset = presets.find((p) => p.value === presetVal)!;
  const [name, setName] = useState(suggestName(selectedPreset.label));
  const [description, setDescription] = useState("");
  const [corpusVal, setCorpusVal] = useState(selectedPreset.corpusId);
  const [pending, startTransition] = useTransition();

  // Auto-fill name + corpus when preset changes (only if name still
  // looks "auto" — i.e. the user hasn't manually edited).
  const onPresetChange = (next: string) => {
    setPresetVal(next);
    const p = presets.find((x) => x.value === next);
    if (!p) return;
    setName(suggestName(p.label));
    setCorpusVal(p.corpusId);
  };

  const selectedCorpus = corpora.find((c) => c.value === corpusVal);
  const derivedVaeType = selectedCorpus?.vaeType ?? "token_vocab";

  const trimmed = name.trim();
  const nameError = !trimmed
    ? "Name is required."
    : taken.has(trimmed.toLowerCase())
      ? "A phrase VAE with this name already exists."
      : undefined;
  const isValid = !nameError && presetVal && corpusVal;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid) return;
    startTransition(async () => {
      try {
        await createPhraseVAEFromPreset({
          presetId: presetVal,
          name: trimmed,
          description: description.trim(),
          corpusId: corpusVal,
        });
      } catch (err) {
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
                    New phrase VAE
                  </Modal.Heading>
                  <p className="text-xs text-muted">
                    Pick a config preset to instantiate from.  The
                    preset&apos;s config is snapshotted into the new
                    VAE — later edits to the preset don&apos;t propagate.
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
                <label htmlFor={presetFieldId} className="flex flex-col gap-1.5">
                  <span className="text-sm text-foreground/90">Config preset</span>
                  <select
                    id={presetFieldId}
                    value={presetVal}
                    onChange={(e) => onPresetChange(e.target.value)}
                    className={`h-10 ${inputClass} ${okBorder}`}
                  >
                    {presets.map((p) => (
                      <option key={p.value} value={p.value}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </label>

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
                    (below) is determined by the corpus you pick.
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
                  <Boxes className="w-4 h-4" />
                  {pending ? "Creating…" : "Create VAE"}
                </button>
              </Modal.Footer>
            </form>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  );
}
