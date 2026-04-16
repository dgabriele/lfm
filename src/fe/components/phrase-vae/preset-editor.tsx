"use client";

import { useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { produce } from "immer";
import { Save, Trash2, ChevronDown } from "lucide-react";
import { Disclosure, DisclosureGroup } from "@heroui/react";
import {
  PhraseVAEConfig,
  PHRASE_VAE_SECTIONS,
  phraseVAEDefaults,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";
import { phraseVAEToYaml } from "@/lib/config-schemas/phrase-vae-yaml";
import { SchemaForm } from "@/components/form/schema-form";
import { YamlPreview } from "@/components/form/yaml-preview";
import {
  createPhraseVAEConfigPreset,
  updatePhraseVAEConfigPreset,
  deletePhraseVAEConfigPreset,
} from "@/lib/models/actions";

/**
 * Client shell for the phrase-VAE *config preset* editor.  Holds form
 * state, wires up `SchemaForm` + `YamlPreview`, and calls the
 * appropriate server action on save.  A preset is a reusable template;
 * instantiating it as an actual PhraseVAE happens via the "Use" button
 * on the presets sidebar (snapshots the config into the new VAE).
 */

export type CorpusOption = { value: string; label: string };

/**
 * Identity section — config `name` + `description`, laid out as an
 * accordion block to match the rest of the editor.  Purposely separate
 * from `SchemaForm` because these fields live on the model row itself
 * (not in the config JSON), but visually identical so the page reads
 * as one continuous stack of sections.
 */
function IdentitySection({
  name,
  description,
  nameError,
  onNameChange,
  onDescriptionChange,
}: {
  name: string;
  description: string;
  nameError?: string;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
}) {
  const inputClass =
    "w-full px-3 rounded-[calc(var(--radius)*0.6)] bg-surface/60 border text-sm text-foreground focus:outline-none focus:bg-surface transition-colors";
  const nameBorder = nameError
    ? "border-red-400/60 focus:border-red-400"
    : "border-separator focus:border-accent/60";
  return (
    <DisclosureGroup defaultExpandedKeys={["identity"]}>
      <Disclosure
        id="identity"
        className="rounded-[calc(var(--radius)*0.7)] border border-separator bg-surface/40"
      >
        <Disclosure.Heading>
          <Disclosure.Trigger className="w-full flex items-start justify-between gap-3 px-4 py-3 text-left hover:bg-default/20 transition-colors">
            <span className="flex flex-col gap-1 flex-1 min-w-0">
              <span className="uppercase tracking-wider text-xs font-semibold text-accent/90">
                Identity
              </span>
              <span className="text-xs text-muted leading-snug font-normal normal-case tracking-normal">
                A short name and description to distinguish this config
                from others.
              </span>
            </span>
            <Disclosure.Indicator>
              <ChevronDown
                className="w-4 h-4 mt-0.5 text-muted transition-transform data-[expanded]:rotate-180"
                strokeWidth={2}
              />
            </Disclosure.Indicator>
          </Disclosure.Trigger>
        </Disclosure.Heading>
        <Disclosure.Content>
          <Disclosure.Body className="flex flex-col gap-4 px-4 pb-4 pt-1">
            <label className="flex flex-col gap-1.5">
              <span className="text-sm text-foreground/90">Name</span>
              <input
                value={name}
                onChange={(e) => onNameChange(e.target.value)}
                placeholder="e.g. v13-english-ortho"
                className={`h-10 ${inputClass} ${nameBorder}`}
                aria-invalid={nameError ? true : undefined}
              />
              {nameError ? (
                <span className="text-xs text-red-400 leading-snug">
                  {nameError}
                </span>
              ) : (
                <span className="text-xs text-muted leading-snug">
                  Must be globally unique.  Shown as the title at the top
                  of this page and in the Recent Configs list.
                </span>
              )}
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-sm text-foreground/90">Description</span>
              <textarea
                value={description}
                onChange={(e) => onDescriptionChange(e.target.value)}
                placeholder="What sets this config apart? Corpus, regularization, objective tweaks, etc."
                rows={4}
                className={`py-2 resize-y ${inputClass} border-separator focus:border-accent/60`}
              />
              <span className="text-xs text-muted leading-snug">
                A couple of sentences explaining what makes this config
                different.  Appears truncated in Recent Configs and in
                full on hover.
              </span>
            </label>
          </Disclosure.Body>
        </Disclosure.Content>
      </Disclosure>
    </DisclosureGroup>
  );
}

export function PhraseVAEPresetEditor({
  initialId,
  initialName,
  initialDescription,
  initialConfig,
  corpora,
  existingNames,
}: {
  initialId?: string;
  initialName?: string;
  initialDescription?: string | null;
  initialConfig?: PhraseVAEConfigShape;
  corpora: CorpusOption[];
  existingNames: string[];
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  const [name, setName] = useState(initialName ?? "");
  const [description, setDescription] = useState(initialDescription ?? "");
  const [config, setConfig] = useState<PhraseVAEConfigShape>(
    initialConfig ?? phraseVAEDefaults(),
  );

  const yamlText = useMemo(() => phraseVAEToYaml(config), [config]);

  const existingSet = useMemo(
    () => new Set(existingNames.map((n) => n.trim().toLowerCase())),
    [existingNames],
  );
  const trimmedName = name.trim();
  const nameTaken = trimmedName.length > 0 && existingSet.has(trimmedName.toLowerCase());
  const nameError = !trimmedName
    ? "Name is required."
    : nameTaken
      ? "A config with this name already exists."
      : undefined;

  const isValid = useMemo(
    () => !nameError && PhraseVAEConfig.safeParse(config).success,
    [nameError, config],
  );

  const onSave = () => {
    if (!isValid) return;
    const payload = {
      name: name.trim(),
      description: description.trim(),
      config,
    };
    startTransition(async () => {
      if (initialId) {
        await updatePhraseVAEConfigPreset(initialId, payload);
        router.refresh();
      } else {
        await createPhraseVAEConfigPreset(payload);
      }
    });
  };

  const onDelete = () => {
    if (!initialId) return;
    if (!confirm(`Delete preset "${name}"?`)) return;
    startTransition(async () => {
      await deletePhraseVAEConfigPreset(initialId);
    });
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-4">
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-2 flex-1 min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">
            {name.trim() || "Untitled preset"}
          </h1>
          {description.trim() && (
            <p
              className="text-sm text-muted leading-relaxed max-w-5xl overflow-hidden"
              style={{
                display: "-webkit-box",
                WebkitLineClamp: 4,
                WebkitBoxOrient: "vertical",
              }}
            >
              {description}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {initialId && (
            <button
              type="button"
              onClick={onDelete}
              disabled={pending}
              className="h-9 px-3 rounded-[calc(var(--radius)*0.6)] text-sm text-muted hover:text-red-400 border border-separator hover:border-red-400/40 transition-colors flex items-center gap-1.5"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
          )}
          <button
            type="button"
            onClick={onSave}
            disabled={!isValid || pending}
            className="h-9 px-4 rounded-[calc(var(--radius)*0.6)] text-sm bg-accent text-accent-foreground font-semibold hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-1.5"
          >
            <Save className="w-4 h-4" />
            {pending ? "Saving…" : initialId ? "Save" : "Create"}
          </button>
        </div>
      </header>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)] gap-4">
        <div className="min-h-0 overflow-y-auto pr-2 -mr-2 flex flex-col gap-2 pb-10">
          <IdentitySection
            name={name}
            description={description}
            nameError={name.length > 0 ? nameError : undefined}
            onNameChange={setName}
            onDescriptionChange={setDescription}
          />
          <SchemaForm
            schema={PhraseVAEConfig}
            sections={PHRASE_VAE_SECTIONS}
            value={config as unknown as Record<string, unknown>}
            onChange={(next) =>
              setConfig(
                produce(config, (draft) => {
                  Object.assign(draft, next);
                }),
              )
            }
            optionProviders={{ corpora }}
          />
        </div>
        <div className="min-h-0 lg:sticky lg:top-0">
          <YamlPreview yaml={yamlText} />
        </div>
      </div>
    </div>
  );
}
