"use client";

import { useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { produce } from "immer";
import { Save, Trash2 } from "lucide-react";
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
  createPhraseVAEModel,
  updatePhraseVAEModel,
  deletePhraseVAEModel,
} from "@/lib/models/actions";

/**
 * Client shell for the phrase-VAE config editor.  Holds form state,
 * wires up `SchemaForm` + `YamlPreview`, and calls the appropriate
 * server action on save.  Used by both `/new` and `/[id]/edit` pages;
 * the difference is whether `initialId` is present.
 */

export type CorpusOption = { value: string; label: string };

export function PhraseVAEEditor({
  initialId,
  initialName,
  initialConfig,
  corpora,
}: {
  initialId?: string;
  initialName?: string;
  initialConfig?: PhraseVAEConfigShape;
  corpora: CorpusOption[];
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  const [name, setName] = useState(initialName ?? "");
  const [config, setConfig] = useState<PhraseVAEConfigShape>(
    initialConfig ?? phraseVAEDefaults(),
  );

  const yamlText = useMemo(() => phraseVAEToYaml(config), [config]);

  const isValid = useMemo(
    () => name.trim().length > 0 && PhraseVAEConfig.safeParse(config).success,
    [name, config],
  );

  const onSave = () => {
    if (!isValid) return;
    startTransition(async () => {
      if (initialId) {
        await updatePhraseVAEModel(initialId, { name: name.trim(), config });
        router.refresh();
      } else {
        await createPhraseVAEModel({ name: name.trim(), config });
      }
    });
  };

  const onDelete = () => {
    if (!initialId) return;
    if (!confirm(`Delete config "${name}"?`)) return;
    startTransition(async () => {
      await deletePhraseVAEModel(initialId);
    });
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col gap-4">
      <header className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1 flex-1 min-w-0">
          <label className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
            Config name
          </label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. v13-english-ortho"
            className="h-10 px-3 rounded-[calc(var(--radius)*0.6)] bg-surface/60 border border-separator text-sm text-foreground focus:outline-none focus:border-accent/60 focus:bg-surface transition-colors max-w-md"
          />
        </div>
        <div className="flex items-center gap-2">
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
        <div className="min-h-0 overflow-y-auto pr-2 -mr-2 flex flex-col gap-3 pb-10">
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
