import {
  listPhraseVAEConfigPresets,
  listPhraseVAEs,
  listPhraseVAENames,
} from "@/lib/models/queries";
import { listCorpora } from "@/lib/corpora";
import { PresetsSidebar } from "@/components/phrase-vae/presets-sidebar";
import { VaesTable } from "@/components/phrase-vae/vaes-table";
import { NewVAEButton } from "@/components/phrase-vae/new-vae-button";
import type {
  PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";
import type { PhraseVaeConfigPresetRow } from "@/lib/db/schema";

export const metadata = {
  title: "Phrase VAEs — LFM",
};

export default async function PhraseVAEIndexPage() {
  const [vaes, presets, corpora, existingVaeNames] = await Promise.all([
    listPhraseVAEs(),
    listPhraseVAEConfigPresets(),
    listCorpora(),
    listPhraseVAENames(),
  ]);

  const presetsById: Record<string, PhraseVaeConfigPresetRow> = {};
  for (const p of presets) presetsById[p.id] = p;
  const corpusOptions = corpora.map((c) => ({
    value: c.id,
    label: c.name,
    vaeType: c.vaeType,
  }));
  const presetOptions = presets.map((p) => ({
    value: p.id,
    label: p.name,
    corpusId: (p.config as PhraseVAEConfigShape).corpus_id,
  }));

  return (
    <div className="flex flex-1 min-h-0">
      <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
        <header className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <h1 className="text-2xl font-semibold tracking-tight">
              Phrase VAEs
            </h1>
            <p className="text-sm text-muted max-w-2xl">
              Models that turn continuous representations into
              variable-length linguistic surface forms.  Each VAE is
              trained on a corpus and produces a frozen decoder that
              downstream agents and translators read from.
            </p>
          </div>
          <NewVAEButton
            presets={presetOptions}
            corpora={corpusOptions}
            existingVaeNames={existingVaeNames}
          />
        </header>
        <VaesTable vaes={vaes} presetsById={presetsById} />
      </section>
      <PresetsSidebar
        presets={presets}
        corpora={corpusOptions}
        existingVaeNames={existingVaeNames}
      />
    </div>
  );
}
