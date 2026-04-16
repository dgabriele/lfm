import Link from "next/link";
import { Plus } from "lucide-react";
import {
  listPhraseVAEConfigPresets,
  listPhraseVAEs,
  listPhraseVAENames,
} from "@/lib/models/queries";
import { listCorpora } from "@/lib/corpora";
import { PresetsSidebar } from "@/components/phrase-vae/presets-sidebar";
import { VaesTable } from "@/components/phrase-vae/vaes-table";
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

  return (
    <div className="flex flex-1 min-h-0">
      <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
        <header className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <h1 className="text-2xl font-semibold tracking-tight">
              Phrase VAEs
            </h1>
            <p className="text-sm text-muted">
              Instantiated phrase-VAE models — each rooted in a config
              preset that was snapshotted at instantiation time.
            </p>
          </div>
          <Link
            href="/models/phrase-vae/presets/new"
            className="h-10 px-4 rounded-[calc(var(--radius)*0.6)] bg-accent text-accent-foreground text-sm font-semibold hover:brightness-110 transition-all flex items-center gap-2 shrink-0"
          >
            <Plus className="w-4 h-4" strokeWidth={2.25} />
            New preset
          </Link>
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
