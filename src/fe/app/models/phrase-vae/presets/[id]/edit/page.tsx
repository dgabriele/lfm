import { notFound } from "next/navigation";
import { listCorpora } from "@/lib/corpora";
import {
  getPhraseVAEConfigPreset,
  listPhraseVAEConfigPresetNames,
} from "@/lib/models/queries";
import { PhraseVAEPresetEditor } from "@/components/phrase-vae/preset-editor";
import {
  PhraseVAEConfig,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";
import { BackLink } from "@/components/back-link";

type Props = { params: Promise<{ id: string }> };

export default async function EditPhraseVAEPresetPage({ params }: Props) {
  const { id } = await params;
  const [row, corpora, existingNames] = await Promise.all([
    getPhraseVAEConfigPreset(id),
    listCorpora(),
    listPhraseVAEConfigPresetNames(id),
  ]);
  if (!row || row.variant !== "phrase-vae") notFound();

  const parsed = PhraseVAEConfig.safeParse(row.config);
  const config: PhraseVAEConfigShape = parsed.success
    ? parsed.data
    : PhraseVAEConfig.parse({});
  const options = corpora.map((c) => ({ value: c.id, label: c.name }));

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref="/models/phrase-vae" label="Back to phrase VAEs" />
      <PhraseVAEPresetEditor
        initialId={row.id}
        initialName={row.name}
        initialDescription={row.description}
        initialConfig={config}
        corpora={options}
        existingNames={existingNames}
      />
    </section>
  );
}
