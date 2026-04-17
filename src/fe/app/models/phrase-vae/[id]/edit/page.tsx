import { notFound } from "next/navigation";
import { listCorpora } from "@/lib/corpora";
import { getPhraseVAE, listPhraseVAENames } from "@/lib/models/queries";
import { PhraseVAEPresetEditor } from "@/components/phrase-vae/preset-editor";
import {
  PhraseVAEConfig,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";
import { BackLink } from "@/components/back-link";

type Props = { params: Promise<{ id: string }> };

export default async function EditPhraseVAEPage({ params }: Props) {
  const { id } = await params;
  const [vae, corpora, existingNames] = await Promise.all([
    getPhraseVAE(id),
    listCorpora(),
    listPhraseVAENames(id),
  ]);
  if (!vae) notFound();

  const parsed = PhraseVAEConfig.safeParse(vae.config);
  const config: PhraseVAEConfigShape = parsed.success
    ? parsed.data
    : PhraseVAEConfig.parse({});
  const options = corpora.map((c) => ({ value: c.id, label: c.name }));

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref={`/models/phrase-vae/${id}`} label="Back to VAE" />
      <PhraseVAEPresetEditor
        target={{ kind: "vae", id: vae.id }}
        initialName={vae.name}
        initialDescription={vae.description}
        initialConfig={config}
        corpora={options}
        existingNames={existingNames}
      />
    </section>
  );
}
