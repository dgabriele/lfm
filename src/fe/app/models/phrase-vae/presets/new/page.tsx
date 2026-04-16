import { listCorpora } from "@/lib/corpora";
import { PhraseVAEPresetEditor } from "@/components/phrase-vae/preset-editor";
import { BackLink } from "@/components/back-link";
import { listPhraseVAEConfigPresetNames } from "@/lib/models/queries";

export const metadata = {
  title: "New phrase VAE config preset — LFM",
};

export default async function NewPhraseVAEPresetPage() {
  const [corpora, existingNames] = await Promise.all([
    listCorpora(),
    listPhraseVAEConfigPresetNames(),
  ]);
  const options = corpora.map((c) => ({ value: c.id, label: c.name }));

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref="/models/phrase-vae" label="Back to phrase VAEs" />
      <PhraseVAEPresetEditor corpora={options} existingNames={existingNames} />
    </section>
  );
}
