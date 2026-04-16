import { listCorpora } from "@/lib/corpora";
import { PhraseVAEEditor } from "@/components/phrase-vae/editor";
import { BackLink } from "@/components/back-link";
import { listVaeModelNames } from "@/lib/models/queries";

export const metadata = {
  title: "New phrase VAE config — LFM",
};

export default async function NewPhraseVAEConfigPage() {
  const [corpora, existingNames] = await Promise.all([
    listCorpora(),
    listVaeModelNames(),
  ]);
  const options = corpora.map((c) => ({ value: c.id, label: c.name }));

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref="/models/phrase-vae" label="Back to configs" />
      <PhraseVAEEditor corpora={options} existingNames={existingNames} />
    </section>
  );
}
