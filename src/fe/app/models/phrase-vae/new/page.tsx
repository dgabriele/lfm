import { listCorpora } from "@/lib/corpora";
import { PhraseVAEEditor } from "@/components/phrase-vae/editor";
import { BackLink } from "@/components/back-link";

export const metadata = {
  title: "New phrase VAE config — LFM",
};

export default async function NewPhraseVAEConfigPage() {
  const corpora = await listCorpora();
  const options = corpora.map((c) => ({ value: c.id, label: c.name }));

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref="/models/phrase-vae" label="Back to configs" />
      <PhraseVAEEditor corpora={options} />
    </section>
  );
}
