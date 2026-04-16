import Link from "next/link";
import { Plus, Sparkles } from "lucide-react";
import { listPhraseVAEModels } from "@/lib/models/queries";
import { ConfigsSidebar } from "@/components/phrase-vae/configs-sidebar";

export const metadata = {
  title: "Phrase VAE — LFM",
};

export default async function PhraseVAEIndexPage() {
  const configs = await listPhraseVAEModels();

  return (
    <div className="flex flex-1 min-h-0">
      <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
        <header className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-1">
            <h1 className="text-2xl font-semibold tracking-tight">
              Phrase VAE
            </h1>
            <p className="text-sm text-muted">
              Configure the frozen phrase-decoder VAE that turns continuous
              representations into variable-length linguistic surface forms.
            </p>
          </div>
          <Link
            href="/models/phrase-vae/new"
            className="h-10 px-4 rounded-[calc(var(--radius)*0.6)] bg-accent text-accent-foreground text-sm font-semibold hover:brightness-110 transition-all flex items-center gap-2 shrink-0"
          >
            <Plus className="w-4 h-4" strokeWidth={2.25} />
            New config
          </Link>
        </header>

        {configs.length === 0 ? (
          <div className="flex-1 min-h-0 flex flex-col items-center justify-center gap-4 text-center">
            <Sparkles className="w-10 h-10 text-accent/70" strokeWidth={1.5} />
            <h2 className="text-lg font-semibold">
              No configs yet
            </h2>
            <p className="text-sm text-muted max-w-md leading-relaxed">
              A phrase-VAE config captures everything the training loop
              needs — dataset, tokenizer, architecture, objectives,
              schedule.  Start with a fresh config or seed the v13 preset.
            </p>
            <Link
              href="/models/phrase-vae/new"
              className="mt-2 h-10 px-5 rounded-[calc(var(--radius)*0.6)] bg-accent text-accent-foreground text-sm font-semibold hover:brightness-110 transition-all flex items-center gap-2"
            >
              <Plus className="w-4 h-4" strokeWidth={2.25} />
              Create first config
            </Link>
          </div>
        ) : (
          <div className="flex-1 min-h-0 flex flex-col gap-3 text-sm text-muted leading-relaxed max-w-prose">
            <p>
              Configs are listed in the right rail, most recently edited
              first.  <strong className="text-foreground/90">Edit</strong> opens
              the editor; <strong className="text-foreground/90">Use</strong>{" "}
              downloads the YAML for immediate consumption by the training
              pipeline.
            </p>
          </div>
        )}
      </section>
      <ConfigsSidebar configs={configs} />
    </div>
  );
}
