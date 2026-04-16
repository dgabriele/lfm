import Link from "next/link";
import { notFound } from "next/navigation";
import { Boxes, FileCog, Database, Cpu } from "lucide-react";
import {
  getPhraseVAE,
  getPhraseVAEConfigPreset,
} from "@/lib/models/queries";
import {
  PhraseVAEConfig,
  type PhraseVAEConfigShape,
} from "@/lib/config-schemas/phrase-vae";
import { phraseVAEToYaml } from "@/lib/config-schemas/phrase-vae-yaml";
import { YamlPreview } from "@/components/form/yaml-preview";
import { TimestampedMeta } from "@/components/phrase-vae/timestamped-meta";
import { BackLink } from "@/components/back-link";

type Props = { params: Promise<{ id: string }> };

const VAE_TYPE_LABEL: Record<string, string> = {
  ipa: "IPA (character / phoneme)",
  token_vocab: "Token vocab (BPE)",
};

export default async function PhraseVAEDetailPage({ params }: Props) {
  const { id } = await params;
  const vae = await getPhraseVAE(id);
  if (!vae) notFound();

  const preset = vae.presetId
    ? await getPhraseVAEConfigPreset(vae.presetId)
    : null;

  const parsed = PhraseVAEConfig.safeParse(vae.config);
  const config: PhraseVAEConfigShape = parsed.success
    ? parsed.data
    : PhraseVAEConfig.parse({});
  const yamlText = phraseVAEToYaml(config);

  return (
    <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
      <BackLink fallbackHref="/models/phrase-vae" label="Back to phrase VAEs" />
      <header className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-2 flex-1 min-w-0">
          <div className="flex items-center gap-3">
            <Boxes className="w-7 h-7 text-accent" strokeWidth={2} />
            <h1 className="text-2xl font-semibold tracking-tight truncate">
              {vae.name}
            </h1>
            <span className="px-2 py-0.5 text-xs rounded-[calc(var(--radius)*0.5)] bg-default/40 text-foreground/80">
              {vae.status}
            </span>
          </div>
          {vae.description && (
            <p className="text-sm text-muted leading-relaxed max-w-5xl">
              {vae.description}
            </p>
          )}
          <TimestampedMeta
            updatedAt={vae.updatedAt.toISOString()}
            description={null}
          />
        </div>
      </header>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)] gap-4">
        <div className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-2 -mr-2">
          <Section label="Origin">
            <Row
              label="Source preset"
              value={
                preset ? (
                  <Link
                    href={`/models/phrase-vae/presets/${preset.id}/edit`}
                    className="inline-flex items-center gap-1.5 text-foreground/90 hover:text-accent"
                  >
                    <FileCog className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
                    {preset.name}
                  </Link>
                ) : (
                  <span className="text-muted italic text-sm">deleted</span>
                )
              }
            />
            <Row
              label="Training corpus"
              value={
                <span className="inline-flex items-center gap-1.5 font-mono text-xs">
                  <Database className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
                  {vae.corpusId}
                </span>
              }
            />
            <Row
              label="VAE type"
              value={
                <span className="inline-flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
                  {VAE_TYPE_LABEL[vae.vaeType] ?? vae.vaeType}
                </span>
              }
            />
          </Section>
          <Section label="Lifecycle">
            <p className="text-xs text-muted leading-relaxed">
              Training launch, monitoring, and analysis tooling will live
              here.  For now, the snapshotted config is on the right —
              hand-edit and re-launch via{" "}
              <code className="text-foreground/90">lfm pretrain</code>{" "}
              against the YAML download.
            </p>
          </Section>
        </div>
        <div className="min-h-0 lg:sticky lg:top-0">
          <YamlPreview yaml={yamlText} />
        </div>
      </div>
    </section>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-3 rounded-[calc(var(--radius)*0.7)] border border-separator bg-surface/40 p-4">
      <h2 className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
        {label}
      </h2>
      <div className="flex flex-col gap-2.5">{children}</div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm">
      <span className="text-muted">{label}</span>
      <span className="text-right text-foreground/90">{value}</span>
    </div>
  );
}
