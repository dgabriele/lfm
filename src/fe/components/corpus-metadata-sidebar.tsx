import type { Corpus } from "@/lib/corpora";
import { formatNumber } from "@/lib/format";
import { InfoHint } from "@/components/info-hint";

/**
 * Right-side metadata rail for the corpus detail view.  Pure server
 * component; renders the full record as labeled key/value rows.
 */
export function CorpusMetadataSidebar({ corpus }: { corpus: Corpus }) {
  return (
    <aside className="w-80 shrink-0 border-l border-separator bg-surface/40 p-6 flex flex-col gap-6 overflow-y-auto">
      <header>
        <h2 className="text-sm uppercase tracking-wider text-muted font-semibold">
          Metadata
        </h2>
      </header>

      <Section label="Identity">
        <Row label="ID" value={<code className="text-xs">{corpus.id}</code>} />
        <Row label="Name" value={corpus.name} />
      </Section>

      <Section label="Scale">
        <Row
          label="Samples"
          value={formatNumber(corpus.sampleCount)}
          hint="Total number of rows in the corpus — one row per extracted constituent (sentence, noun phrase, verb phrase, etc.) after cleaning and deduplication."
        />
        {corpus.vocabSize && (
          <Row
            label="Vocab size"
            value={formatNumber(corpus.vocabSize)}
            hint="Number of unique subword tokens in the SentencePiece tokenizer trained on this corpus. Phrase-type tags (<NP>, <VP>, ...) are included as user-defined symbols."
          />
        )}
        {corpus.maxSeqLen && (
          <Row
            label="Max seq len"
            value={formatNumber(corpus.maxSeqLen)}
            hint="Longest sample in tokens (after SPM encoding). Sets the upper bound of the decoder's positional embedding table — training batches are bucketed up to this length."
          />
        )}
      </Section>

      {corpus.lengthStats && (
        <Section label="Length stats (chars)">
          <Row label="Min" value={formatNumber(corpus.lengthStats.min)} />
          <Row label="Max" value={formatNumber(corpus.lengthStats.max)} />
          <Row label="Mean" value={corpus.lengthStats.mean.toFixed(1)} />
          <Row label="p99" value={formatNumber(corpus.lengthStats.p99)} />
        </Section>
      )}

      {corpus.tags.length > 0 && (
        <Section label="Tags">
          <div className="flex flex-wrap gap-2">
            {corpus.tags.map((t) => (
              <span
                key={t}
                className="px-2 py-0.5 text-xs rounded-[calc(var(--radius)*0.6)] bg-default/70 text-foreground/80"
              >
                {t}
              </span>
            ))}
          </div>
        </Section>
      )}

      {corpus.sourcePaths && corpus.sourcePaths.length > 0 && (
        <Section label="Files">
          <ul className="flex flex-col gap-1">
            {corpus.sourcePaths.map((p) => (
              <li key={p}>
                <code className="text-xs text-foreground/80 break-all">
                  {p}
                </code>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </aside>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-xs uppercase tracking-wider text-accent/80 font-semibold">{label}</h3>
      <div className="flex flex-col gap-1.5">{children}</div>
    </section>
  );
}

function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-sm">
      <span className="text-muted flex items-center gap-1.5">
        {label}
        {hint && <InfoHint label={label}>{hint}</InfoHint>}
      </span>
      <span className="text-right text-foreground/90">{value}</span>
    </div>
  );
}
