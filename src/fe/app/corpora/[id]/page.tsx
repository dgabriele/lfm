import { notFound } from "next/navigation";
import { getCorpus, getSamples } from "@/lib/corpora";
import { formatNumber } from "@/lib/format";
import { SamplePager } from "@/components/sample-pager";
import { SampleText } from "@/components/sample-text";

const PAGE_SIZE = 20;

type Props = {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ page?: string }>;
};

export default async function CorpusDetailPage({ params, searchParams }: Props) {
  const { id } = await params;
  const { page: rawPage } = await searchParams;
  const corpus = await getCorpus(id);
  if (!corpus) notFound();

  const page = Math.max(1, Number(rawPage) || 1);
  const offset = (page - 1) * PAGE_SIZE;
  const { total, samples } = await getSamples(id, { offset, limit: PAGE_SIZE });
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const highlight = corpus.tags.includes("phrase-tags");

  return (
    <section className="flex-1 flex flex-col gap-4 min-h-0">
      <header className="flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">
          Samples
        </h2>
        <span className="text-xs text-muted">
          Page {formatNumber(page)} of {formatNumber(totalPages)} ·{" "}
          {formatNumber(total)} total
        </span>
      </header>

      <div className="relative flex-1 min-h-0">
        <ol className="absolute inset-0 flex flex-col gap-2 overflow-y-auto pr-2 -mr-2 pt-3 pb-10">
          {samples.map((s) => (
            <li
              key={s.index}
              className="flex gap-4 items-start rounded-[var(--radius)] border border-separator bg-surface/40 px-4 py-3"
            >
              <span className="shrink-0 text-xs text-muted tabular-nums mt-0.5 w-16">
                #{formatNumber(s.index)}
              </span>
              <SampleText text={s.text} highlight={highlight} />
            </li>
          ))}
        </ol>
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-3 bg-gradient-to-b from-[var(--background)] to-transparent"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-[var(--background)] to-transparent"
        />
      </div>

      <SamplePager
        corpusId={id}
        page={page}
        totalPages={totalPages}
      />
    </section>
  );
}
