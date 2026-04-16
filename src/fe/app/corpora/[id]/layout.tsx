import type { ReactNode } from "react";
import { notFound } from "next/navigation";
import { getCorpus } from "@/lib/corpora";
import { CorpusMetadataSidebar } from "@/components/corpus-metadata-sidebar";
import { BackLink } from "@/components/back-link";

type Props = {
  children: ReactNode;
  params: Promise<{ id: string }>;
};

/**
 * Detail-page layout: main content on the left, metadata rail on the
 * right (opposite the nav rail).  Both the layout and the
 * `CorpusMetadataSidebar` server-render from the corpus record, so
 * no client roundtrip is needed for the metadata panel.
 */
export default async function CorpusDetailLayout({ children, params }: Props) {
  const { id } = await params;
  const corpus = await getCorpus(id);
  if (!corpus) notFound();

  return (
    <div className="flex flex-1 min-h-0">
      <section className="flex-1 min-w-0 min-h-0 flex flex-col p-10 gap-6 overflow-hidden">
        <BackLink />
        <header className="flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">
            {corpus.name}
          </h1>
          <p className="text-sm text-muted">{corpus.description}</p>
        </header>
        {children}
      </section>
      <CorpusMetadataSidebar corpus={corpus} />
    </div>
  );
}
