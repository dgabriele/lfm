import Link from "next/link";

export default function CorpusNotFound() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 p-10">
      <h2 className="text-xl font-semibold">Corpus not found</h2>
      <p className="text-sm text-muted">
        The corpus you requested doesn&apos;t exist.
      </p>
      <Link
        href="/corpora"
        className="text-sm underline underline-offset-4 hover:text-foreground"
      >
        Back to corpora
      </Link>
    </div>
  );
}
