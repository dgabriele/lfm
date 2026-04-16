import path from "node:path";
import { cache } from "react";
import { corpusDb, corpusDbDir } from "./duckdb";

/**
 * Corpus data access.  Reads from `data/corpora.duckdb` (metadata) and
 * from each corpus's own `samples.parquet` (rows).  The duckdb file is
 * built and updated by `scripts/etl_corpora_duckdb.py`.
 *
 * Pagination is keyset on the monotonic `index` column so parquet
 * row-group statistics can skip everything outside the requested
 * window — no full scans, no growing OFFSET cost.
 */

export type Corpus = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  sampleCount: number;
  vocabSize?: number;
  maxSeqLen?: number;
  lengthStats?: { min: number; max: number; mean: number; p50: number; p99: number };
  sourcePaths?: string[];
};

export type Sample = {
  index: number;
  text: string;
  language?: string;
  source?: string;
  ipa?: string;
  length?: number;
};

type CorpusRow = {
  id: string;
  name: string;
  description: string | null;
  tags: string[] | null;
  source_paths: string[] | null;
  parquet_path: string;
  sample_count: bigint | number | null;
  vocab_size: number | null;
  max_seq_len: number | null;
  length_min: number | null;
  length_max: number | null;
  length_mean: number | null;
  length_p50: number | null;
  length_p99: number | null;
};

function toNum(v: bigint | number | null | undefined): number {
  if (v == null) return 0;
  return typeof v === "bigint" ? Number(v) : v;
}

function rowToCorpus(r: CorpusRow): Corpus {
  const c: Corpus = {
    id: r.id,
    name: r.name,
    description: r.description ?? "",
    tags: r.tags ?? [],
    sampleCount: toNum(r.sample_count),
    sourcePaths: r.source_paths ?? [],
  };
  if (r.vocab_size != null) c.vocabSize = r.vocab_size;
  if (r.max_seq_len != null) c.maxSeqLen = r.max_seq_len;
  if (
    r.length_min != null &&
    r.length_max != null &&
    r.length_mean != null &&
    r.length_p50 != null &&
    r.length_p99 != null
  ) {
    c.lengthStats = {
      min: r.length_min,
      max: r.length_max,
      mean: r.length_mean,
      p50: r.length_p50,
      p99: r.length_p99,
    };
  }
  return c;
}

export const listCorpora = cache(async (): Promise<Corpus[]> => {
  const db = await corpusDb();
  const reader = await db.runAndReadAll(
    "SELECT * FROM corpora ORDER BY id DESC",
  );
  const rows = reader.getRowObjectsJson() as unknown as CorpusRow[];
  return rows.map(rowToCorpus);
});

export const getCorpus = cache(async (id: string): Promise<Corpus | null> => {
  const db = await corpusDb();
  const reader = await db.runAndReadAll(
    "SELECT * FROM corpora WHERE id = $id",
    { id },
  );
  const rows = reader.getRowObjectsJson() as unknown as CorpusRow[];
  return rows[0] ? rowToCorpus(rows[0]) : null;
});

export async function getSamples(
  id: string,
  { offset, limit }: { offset: number; limit: number },
): Promise<{ total: number; samples: Sample[] }> {
  const corpus = await getCorpus(id);
  if (!corpus) return { total: 0, samples: [] };

  const db = await corpusDb();
  const lookup = await db.runAndReadAll(
    "SELECT parquet_path FROM corpora WHERE id = $id",
    { id },
  );
  const parquetRel = (lookup.getRowsJson() as unknown as string[][])[0]?.[0];
  if (!parquetRel) return { total: corpus.sampleCount, samples: [] };
  const parquetPath = path.isAbsolute(parquetRel)
    ? parquetRel
    : path.resolve(corpusDbDir(), parquetRel);

  const start = offset;
  const end = offset + limit;
  const reader = await db.runAndReadAll(
    `
    SELECT index, text, language, source, length
    FROM read_parquet($path)
    WHERE index >= $start AND index < $end
    ORDER BY index
    `,
    { path: parquetPath, start, end },
  );
  const rows = reader.getRowObjectsJson() as unknown as Array<{
    index: bigint | number;
    text: string;
    language: string | null;
    source: string | null;
    length: number | null;
  }>;
  const samples: Sample[] = rows.map((r) => ({
    index: toNum(r.index),
    text: r.text,
    ...(r.language ? { language: r.language } : {}),
    ...(r.source ? { source: r.source } : {}),
    ...(r.length != null ? { length: r.length } : {}),
  }));
  return { total: corpus.sampleCount, samples };
}
