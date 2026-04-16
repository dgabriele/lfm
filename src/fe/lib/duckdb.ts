import path from "node:path";
import { DuckDBInstance, type DuckDBConnection } from "@duckdb/node-api";

/**
 * Singleton DuckDB connection.  The corpora database is opened in
 * read-only mode — the front-end never writes; all ingestion happens
 * via `scripts/etl_corpora_duckdb.py`.  In dev, Next hot-reload
 * re-evaluates this module, so we stash the instance on `globalThis`
 * to avoid leaking connections across HMR reloads.
 */

const DB_PATH =
  process.env.LFM_CORPORA_DB ??
  path.resolve(process.cwd(), "..", "..", "data", "corpora.duckdb");

type Cache = { instance?: Promise<DuckDBInstance>; conn?: Promise<DuckDBConnection> };
const g = globalThis as unknown as { __lfmDuck?: Cache };
const cache: Cache = (g.__lfmDuck ??= {});

export async function corpusDb(): Promise<DuckDBConnection> {
  if (!cache.conn) {
    cache.instance ??= DuckDBInstance.create(DB_PATH, { access_mode: "read_only" });
    cache.conn = cache.instance.then((i) => i.connect());
  }
  return cache.conn;
}

export function corpusDbDir(): string {
  return path.dirname(DB_PATH);
}
