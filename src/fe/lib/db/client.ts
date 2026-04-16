import "server-only";
import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import * as schema from "./schema";

/**
 * Singleton postgres + drizzle connection.  Next dev HMR re-evaluates
 * this module, so the client is stashed on `globalThis` to avoid
 * leaking connection pools across reloads.
 */

const DATABASE_URL =
  process.env.DATABASE_URL ?? "postgres://lfm:lfm@localhost:5432/lfm";

type Cache = {
  sql?: ReturnType<typeof postgres>;
  db?: ReturnType<typeof drizzle<typeof schema>>;
};
const g = globalThis as unknown as { __lfmPg?: Cache };
const cache: Cache = (g.__lfmPg ??= {});

cache.sql ??= postgres(DATABASE_URL, { max: 4 });
cache.db ??= drizzle(cache.sql, { schema });

export const db = cache.db;
export { schema };
