import "server-only";
import { cache } from "react";
import { createHighlighter, type Highlighter, type ThemedToken } from "shiki";

/**
 * Shiki-powered XML tokenizer for corpus samples.  We only load the
 * `xml` grammar and a single dark theme — this keeps the first-call
 * cost low (~30ms) and the highlighter is reused across requests via
 * React's per-request cache.
 */

const getHighlighter = cache(async (): Promise<Highlighter> => {
  return createHighlighter({
    themes: ["github-dark-default"],
    langs: ["xml"],
  });
});

export type HighlightedToken = { text: string; color: string | undefined };

export async function highlightXmlLine(code: string): Promise<HighlightedToken[]> {
  const hl = await getHighlighter();
  const { tokens } = hl.codeToTokens(code, {
    lang: "xml",
    theme: "github-dark-default",
  });
  // Corpus samples are single-line; flatten just in case.
  const out: HighlightedToken[] = [];
  for (const line of tokens) {
    for (const t of line as ThemedToken[]) {
      out.push({ text: t.content, color: t.color });
    }
  }
  return out;
}
