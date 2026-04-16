"use client";

import { useEffect, useState } from "react";
import { createHighlighter, type Highlighter } from "shiki";

/**
 * Client-side YAML preview with Shiki syntax highlighting.  Keeps a
 * single highlighter per tab (lazy-init) and re-renders on `yaml`
 * change.  Rendering is ~1ms per KB; we skip debouncing for now.
 */

let highlighterPromise: Promise<Highlighter> | null = null;
function getHighlighter(): Promise<Highlighter> {
  highlighterPromise ??= createHighlighter({
    themes: ["github-dark-default"],
    langs: ["yaml"],
  });
  return highlighterPromise;
}

export function YamlPreview({ yaml }: { yaml: string }) {
  const [html, setHtml] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    getHighlighter().then((hl) => {
      if (cancelled) return;
      const out = hl.codeToHtml(yaml, {
        lang: "yaml",
        theme: "github-dark-default",
      });
      setHtml(out);
    });
    return () => {
      cancelled = true;
    };
  }, [yaml]);

  return (
    <div className="relative h-full flex flex-col rounded-[calc(var(--radius)*0.7)] border border-separator bg-surface/40 overflow-hidden">
      <header className="flex items-center justify-between px-4 py-2 border-b border-separator">
        <h3 className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
          YAML preview
        </h3>
        <CopyButton text={yaml} />
      </header>
      <div
        className="flex-1 overflow-auto text-xs [&_pre]:!bg-transparent [&_pre]:m-0 [&_pre]:p-4"
        // Shiki's HTML is trusted server-generated output.
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        } catch {
          /* ignore */
        }
      }}
      className="text-xs text-muted hover:text-accent transition-colors"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}
