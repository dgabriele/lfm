import { highlightXmlLine } from "@/lib/highlight";

/**
 * Render a corpus sample.  If `highlight` is true, tokenize as XML
 * with Shiki so phrase-type tags (`<NP>`, `<VP>`, ...) get colored;
 * otherwise render as plain monospace text.
 */
export async function SampleText({
  text,
  highlight,
}: {
  text: string;
  highlight: boolean;
}) {
  if (!highlight) {
    return (
      <pre className="flex-1 min-w-0 text-sm whitespace-pre-wrap break-words font-mono text-foreground/90">
        {text}
      </pre>
    );
  }
  const tokens = await highlightXmlLine(text);
  return (
    <pre className="flex-1 min-w-0 text-sm whitespace-pre-wrap break-words font-mono">
      {tokens.map((t, i) => (
        <span key={i} style={t.color ? { color: t.color } : undefined}>
          {t.text}
        </span>
      ))}
    </pre>
  );
}
