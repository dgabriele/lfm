import Link from "next/link";
import { Pencil, FileCog } from "lucide-react";
import type { VaeModelRow } from "@/lib/db/schema";
import { phraseVAEToYaml } from "@/lib/config-schemas/phrase-vae-yaml";
import type { PhraseVAEConfigShape } from "@/lib/config-schemas/phrase-vae";
import { UseButton } from "./use-button";
import { TimestampedMeta } from "./timestamped-meta";

/**
 * Right-side rail listing phrase-VAE configs in most-recently-updated
 * order.  Each entry gets two quick actions: Edit (→ editor) and Use
 * (→ download the YAML for immediate consumption by training tools).
 */

export function ConfigsSidebar({ configs }: { configs: VaeModelRow[] }) {
  return (
    <aside className="w-96 shrink-0 border-l border-separator bg-surface/40 p-6 flex flex-col gap-4 overflow-y-auto">
      <header>
        <h2 className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
          Recent configs
        </h2>
        <p className="text-xs text-muted mt-1">
          Most recently edited first.
        </p>
      </header>

      {configs.length === 0 ? (
        <p className="text-sm text-muted leading-relaxed">
          No configs yet.  Create one to get started.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {configs.map((c) => (
            <li
              key={c.id}
              className="flex flex-col gap-1 rounded-[calc(var(--radius)*0.6)] px-2 py-1.5 hover:bg-default/40 transition-colors"
            >
              <div className="flex items-center gap-2">
                <FileCog
                  className="w-3.5 h-3.5 shrink-0 text-accent/70"
                  strokeWidth={2}
                  aria-hidden
                />
                <span className="flex-1 min-w-0 text-sm text-foreground/90 truncate">
                  {c.name}
                </span>
                <Link
                  href={`/models/phrase-vae/${c.id}/edit`}
                  aria-label={`Edit ${c.name}`}
                  className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-accent hover:bg-accent/10 transition-colors"
                >
                  <Pencil className="w-3.5 h-3.5" strokeWidth={2} />
                </Link>
                <UseButton
                  name={c.name}
                  yaml={phraseVAEToYaml(c.config as PhraseVAEConfigShape)}
                  id={c.id}
                />
              </div>
              <TimestampedMeta
                updatedAt={c.updatedAt.toISOString()}
                description={c.description}
              />
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
