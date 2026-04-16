import Link from "next/link";
import { Pencil, FileCog } from "lucide-react";
import type { PhraseVaeConfigPresetRow } from "@/lib/db/schema";
import { phraseVAEToYaml } from "@/lib/config-schemas/phrase-vae-yaml";
import type { PhraseVAEConfigShape } from "@/lib/config-schemas/phrase-vae";
import { PresetOverflowMenu } from "./preset-overflow-menu";
import { UsePresetButton, type CorpusOption } from "./use-preset-modal";
import { TimestampedMeta } from "./timestamped-meta";

/**
 * Right-side rail listing phrase-VAE *config presets* in most-recently
 * -updated order.  Per row:
 *
 *   - primary actions on the row itself: Edit (icon-only), Use (label)
 *   - secondary actions in a "…" overflow menu: Download YAML, Duplicate
 *
 * Splitting them this way keeps the row scannable when the preset name
 * is long, and makes "Use" the visually obvious next step.
 */

export function PresetsSidebar({
  presets,
  corpora,
  existingVaeNames,
}: {
  presets: PhraseVaeConfigPresetRow[];
  corpora: CorpusOption[];
  existingVaeNames: string[];
}) {
  return (
    <aside className="w-96 shrink-0 border-l border-separator bg-surface/40 p-6 flex flex-col gap-4 overflow-y-auto">
      <header>
        <h2 className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
          Config presets
        </h2>
        <p className="text-xs text-muted mt-1">
          Reusable templates.  Most recently edited first.
        </p>
      </header>

      {presets.length === 0 ? (
        <p className="text-sm text-muted leading-relaxed">
          No presets yet.  Create one to get started.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {presets.map((p) => {
            const cfg = p.config as PhraseVAEConfigShape;
            return (
              <li
                key={p.id}
                className="flex flex-col gap-1 rounded-[calc(var(--radius)*0.6)] px-2 py-1.5 hover:bg-default/40 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <FileCog
                    className="w-3.5 h-3.5 shrink-0 text-accent/70"
                    strokeWidth={2}
                    aria-hidden
                  />
                  <span className="flex-1 min-w-0 text-sm text-foreground/90 truncate">
                    {p.name}
                  </span>
                  <Link
                    href={`/models/phrase-vae/presets/${p.id}/edit`}
                    aria-label={`Edit ${p.name}`}
                    title="Edit preset"
                    className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-accent hover:bg-accent/10 transition-colors"
                  >
                    <Pencil className="w-3.5 h-3.5" strokeWidth={2} />
                  </Link>
                  <UsePresetButton
                    presetId={p.id}
                    presetName={p.name}
                    presetConfig={cfg}
                    corpora={corpora}
                    existingVaeNames={existingVaeNames}
                  />
                  <PresetOverflowMenu
                    id={p.id}
                    name={p.name}
                    yaml={phraseVAEToYaml(cfg)}
                  />
                </div>
                <TimestampedMeta
                  updatedAt={p.updatedAt.toISOString()}
                  description={p.description}
                />
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
