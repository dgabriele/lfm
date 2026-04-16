"use client";

import Link from "next/link";
import { Boxes, FileCog, Database, Cpu } from "lucide-react";
import type { PhraseVaeRow, PhraseVaeConfigPresetRow } from "@/lib/db/schema";
import { TimestampedMeta } from "./timestamped-meta";

/**
 * Filesystem-like listing of trained PhraseVAEs.  Rows are dense and
 * scannable: name + status + corpus + type + source preset + last
 * updated.  Click name to drill into the VAE detail page.
 *
 * (Built on a styled native table rather than HeroUI's React-Aria
 * Table primitive — the React-Aria Table API is heavy for a static
 * listing and doesn't add accessibility we don't already have via
 * semantic <table> markup.  We can swap in HeroUI Table later if we
 * need column-resize / sort / row-selection.)
 */

const STATUS_COLOR: Record<string, string> = {
  initialized: "text-muted bg-default/40",
  training: "text-amber-300 bg-amber-400/15",
  paused: "text-muted bg-default/40",
  trained: "text-emerald-300 bg-emerald-400/15",
  failed: "text-red-300 bg-red-400/15",
  archived: "text-muted bg-default/30",
};

const VAE_TYPE_LABEL: Record<string, string> = {
  ipa: "IPA",
  token_vocab: "BPE",
};

export function VaesTable({
  vaes,
  presetsById,
}: {
  vaes: PhraseVaeRow[];
  presetsById: Record<string, PhraseVaeConfigPresetRow>;
}) {
  if (vaes.length === 0) {
    return (
      <div className="flex-1 min-h-0 flex flex-col items-center justify-center gap-3 text-center">
        <Boxes className="w-10 h-10 text-accent/70" strokeWidth={1.5} />
        <h2 className="text-lg font-semibold">No phrase VAEs yet</h2>
        <p className="text-sm text-muted max-w-md leading-relaxed">
          Use the <span className="text-accent">Use</span> action on a
          config preset in the right rail to instantiate your first
          phrase VAE.
        </p>
      </div>
    );
  }
  return (
    <div className="flex-1 min-h-0 overflow-y-auto pr-2 -mr-2">
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-background/95 backdrop-blur z-10">
          <tr className="text-xs uppercase tracking-wider text-accent/80 font-semibold">
            <Th>Name</Th>
            <Th>Status</Th>
            <Th>Corpus</Th>
            <Th>Type</Th>
            <Th>Source preset</Th>
            <Th>Updated</Th>
          </tr>
        </thead>
        <tbody>
          {vaes.map((v) => {
            const preset = v.presetId ? presetsById[v.presetId] : null;
            return (
              <tr
                key={v.id}
                className="border-t border-separator/60 hover:bg-default/30 transition-colors"
              >
                <Td>
                  <Link
                    href={`/models/phrase-vae/${v.id}`}
                    className="flex items-center gap-2 text-foreground/90 hover:text-accent"
                  >
                    <Boxes
                      className="w-4 h-4 shrink-0 text-accent/80"
                      strokeWidth={2}
                      aria-hidden
                    />
                    <span className="truncate font-medium">{v.name}</span>
                  </Link>
                </Td>
                <Td>
                  <span
                    className={`inline-block px-2 py-0.5 text-xs rounded-[calc(var(--radius)*0.5)] ${
                      STATUS_COLOR[v.status] ?? "text-muted"
                    }`}
                  >
                    {v.status}
                  </span>
                </Td>
                <Td>
                  <span className="inline-flex items-center gap-1.5 text-foreground/80">
                    <Database className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
                    <span className="truncate font-mono text-xs">
                      {v.corpusId}
                    </span>
                  </span>
                </Td>
                <Td>
                  <span className="inline-flex items-center gap-1.5 text-foreground/80">
                    <Cpu className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
                    {VAE_TYPE_LABEL[v.vaeType] ?? v.vaeType}
                  </span>
                </Td>
                <Td>
                  {preset ? (
                    <Link
                      href={`/models/phrase-vae/presets/${preset.id}/edit`}
                      className="inline-flex items-center gap-1.5 text-foreground/80 hover:text-accent"
                    >
                      <FileCog
                        className="w-3.5 h-3.5 text-muted"
                        strokeWidth={2}
                      />
                      <span className="truncate">{preset.name}</span>
                    </Link>
                  ) : (
                    <span className="text-xs text-muted italic">deleted</span>
                  )}
                </Td>
                <Td>
                  <TimestampedMeta
                    updatedAt={v.updatedAt.toISOString()}
                    description={null}
                  />
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="text-left px-3 py-2 font-semibold border-b border-separator">
      {children}
    </th>
  );
}

function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-3 py-2 align-middle">{children}</td>;
}
