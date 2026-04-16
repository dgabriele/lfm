"use client";

import { useTransition } from "react";
import { Copy } from "lucide-react";
import { duplicatePhraseVAEConfigPreset } from "@/lib/models/actions";

/**
 * Quick action: fork an existing preset into a new "(copy)" preset.
 * The server picks a non-colliding name suffix and redirects to the
 * new preset's editor.
 */
export function DuplicatePresetButton({ id, name }: { id: string; name: string }) {
  const [pending, startTransition] = useTransition();
  return (
    <button
      type="button"
      aria-label={`Duplicate ${name}`}
      title="Duplicate preset"
      disabled={pending}
      onClick={() => startTransition(() => duplicatePhraseVAEConfigPreset(id))}
      className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-accent hover:bg-accent/10 transition-colors disabled:opacity-40"
    >
      <Copy className="w-3.5 h-3.5" strokeWidth={2} />
    </button>
  );
}
