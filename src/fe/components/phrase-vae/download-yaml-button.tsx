"use client";

import { useTransition } from "react";
import { Download } from "lucide-react";
import { touchPhraseVAEConfigPreset } from "@/lib/models/actions";

/**
 * Quick action: download the preset's YAML to disk and bump
 * `updated_at` so the row rises in the MRU presets list.
 */
export function DownloadYamlButton({
  id,
  name,
  yaml,
}: {
  id: string;
  name: string;
  yaml: string;
}) {
  const [pending, startTransition] = useTransition();

  const onClick = () => {
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name || "phrase-vae"}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
    startTransition(() => touchPhraseVAEConfigPreset(id));
  };

  return (
    <button
      type="button"
      aria-label={`Download YAML for ${name}`}
      title="Download YAML"
      onClick={onClick}
      disabled={pending}
      className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-accent hover:bg-accent/10 transition-colors disabled:opacity-40"
    >
      <Download className="w-3.5 h-3.5" strokeWidth={2} />
    </button>
  );
}
