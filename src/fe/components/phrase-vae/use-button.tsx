"use client";

import { useTransition } from "react";
import { Download } from "lucide-react";
import { touchPhraseVAEModel } from "@/lib/models/actions";

/**
 * "Use" quick action: downloads the YAML to the user's machine and
 * bumps the config's updated_at so the MRU order reflects activity.
 */
export function UseButton({
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
    startTransition(() => touchPhraseVAEModel(id));
  };

  return (
    <button
      type="button"
      aria-label={`Use ${name} — download YAML`}
      onClick={onClick}
      disabled={pending}
      className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-accent hover:bg-accent/10 transition-colors disabled:opacity-40"
    >
      <Download className="w-3.5 h-3.5" strokeWidth={2} />
    </button>
  );
}
