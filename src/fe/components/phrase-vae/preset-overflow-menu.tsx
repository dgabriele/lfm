"use client";

import { useTransition } from "react";
import { Dropdown } from "@heroui/react";
import { MoreVertical, Download, Copy } from "lucide-react";
import {
  duplicatePhraseVAEConfigPreset,
  touchPhraseVAEConfigPreset,
} from "@/lib/models/actions";

/**
 * Overflow ("…") menu for a preset row.  Houses the secondary
 * actions (Download YAML, Duplicate) so the row's primary surface
 * stays uncluttered with just Edit + Use buttons.
 */
export function PresetOverflowMenu({
  id,
  name,
  yaml,
}: {
  id: string;
  name: string;
  yaml: string;
}) {
  const [pending, startTransition] = useTransition();

  const onDownload = () => {
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name || "phrase-vae"}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
    startTransition(() => touchPhraseVAEConfigPreset(id));
  };

  const onDuplicate = () => {
    startTransition(() => duplicatePhraseVAEConfigPreset(id));
  };

  return (
    <Dropdown>
      <Dropdown.Trigger>
        <button
          type="button"
          aria-label={`More actions for ${name}`}
          title="More actions"
          disabled={pending}
          className="p-1.5 rounded-[calc(var(--radius)*0.5)] text-muted hover:text-foreground hover:bg-default/40 transition-colors disabled:opacity-40"
        >
          <MoreVertical className="w-3.5 h-3.5" strokeWidth={2} />
        </button>
      </Dropdown.Trigger>
      <Dropdown.Popover className="rounded-[calc(var(--radius)*0.5)] min-w-40 p-1 bg-surface border border-separator shadow-lg">
        <Dropdown.Menu aria-label={`Actions for ${name}`}>
          <Dropdown.Item
            onAction={onDownload}
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-[calc(var(--radius)*0.4)] hover:bg-default/40 outline-none"
          >
            <Download className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
            Download YAML
          </Dropdown.Item>
          <Dropdown.Item
            onAction={onDuplicate}
            className="flex items-center gap-2 px-3 py-2 text-sm rounded-[calc(var(--radius)*0.4)] hover:bg-default/40 outline-none"
          >
            <Copy className="w-3.5 h-3.5 text-muted" strokeWidth={2} />
            Duplicate
          </Dropdown.Item>
        </Dropdown.Menu>
      </Dropdown.Popover>
    </Dropdown>
  );
}
