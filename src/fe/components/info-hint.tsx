"use client";

import { Info } from "lucide-react";
import { Popover } from "@heroui/react";

/**
 * Small info-circle trigger that reveals an explanatory popover on
 * click.  Used inline next to metadata labels when the raw number
 * alone doesn't convey what it measures.
 */
export function InfoHint({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Popover>
      <Popover.Trigger>
        <button
          type="button"
          aria-label={`About ${label}`}
          className="inline-flex items-center justify-center text-muted/70 hover:text-foreground transition-colors rounded-full focus:outline-none focus:ring-2 focus:ring-foreground/30"
        >
          <Info className="w-[0.9em] h-[0.9em]" strokeWidth={2} />
        </button>
      </Popover.Trigger>
      <Popover.Content className="max-w-xs rounded-[calc(var(--radius)*0.5)] overflow-hidden">
        <Popover.Dialog className="p-3 text-xs leading-relaxed rounded-[calc(var(--radius)*0.5)]">
          <div className="text-accent/80 font-semibold uppercase tracking-wider mb-1">
            {label}
          </div>
          <div className="text-foreground/90">{children}</div>
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  );
}
