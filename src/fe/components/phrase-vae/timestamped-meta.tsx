"use client";

import { useEffect, useState } from "react";
import { Tooltip } from "@heroui/react";

/**
 * Renders a line of metadata for a config row:
 *
 *   <Updated 2d 5h ago> · <description, if any>
 *
 * The "ago" fragment re-computes every 60s so long-open tabs don't
 * go stale.  We deliberately don't show the absolute date — the
 * relative "Updated …" form is what the user wants to scan.
 */

function formatAgo(d: Date, now: number): string {
  const s = Math.max(0, Math.floor((now - d.getTime()) / 1000));
  if (s < 60) return "Updated just now";
  if (s < 3600) return `Updated ${Math.floor(s / 60)}m ago`;
  if (s < 86400) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return m ? `Updated ${h}h ${m}m ago` : `Updated ${h}h ago`;
  }
  const days = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  return h ? `Updated ${days}d ${h}h ago` : `Updated ${days}d ago`;
}

export function TimestampedMeta({
  updatedAt,
  description,
}: {
  updatedAt: string; // ISO string; Dates don't serialize across the client boundary
  description?: string | null;
}) {
  const date = new Date(updatedAt);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(t);
  }, []);

  const ago = formatAgo(date, now);
  const descr = description ?? "";
  const truncated =
    descr.length > DESCRIPTION_CAP
      ? descr.slice(0, DESCRIPTION_CAP).trimEnd() + "…"
      : descr;
  const needsTooltip = descr.length > DESCRIPTION_CAP;

  const body = (
    <p
      className="text-xs text-muted leading-snug"
      suppressHydrationWarning
    >
      <span className="text-foreground/80">{ago}</span>
      {truncated ? ` · ${truncated}` : null}
    </p>
  );

  if (!needsTooltip) return body;

  return (
    <Tooltip delay={200}>
      <Tooltip.Trigger>{body}</Tooltip.Trigger>
      <Tooltip.Content className="max-w-sm rounded-[calc(var(--radius)*0.5)] p-3 text-xs leading-relaxed">
        {descr}
      </Tooltip.Content>
    </Tooltip>
  );
}

const DESCRIPTION_CAP = 140;
