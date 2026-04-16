"use client";

import { useEffect, useState } from "react";
import { Tooltip } from "@heroui/react";

/**
 * Renders a line of metadata for a config row:
 *
 *   <Apr 15, 8:45 PM> · <2d 5h ago> · <description, if any>
 *
 * The "ago" fragment re-computes every 60s so long-open tabs don't
 * go stale.  The absolute date comes from the server timestamp and
 * is locale-stable ("en-US") so SSR and client agree on formatting.
 */

const ABSOLUTE = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

function formatAgo(d: Date, now: number): string {
  const s = Math.max(0, Math.floor((now - d.getTime()) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return m ? `${h}h ${m}m ago` : `${h}h ago`;
  }
  const days = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  return h ? `${days}d ${h}h ago` : `${days}d ago`;
}

export function TimestampedMeta({
  updatedAt,
  description,
}: {
  updatedAt: string; // ISO string; Dates don't serialize across the client boundary
  description?: string | null;
}) {
  const date = new Date(updatedAt);
  // `mounted` gates the absolute datetime: server TZ ≠ browser TZ, so
  // we only render the absolute on the client to avoid hydration
  // mismatch and a brief UTC flash.
  const [mounted, setMounted] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    setMounted(true);
    const t = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(t);
  }, []);

  const timeParts: string[] = [];
  if (mounted) timeParts.push(ABSOLUTE.format(date));
  timeParts.push(formatAgo(date, now));

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
      <span className="text-foreground/80">
        {timeParts.join(" · ")}
      </span>
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
