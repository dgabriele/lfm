"use client";

import Link from "next/link";
import { useCallback } from "react";

/**
 * Pagination for the sample list.  URL-driven (`?page=N`) so deep-
 * links / back-button behavior are both natural: each page is a
 * distinct history entry, and pressing Back walks through pages
 * exactly.  We render the edges + a contextual window around the
 * current page to keep the bar compact on long corpora.
 */
export function SamplePager({
  corpusId,
  page,
  totalPages,
}: {
  corpusId: string;
  page: number;
  totalPages: number;
}) {
  const href = useCallback(
    (p: number) =>
      p === 1
        ? `/corpora/${encodeURIComponent(corpusId)}`
        : `/corpora/${encodeURIComponent(corpusId)}?page=${p}`,
    [corpusId],
  );

  if (totalPages <= 1) return null;

  const visible = window_around(page, totalPages, 2);

  return (
    <nav
      aria-label="Sample pagination"
      className="flex items-center gap-1 pt-2"
    >
      <PageLink
        href={href(Math.max(1, page - 1))}
        disabled={page <= 1}
        label="Prev"
      />
      {visible.map((p, i) =>
        p === "…" ? (
          <span
            key={`gap-${i}`}
            className="px-2 text-muted text-sm select-none"
          >
            …
          </span>
        ) : (
          <PageLink
            key={p}
            href={href(p)}
            label={p.toLocaleString()}
            active={p === page}
          />
        ),
      )}
      <PageLink
        href={href(Math.min(totalPages, page + 1))}
        disabled={page >= totalPages}
        label="Next"
      />
    </nav>
  );
}

function PageLink({
  href,
  label,
  active,
  disabled,
}: {
  href: string;
  label: string;
  active?: boolean;
  disabled?: boolean;
}) {
  const base =
    "inline-flex items-center justify-center min-w-9 h-9 px-3 rounded-[calc(var(--radius)*0.8)] text-sm transition-colors tabular-nums";
  if (disabled) {
    return (
      <span
        className={`${base} text-muted/40 pointer-events-none`}
        aria-disabled
      >
        {label}
      </span>
    );
  }
  return (
    <Link
      href={href}
      className={[
        base,
        active
          ? "bg-accent/15 text-accent font-semibold"
          : "text-muted hover:bg-default/60 hover:text-foreground",
      ].join(" ")}
      aria-current={active ? "page" : undefined}
    >
      {label}
    </Link>
  );
}

/** Page numbers to render: edges + window around current. */
function window_around(
  page: number,
  total: number,
  span: number,
): Array<number | "…"> {
  const out: Array<number | "…"> = [];
  const push = (v: number | "…") => {
    if (
      out.length &&
      typeof out[out.length - 1] === "number" &&
      v === out[out.length - 1]
    )
      return;
    out.push(v);
  };
  const start = Math.max(2, page - span);
  const end = Math.min(total - 1, page + span);
  push(1);
  if (start > 2) push("…");
  for (let i = start; i <= end; i++) push(i);
  if (end < total - 1) push("…");
  if (total > 1) push(total);
  return out;
}
