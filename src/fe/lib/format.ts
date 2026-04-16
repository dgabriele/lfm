/**
 * Shared number formatting.  Pinned to `en-US` so SSR output is stable
 * regardless of the container's system locale (node:22-slim defaults
 * to POSIX, which renders `Number.toLocaleString()` without thousands
 * separators).
 */

const NUM = new Intl.NumberFormat("en-US");

export function formatNumber(n: number | bigint): string {
  return NUM.format(n);
}
