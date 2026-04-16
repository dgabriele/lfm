"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";

/**
 * Back-navigation that stays inside the site.
 *
 * History inside a SPA can branch in ways that make a naked
 * `router.back()` unreliable: if the user landed deep-linked on this
 * page (e.g. they opened `/corpora/english-constituents-v13` in a new
 * tab), calling `back()` would take them out of the app entirely.
 *
 * Strategy: on mount, record the initial history `length`.  If the
 * user navigated within the app since then (history grew), we can
 * safely go back.  Otherwise we fall through to a normal
 * `<Link href="/corpora">` which pushes a fresh history entry onto
 * the stack.
 */
export function BackLink({
  fallbackHref = "/corpora",
  label = "Back",
}: {
  fallbackHref?: string;
  label?: string;
}) {
  const router = useRouter();
  const [canGoBack, setCanGoBack] = useState(false);
  const [initialLen, setInitialLen] = useState<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setInitialLen(window.history.length);
  }, []);

  useEffect(() => {
    if (initialLen == null) return;
    // We consider the user "inside the app" if history has grown since
    // mount AND the document referrer is same-origin.
    const sameOrigin =
      document.referrer && new URL(document.referrer).origin === location.origin;
    setCanGoBack(window.history.length > initialLen || !!sameOrigin);
  }, [initialLen]);

  const onClick = useCallback(
    (e: React.MouseEvent) => {
      if (canGoBack) {
        e.preventDefault();
        router.back();
      }
      // else: let the Link navigate to fallbackHref normally.
    },
    [canGoBack, router],
  );

  return (
    <Link
      href={fallbackHref}
      onClick={onClick}
      className="self-start text-sm text-muted hover:text-accent transition-colors flex items-center gap-1.5"
    >
      <ArrowLeft aria-hidden className="w-[1em] h-[1em]" strokeWidth={2} />
      {label}
    </Link>
  );
}
