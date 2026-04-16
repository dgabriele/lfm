"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Disclosure, DisclosureGroup } from "@heroui/react";
import {
  Library,
  Boxes,
  ChevronDown,
  type LucideIcon,
} from "lucide-react";

/**
 * Fixed left rail.  Two kinds of nav entries:
 *
 *   - Flat links (e.g. Corpora) — single destination.
 *   - Groups (e.g. Models) — a section header that expands to reveal
 *     nested child links.  Built on HeroUI's Disclosure primitive so
 *     keyboard accessibility and aria semantics come for free.
 *
 * Active highlighting is derived from the current pathname so deep
 * links light up the right item; group containers stay expanded by
 * default whenever any child is on the active route.
 */

type NavLink = { kind: "link"; label: string; href: string; icon: LucideIcon };
type NavGroup = {
  kind: "group";
  id: string;
  label: string;
  icon: LucideIcon;
  children: { label: string; href: string }[];
};
type NavEntry = NavLink | NavGroup;

const NAV: NavEntry[] = [
  { kind: "link", label: "Corpora", href: "/corpora", icon: Library },
  {
    kind: "group",
    id: "models",
    label: "Models",
    icon: Boxes,
    children: [
      { label: "Phrase VAE", href: "/models/phrase-vae" },
    ],
  },
];

const FLAT_BASE =
  "flex items-center gap-2.5 px-3 py-2 rounded-[var(--radius)] text-sm transition-colors";

export function SideNav() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    pathname === href || pathname.startsWith(`${href}/`);

  // Default-expand any group that contains the active route.
  const defaultExpanded = NAV.flatMap((entry) =>
    entry.kind === "group" && entry.children.some((c) => isActive(c.href))
      ? [entry.id]
      : [],
  );

  return (
    <aside className="w-60 shrink-0 border-r border-separator flex flex-col py-6 px-4 gap-1">
      <div className="px-3 pb-6">
        <Link
          href="/"
          className="font-semibold tracking-tight text-lg text-foreground"
        >
          LFM
        </Link>
      </div>
      <nav className="flex flex-col gap-1">
        <DisclosureGroup
          defaultExpandedKeys={defaultExpanded.length ? defaultExpanded : ["models"]}
        >
          {NAV.map((entry) =>
            entry.kind === "link" ? (
              <Link
                key={entry.href}
                href={entry.href}
                className={[
                  FLAT_BASE,
                  isActive(entry.href)
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:bg-default/60 hover:text-foreground",
                ].join(" ")}
              >
                <entry.icon className="w-[1.1em] h-[1.1em]" strokeWidth={2} />
                {entry.label}
              </Link>
            ) : (
              <Disclosure key={entry.id} id={entry.id}>
                <Disclosure.Heading>
                  <Disclosure.Trigger
                    className={[
                      FLAT_BASE,
                      "w-full justify-between text-muted hover:bg-default/60 hover:text-foreground",
                    ].join(" ")}
                  >
                    <span className="flex items-center gap-2.5">
                      <entry.icon className="w-[1.1em] h-[1.1em]" strokeWidth={2} />
                      {entry.label}
                    </span>
                    <Disclosure.Indicator>
                      <ChevronDown
                        className="w-3.5 h-3.5 transition-transform data-[expanded]:rotate-180"
                        strokeWidth={2}
                      />
                    </Disclosure.Indicator>
                  </Disclosure.Trigger>
                </Disclosure.Heading>
                <Disclosure.Content>
                  <Disclosure.Body className="flex flex-col gap-1 pl-7 pt-1">
                    {entry.children.map((c) => (
                      <Link
                        key={c.href}
                        href={c.href}
                        className={[
                          "px-3 py-1.5 rounded-[var(--radius)] text-sm transition-colors",
                          isActive(c.href)
                            ? "bg-accent/15 text-accent"
                            : "text-muted hover:bg-default/60 hover:text-foreground",
                        ].join(" ")}
                      >
                        {c.label}
                      </Link>
                    ))}
                  </Disclosure.Body>
                </Disclosure.Content>
              </Disclosure>
            ),
          )}
        </DisclosureGroup>
      </nav>
    </aside>
  );
}
