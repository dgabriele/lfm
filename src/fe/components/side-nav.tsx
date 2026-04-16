"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Library, type LucideIcon } from "lucide-react";

const NAV_ITEMS: { label: string; href: string; icon: LucideIcon }[] = [
  { label: "Corpora", href: "/corpora", icon: Library },
];

/**
 * Fixed left rail.  Single-item menu for now; structure lets us add
 * more sections later without touching any page.  Active-item styling
 * is derived from the current pathname so deep links highlight
 * correctly.
 */
export function SideNav() {
  const pathname = usePathname();

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
        {NAV_ITEMS.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={[
                "flex items-center gap-2.5 px-3 py-2 rounded-[var(--radius)] text-sm",
                "transition-colors",
                isActive
                  ? "bg-accent/15 text-accent"
                  : "text-muted hover:bg-default/60 hover:text-foreground",
              ].join(" ")}
            >
              <item.icon className="w-[1.1em] h-[1.1em]" strokeWidth={2} />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
