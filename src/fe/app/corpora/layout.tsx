import type { ReactNode } from "react";
import { SideNav } from "@/components/side-nav";

/**
 * Layout shared by every page under `/corpora`: a fixed-width left nav
 * rail that holds top-level navigation and a flex-fluid main column
 * that fills the remainder of the viewport.  Responsive by virtue of
 * the fluid root font size; no media queries required.
 */
export default function CorporaSectionLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="flex h-screen overflow-hidden">
      <SideNav />
      <main className="flex-1 min-w-0 min-h-0 flex flex-col">{children}</main>
    </div>
  );
}
