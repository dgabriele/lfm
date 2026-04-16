import type { ReactNode } from "react";
import { SideNav } from "@/components/side-nav";

export default function ModelsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <SideNav />
      <main className="flex-1 min-w-0 min-h-0 flex flex-col">{children}</main>
    </div>
  );
}
