"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ReactNode } from "react";

/**
 * Client-side providers shared across the app.  Currently just
 * next-themes for dark/light switching; the HTML element is annotated
 * with `suppressHydrationWarning` in the root layout because
 * next-themes writes the class before React hydrates.
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}
