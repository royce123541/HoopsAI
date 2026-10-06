import type { Metadata } from "next";
import { Barlow, Barlow_Condensed } from "next/font/google";
import Link from "next/link";

import { NavLinks } from "@/components/NavLinks";
import "./globals.css";

const barlow = Barlow({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const barlowCondensed = Barlow_Condensed({
  variable: "--font-barlow-condensed",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

export const metadata: Metadata = {
  title: "HoopsAI",
  description:
    "Machine-learned NBA win probabilities, before tip-off and live.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${barlow.variable} ${barlowCondensed.variable} h-full`}
    >
      <body className="flex min-h-full flex-col">
        <header className="border-b border-line">
          <div className="mx-auto flex w-full max-w-4xl items-center justify-between gap-6 px-4 py-3">
            <Link
              href="/"
              className="font-condensed text-2xl font-bold tracking-tight text-ink"
              aria-label="HoopsAI home"
            >
              HoopsAI
            </Link>
            <NavLinks />
          </div>
        </header>
        <main className="mx-auto w-full max-w-4xl flex-1 px-4 pt-8 pb-16">
          {children}
        </main>
        <footer className="border-t border-line">
          <p className="mx-auto max-w-4xl px-4 py-4 text-sm text-muted">
            Probabilities come from a model trained on NBA results since 2005.
            They are estimates, not guarantees.
          </p>
        </footer>
      </body>
    </html>
  );
}
