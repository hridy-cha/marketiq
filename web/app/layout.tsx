import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "MarketIQ",
  description: "Explainable stock research platform. Research tool only, not investment advice.",
};

const nav = [
  { href: "/", label: "Rankings" },
  { href: "/watchlist", label: "Watchlist" },
  { href: "/backtests", label: "Backtests" },
  { href: "/paper", label: "Paper trading" },
  { href: "/ml", label: "ML model" },
  { href: "/settings", label: "Settings" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-ink-800 bg-ink-900/60">
          <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              Market<span className="text-accent">IQ</span>
            </Link>
            <nav className="flex gap-4 text-sm text-ink-300">
              {nav.map((n) => (
                <Link key={n.href} href={n.href} className="hover:text-ink-50">
                  {n.label}
                </Link>
              ))}
            </nav>
            <span className="ml-auto text-xs text-ink-500">Research tool. Not investment advice.</span>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
