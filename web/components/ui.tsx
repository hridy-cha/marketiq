"use client";

import type { Signal } from "@/lib/api";

export function SignalBadge({ signal }: { signal: Signal | null | undefined }) {
  if (!signal) return <span className="text-ink-500">n/a</span>;
  const cls = { BUY: "bg-bull/20 text-bull border-bull/40", SELL: "bg-bear/20 text-bear border-bear/40",
    WATCH: "bg-watch/20 text-watch border-watch/40", HOLD: "bg-ink-700/40 text-ink-300 border-ink-600" }[signal];
  return <span className={`inline-block rounded border px-2 py-0.5 text-xs font-semibold ${cls}`}>{signal}</span>;
}

export function ScoreBar({ score, small }: { score: number | null | undefined; small?: boolean }) {
  if (score == null) return <span className="text-ink-500">n/a</span>;
  const color = score >= 70 ? "bg-bull" : score <= 35 ? "bg-bear" : score >= 55 ? "bg-watch" : "bg-ink-400";
  return (
    <div className="flex items-center gap-2">
      <div className={`${small ? "h-1.5 w-16" : "h-2 w-28"} overflow-hidden rounded bg-ink-800`}>
        <div className={`h-full ${color}`} style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
      </div>
      <span className="tabular-nums">{score.toFixed(small ? 0 : 1)}</span>
    </div>
  );
}

export function TrendTag({ trend }: { trend: string | null | undefined }) {
  const cls = trend === "bullish" ? "text-bull" : trend === "bearish" ? "text-bear" : "text-ink-400";
  return <span className={`text-xs ${cls}`}>{trend ?? "n/a"}</span>;
}

export function Pct({ v, digits = 1 }: { v: number | null | undefined; digits?: number }) {
  if (v == null) return <span className="text-ink-500">n/a</span>;
  const cls = v > 0 ? "text-bull" : v < 0 ? "text-bear" : "text-ink-300";
  return <span className={`tabular-nums ${cls}`}>{(v * 100).toFixed(digits)}%</span>;
}

export function Loading({ what = "" }: { what?: string }) {
  return <div className="py-8 text-center text-sm text-ink-500">Loading {what}…</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="rounded border border-bear/40 bg-bear/10 p-3 text-sm text-bear">{msg}</div>;
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="rounded border border-dashed border-ink-700 p-6 text-center text-sm text-ink-400">{children}</div>;
}
