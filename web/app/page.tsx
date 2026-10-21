"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { api, fetcher, fmtNum, type Rankings, type RankingRow, type Signal } from "@/lib/api";
import { Empty, ErrorBox, Loading, Pct, ScoreBar, SignalBadge, TrendTag } from "@/components/ui";

type SortKey = "rank" | "score" | "ret_1" | "ret_20" | "rsi_14" | "symbol";

export default function RankingsPage() {
  const { data, error, isLoading, mutate } = useSWR<Rankings>("/api/rankings", fetcher);
  const [filter, setFilter] = useState<Signal | "ALL">("ALL");
  const [sort, setSort] = useState<SortKey>("rank");
  const [asc, setAsc] = useState(true);
  const [scanning, setScanning] = useState(false);

  const rows = useMemo(() => {
    if (!data) return [];
    let r = data.rows.filter((x) => filter === "ALL" || x.signal === filter);
    r = [...r].sort((a, b) => {
      const av = a[sort] ?? -Infinity;
      const bv = b[sort] ?? -Infinity;
      const cmp = typeof av === "string" ? String(av).localeCompare(String(bv)) : Number(av) - Number(bv);
      return asc ? cmp : -cmp;
    });
    return r;
  }, [data, filter, sort, asc]);

  const runScan = async () => {
    setScanning(true);
    try {
      await api("/api/scan", { method: "POST" });
      await mutate();
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  };

  const th = (key: SortKey, label: string) => (
    <th className="cursor-pointer select-none hover:text-ink-100" onClick={() => { if (sort === key) setAsc(!asc); else { setSort(key); setAsc(key === "rank" || key === "symbol"); } }}>
      {label}{sort === key ? (asc ? " ▲" : " ▼") : ""}
    </th>
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = { BUY: 0, WATCH: 0, HOLD: 0, SELL: 0 };
    for (const r of data?.rows ?? []) if (r.signal) c[r.signal] = (c[r.signal] ?? 0) + 1;
    return c;
  }, [data]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Rankings</h1>
          <p className="text-sm text-ink-400">
            Composite 0–100 score across trend, momentum, volume, volatility, fundamentals, sentiment and ML.
            {data?.as_of && <> As of <span className="font-mono text-ink-200">{data.as_of}</span>.</>}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {(["ALL", "BUY", "WATCH", "HOLD", "SELL"] as const).map((f) => (
            <button key={f} className={`btn ${filter === f ? "btn-primary" : ""}`} onClick={() => setFilter(f)}>
              {f}{f !== "ALL" && <span className="ml-1 text-xs opacity-70">{counts[f] ?? 0}</span>}
            </button>
          ))}
          <button className="btn" onClick={runScan} disabled={scanning}>{scanning ? "Scanning…" : "Run scan"}</button>
        </div>
      </div>

      {isLoading && <Loading what="rankings" />}
      {error && <ErrorBox error={error} />}
      {data && data.rows.length === 0 && <Empty>{data.message ?? "No data yet."} Click “Run scan” or run <code>marketiq scan</code>.</Empty>}

      {rows.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <table className="table-base">
            <thead>
              <tr>
                {th("rank", "#")}
                {th("symbol", "Symbol")}
                <th>Name</th>
                {th("score", "Score")}
                <th>Signal</th>
                <th>Trend</th>
                <th>Close</th>
                {th("ret_1", "1d")}
                {th("ret_20", "20d")}
                {th("rsi_14", "RSI")}
                <th>Components</th>
                <th>Recent patterns / flags</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => <Row key={r.symbol} r={r} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Row({ r }: { r: RankingRow }) {
  const comps = ["trend", "momentum", "volume", "volatility", "fundamentals", "sentiment", "ml"];
  return (
    <tr>
      <td className="text-ink-500">{r.rank}</td>
      <td><Link href={`/stocks/${r.symbol}`} className="font-semibold text-accent hover:underline">{r.symbol}</Link></td>
      <td className="text-ink-300">{r.name}<span className="ml-2 text-xs text-ink-500">{r.sector}</span></td>
      <td><ScoreBar score={r.score} /></td>
      <td><SignalBadge signal={r.signal} /></td>
      <td><TrendTag trend={r.trend} /></td>
      <td>{fmtNum(r.close)}</td>
      <td><Pct v={r.ret_1} /></td>
      <td><Pct v={r.ret_20} /></td>
      <td>{fmtNum(r.rsi_14, 0)}</td>
      <td>
        <div className="flex gap-1">
          {comps.map((c) => {
            const v = r.components[c];
            return (
              <span key={c} title={`${c}: ${v == null ? "unavailable" : v.toFixed(0)}`}
                className={`inline-block h-4 w-3 rounded-sm ${v == null ? "bg-ink-800" : v >= 70 ? "bg-bull" : v <= 35 ? "bg-bear" : "bg-ink-500"}`}
                style={{ opacity: v == null ? 0.5 : 0.35 + 0.65 * (v / 100) }} />
            );
          })}
        </div>
      </td>
      <td className="max-w-xs text-xs text-ink-400">
        {r.recent_patterns.map((p) => <span key={p} className="mr-1 rounded bg-ink-800 px-1 py-0.5">{p.replace(/_/g, " ")}</span>)}
        {r.risk_flags.map((f) => <span key={f} className="mr-1 rounded bg-watch/20 px-1 py-0.5 text-watch">{f.split(":")[0]}</span>)}
      </td>
    </tr>
  );
}
