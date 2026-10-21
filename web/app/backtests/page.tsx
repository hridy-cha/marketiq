"use client";

import { useState } from "react";
import useSWR from "swr";
import { api, fetcher, fmtMoney, fmtNum, type BacktestDetail, type BacktestSummary } from "@/lib/api";
import { Empty, ErrorBox, Loading, Pct } from "@/components/ui";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";

const PCT_KEYS = new Set(["total_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate", "benchmark_total_return",
  "benchmark_annual_return", "benchmark_max_drawdown", "excess_return", "trade_win_rate", "avg_exposure"]);

export default function BacktestsPage() {
  const list = useSWR<{ runs: BacktestSummary[] }>("/api/backtests", fetcher);
  const [selected, setSelected] = useState<number | null>(null);
  const detail = useSWR<BacktestDetail>(selected ? `/api/backtests/${selected}` : null, fetcher);
  const [form, setForm] = useState({ start: "2021-01-04", end: "2025-10-15", max_positions: 8, rebalance_every: 5, exit_score: 45, stop_loss_pct: 0.15, enter_on_watch: false, use_ml: true, name: "signal-strategy" });
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setRunning(true); setErr(null);
    try {
      const r = await api<{ id: number }>("/api/backtests", { method: "POST", body: JSON.stringify(form) });
      await list.mutate();
      setSelected(r.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const d = detail.data;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Backtests</h1>
        <p className="text-sm text-ink-400">Signal strategy, long only: enter on BUY, exit on SELL / low score / stop loss. Fills at the next open with commission and slippage. Universe is the current large-cap list (survivorship bias applies).</p>
      </div>

      <div className="card">
        <div className="card-title">New run (synchronous; a 4-year weekly run takes 1–2 minutes)</div>
        <div className="flex flex-wrap items-end gap-3 text-xs">
          {([["name", "Name", "text"], ["start", "Start", "date"], ["end", "End", "date"], ["max_positions", "Max positions", "number"],
            ["rebalance_every", "Rebalance every (days)", "number"], ["exit_score", "Exit score", "number"], ["stop_loss_pct", "Stop loss (0 = off)", "number"]] as const).map(([k, label, type]) => (
            <label key={k} className="flex flex-col gap-1 text-ink-400">{label}
              <input className="input w-36" type={type} step={k === "stop_loss_pct" ? 0.01 : 1} value={String(form[k])}
                onChange={(e) => setForm({ ...form, [k]: type === "number" ? Number(e.target.value) : e.target.value })} />
            </label>
          ))}
          <label className="flex items-center gap-1 text-ink-400"><input type="checkbox" checked={form.enter_on_watch} onChange={(e) => setForm({ ...form, enter_on_watch: e.target.checked })} />Enter on WATCH too</label>
          <label className="flex items-center gap-1 text-ink-400"><input type="checkbox" checked={form.use_ml} onChange={(e) => setForm({ ...form, use_ml: e.target.checked })} />Use ML (if gate passes)</label>
          <button className="btn btn-primary" onClick={run} disabled={running}>{running ? "Running…" : "Run backtest"}</button>
          {err && <span className="text-bear">{err}</span>}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card p-0 lg:col-span-1">
          <div className="card-title p-4 pb-0">Runs</div>
          {list.isLoading && <Loading />}
          {list.error && <div className="p-4"><ErrorBox error={list.error} /></div>}
          {list.data && list.data.runs.length === 0 && <div className="p-4"><Empty>No runs yet.</Empty></div>}
          <table className="table-base">
            <tbody>
              {list.data?.runs.map((r) => (
                <tr key={r.id} className={selected === r.id ? "bg-ink-800/60" : "cursor-pointer"} onClick={() => setSelected(r.id)}>
                  <td className="text-ink-500">#{r.id}</td>
                  <td>{r.name}<div className="text-xs text-ink-500">{r.kind} · {r.start_date} → {r.end_date}</div></td>
                  <td className="text-right"><Pct v={r.metrics.total_return as number} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="space-y-4 lg:col-span-2">
          {detail.isLoading && <Loading what="run" />}
          {d && (
            <>
              <div className="card">
                <div className="card-title">Equity curve · {d.name} #{d.id}</div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={d.equity_curve}>
                    <CartesianGrid stroke="#22262e" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#7c8596" }} minTickGap={60} />
                    <YAxis tick={{ fontSize: 10, fill: "#7c8596" }} width={64} tickFormatter={(v) => fmtMoney(v)} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ background: "#15181d", border: "1px solid #343a45", fontSize: 12 }} formatter={(v: number) => fmtMoney(v)} />
                    <Legend />
                    <Line type="monotone" dataKey="equity" name="Strategy" stroke="#3b6ff0" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="benchmark" name="Benchmark (SPY)" stroke="#7c8596" dot={false} strokeWidth={1} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="card">
                  <div className="card-title">Metrics</div>
                  <table className="w-full text-xs">
                    <tbody>
                      {Object.entries(d.metrics).map(([k, v]) => (
                        <tr key={k}><td className="py-0.5 text-ink-400">{k.replace(/_/g, " ")}</td>
                          <td className="py-0.5 text-right font-mono">{v == null ? "n/a" : typeof v === "number" ? (PCT_KEYS.has(k) ? `${(v * 100).toFixed(2)}%` : fmtNum(v, 3)) : String(v)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  {d.config.ml_status != null && <div className="mt-2 text-xs text-ink-500">ML: {String(d.config.ml_status)}</div>}
                </div>
                <div className="card max-h-[480px] overflow-auto">
                  <div className="card-title">Trades ({d.trades.length})</div>
                  <table className="w-full text-xs">
                    <thead><tr className="text-ink-500"><th className="text-left">Date</th><th className="text-left">Side</th><th className="text-left">Sym</th><th className="text-right">Price</th><th className="text-right">P&amp;L</th><th className="text-left">Reason</th></tr></thead>
                    <tbody>
                      {d.trades.slice().reverse().map((t, i) => (
                        <tr key={i}><td>{t.date}</td><td className={t.side === "BUY" ? "text-bull" : "text-bear"}>{t.side}</td><td>{t.symbol}</td>
                          <td className="text-right font-mono">{fmtNum(t.price)}</td>
                          <td className={`text-right font-mono ${(t.pnl ?? 0) > 0 ? "text-bull" : (t.pnl ?? 0) < 0 ? "text-bear" : ""}`}>{t.pnl == null ? "" : t.pnl.toFixed(0)}</td>
                          <td className="text-ink-400">{t.reason}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
