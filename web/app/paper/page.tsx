"use client";

import { useState } from "react";
import useSWR from "swr";
import { api, fetcher, fmtMoney, fmtNum, type Portfolio } from "@/lib/api";
import { Empty, ErrorBox, Loading, Pct } from "@/components/ui";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function PaperPage() {
  const [account, setAccount] = useState("default");
  const { data, error, isLoading, mutate } = useSWR<Portfolio>(`/api/paper/accounts/${account}`, fetcher, { shouldRetryOnError: false });
  const [order, setOrder] = useState({ symbol: "", side: "BUY", notional: 5000 });
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const call = async (fn: () => Promise<unknown>) => {
    setBusy(true); setMsg(null);
    try { await fn(); await mutate(); } catch (e) { setMsg(e instanceof Error ? e.message : String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Paper trading</h1>
        <p className="text-sm text-ink-400">Simulated account filled at the latest stored close. No real money moves; live trading is disabled by default.</p>
      </div>
      <div className="card flex flex-wrap items-center gap-2 text-sm">
        <label className="text-ink-400">Account <input className="input ml-1 w-32" value={account} onChange={(e) => setAccount(e.target.value)} /></label>
        <button className="btn" disabled={busy} onClick={() => call(() => api("/api/paper/accounts", { method: "POST", body: JSON.stringify({ name: account, starting_cash: 100000 }) }))}>Create ($100k)</button>
        <button className="btn" disabled={busy} onClick={() => { if (confirm("Reset this paper account to $100k and clear history?")) call(() => api("/api/paper/accounts", { method: "POST", body: JSON.stringify({ name: account, starting_cash: 100000, reset: true }) })); }}>Reset</button>
        <button className="btn btn-primary" disabled={busy} onClick={() => call(() => api(`/api/paper/accounts/${account}/rebalance`, { method: "POST" }))}>Rebalance from latest scan</button>
        <span className="mx-2 text-ink-700">|</span>
        <input className="input w-24" placeholder="Symbol" value={order.symbol} onChange={(e) => setOrder({ ...order, symbol: e.target.value.toUpperCase() })} />
        <select className="input" value={order.side} onChange={(e) => setOrder({ ...order, side: e.target.value })}><option>BUY</option><option>SELL</option></select>
        <input className="input w-28" type="number" value={order.notional} onChange={(e) => setOrder({ ...order, notional: Number(e.target.value) })} />
        <button className="btn" disabled={busy || !order.symbol} onClick={() => call(() => api(`/api/paper/accounts/${account}/orders`, { method: "POST", body: JSON.stringify({ symbol: order.symbol, side: order.side, notional: order.side === "BUY" ? order.notional : undefined }) }))}>Place order</button>
        {msg && <span className="text-xs text-bear">{msg}</span>}
      </div>

      {isLoading && <Loading />}
      {error && <Empty>Account “{account}” does not exist yet. Click Create.</Empty>}
      {data && (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            {[["Equity", fmtMoney(data.equity)], ["Cash", fmtMoney(data.cash)], ["Invested", fmtMoney(data.invested)]].map(([k, v]) => (
              <div key={k} className="card"><div className="card-title">{k}</div><div className="text-xl font-semibold tabular-nums">{v}</div></div>
            ))}
            <div className="card"><div className="card-title">Total return</div><div className="text-xl font-semibold"><Pct v={data.total_return} digits={2} /></div></div>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="card p-0">
              <div className="card-title p-4 pb-2">Positions</div>
              {data.positions.length === 0 ? <div className="p-4"><Empty>No positions.</Empty></div> : (
                <table className="table-base">
                  <thead><tr><th>Symbol</th><th>Qty</th><th>Avg cost</th><th>Last</th><th>Value</th><th>P&amp;L</th></tr></thead>
                  <tbody>{data.positions.map((p) => (
                    <tr key={p.symbol}><td className="font-semibold">{p.symbol}</td><td>{fmtNum(p.quantity, 3)}</td><td>{fmtNum(p.avg_cost)}</td><td>{fmtNum(p.last_price)}</td><td>{fmtMoney(p.market_value)}</td><td><Pct v={p.unrealized_pnl_pct} digits={2} /></td></tr>
                  ))}</tbody>
                </table>
              )}
            </div>
            <div className="card">
              <div className="card-title">Equity history</div>
              {data.equity_history.length < 2 ? <Empty>Marks accumulate with each rebalance.</Empty> : (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={data.equity_history}>
                    <XAxis dataKey="as_of" tick={{ fontSize: 10, fill: "#7c8596" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#7c8596" }} width={64} domain={["auto", "auto"]} tickFormatter={(v) => fmtMoney(v)} />
                    <Tooltip contentStyle={{ background: "#15181d", border: "1px solid #343a45", fontSize: 12 }} />
                    <Line type="monotone" dataKey="equity" stroke="#3b6ff0" dot strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
          <div className="card p-0">
            <div className="card-title p-4 pb-2">Trades</div>
            <table className="table-base">
              <thead><tr><th>Date</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Price</th><th>Commission</th><th>Reason</th></tr></thead>
              <tbody>{data.trades.map((t, i) => (
                <tr key={i}><td>{t.trade_date}</td><td className={t.side === "BUY" ? "text-bull" : "text-bear"}>{t.side}</td><td>{t.symbol}</td><td>{fmtNum(t.quantity, 3)}</td><td>{fmtNum(t.price)}</td><td>{fmtNum(t.commission)}</td><td className="text-ink-400">{t.reason}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
