"use client";

import { useState } from "react";
import useSWR from "swr";
import dynamic from "next/dynamic";
import { api, fetcher, fmtBig, fmtNum, fmtPct, type Bar, type Fundamentals, type Indicators, type NewsResponse, type ScoreDetail } from "@/lib/api";
import { Empty, ErrorBox, Loading, Pct, ScoreBar, SignalBadge } from "@/components/ui";
import type { ChartOverlays } from "@/components/PriceChart";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const PriceChart = dynamic(() => import("@/components/PriceChart"), { ssr: false });

export default function StockPage({ params }: { params: { symbol: string } }) {
  const symbol = params.symbol.toUpperCase();
  const bars = useSWR<{ symbol: string; bars: Bar[] }>(`/api/stocks/${symbol}/bars?limit=750`, fetcher);
  const ind = useSWR<Indicators>(`/api/stocks/${symbol}/indicators?limit=750`, fetcher);
  const score = useSWR<ScoreDetail>(`/api/stocks/${symbol}/score`, fetcher, { shouldRetryOnError: false });
  const fund = useSWR<Fundamentals>(`/api/stocks/${symbol}/fundamentals`, fetcher);
  const news = useSWR<NewsResponse>(`/api/stocks/${symbol}/news?days=30`, fetcher);
  const [overlays, setOverlays] = useState<ChartOverlays>({ sma20: true, sma50: true, sma200: true, bollinger: false });
  const [watchMsg, setWatchMsg] = useState<string | null>(null);

  const latest = ind.data?.latest;
  const addToWatchlist = async () => {
    try {
      await api("/api/watchlist", { method: "POST", body: JSON.stringify({ symbol }) });
      setWatchMsg("Added to watchlist");
    } catch (e) {
      setWatchMsg(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{symbol}</h1>
          <p className="text-sm text-ink-400">
            {latest && <>Close <span className="font-mono text-ink-100">{fmtNum(latest.close)}</span> as of {ind.data?.as_of} · 1d <Pct v={latest.ret_1} /> · 20d <Pct v={latest.ret_20} /> · 52w high {fmtNum(latest.high_52w)} · 52w low {fmtNum(latest.low_52w)}</>}
          </p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {score.data && <ScoreBar score={score.data.score} />}
          {score.data?.signal && <SignalBadge signal={score.data.signal.signal} />}
          <button className="btn" onClick={addToWatchlist}>+ Watchlist</button>
          {watchMsg && <span className="text-xs text-ink-400">{watchMsg}</span>}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card lg:col-span-2">
          <div className="mb-2 flex items-center gap-3 text-xs text-ink-400">
            <span className="card-title mb-0">Price &amp; indicators</span>
            {(["sma20", "sma50", "sma200", "bollinger"] as const).map((k) => (
              <label key={k} className="flex items-center gap-1">
                <input type="checkbox" checked={overlays[k]} onChange={(e) => setOverlays({ ...overlays, [k]: e.target.checked })} />
                {k.toUpperCase()}
              </label>
            ))}
          </div>
          {bars.isLoading && <Loading what="chart" />}
          {bars.error && <ErrorBox error={bars.error} />}
          {bars.data && <PriceChart bars={bars.data.bars} indicators={ind.data ?? null} overlays={overlays} />}
        </div>

        <div className="space-y-4">
          <div className="card">
            <div className="card-title">Signal</div>
            {score.error && <Empty>No scan snapshot yet. Run a scan from the Rankings page.</Empty>}
            {score.data?.signal && (
              <>
                <div className="mb-2 flex items-center gap-2"><SignalBadge signal={score.data.signal.signal} /><span className="font-mono text-xs text-ink-400">{score.data.signal.rule}</span></div>
                <ul className="list-disc space-y-1 pl-4 text-sm text-ink-200">
                  {score.data.signal.reasons.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
                {score.data.signal.risk_flags.length > 0 && (
                  <div className="mt-2 text-xs text-watch">Risk flags: {score.data.signal.risk_flags.join("; ")}</div>
                )}
              </>
            )}
          </div>
          <div className="card">
            <div className="card-title">Score breakdown</div>
            {score.data ? <Breakdown s={score.data} /> : <span className="text-sm text-ink-500">n/a</span>}
          </div>
          {score.data && score.data.history.length > 1 && (
            <div className="card">
              <div className="card-title">Score history</div>
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={score.data.history}>
                  <XAxis dataKey="as_of" hide />
                  <YAxis domain={[0, 100]} width={28} tick={{ fontSize: 10, fill: "#7c8596" }} />
                  <Tooltip contentStyle={{ background: "#15181d", border: "1px solid #343a45", fontSize: 12 }} />
                  <Line type="monotone" dataKey="score" stroke="#3b6ff0" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card">
          <div className="card-title">Technical snapshot</div>
          {latest ? <Snapshot latest={latest} /> : <Loading />}
          {ind.data && (
            <>
              <div className="card-title mt-4">Patterns</div>
              <ul className="space-y-1 text-xs">
                {Object.values(ind.data.patterns).map((p) => (
                  <li key={p.name} className={p.recent ? "text-ink-100" : "text-ink-500"}>
                    <span className={`mr-1 inline-block w-2 ${p.direction === "bullish" ? "text-bull" : p.direction === "bearish" ? "text-bear" : "text-ink-500"}`}>●</span>
                    <span className="font-medium">{p.name.replace(/_/g, " ")}</span>
                    {p.bars_ago != null && <span className="ml-1 text-ink-500">({p.bars_ago}d ago)</span>}
                    <div className="ml-3 text-ink-400">{p.reason}</div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
        <div className="card">
          <div className="card-title">Fundamentals</div>
          {fund.isLoading && <Loading />}
          {fund.error && <ErrorBox error={fund.error} />}
          {fund.data && <FundamentalsCard f={fund.data} />}
        </div>
        <div className="card">
          <div className="card-title">News &amp; sentiment (30 days)</div>
          {news.isLoading && <Loading />}
          {news.data && (
            <>
              <div className="mb-2 text-xs text-ink-400">{news.data.aggregate.reason}{news.data.aggregate.available && <> · score {fmtNum(news.data.aggregate.score)}</>}</div>
              {news.data.articles.length === 0 && <Empty>No stored news. Run <code>marketiq fetch --news</code> on a machine with internet access.</Empty>}
              <ul className="space-y-2 text-sm">
                {news.data.articles.slice(0, 15).map((a, i) => (
                  <li key={i}>
                    <a href={a.url} target="_blank" rel="noreferrer" className="hover:underline">{a.title}</a>
                    <div className="text-xs text-ink-500">
                      {a.published_at.slice(0, 10)} · {a.publisher}
                      {a.sentiment_label && <span className={`ml-2 ${a.sentiment_label === "positive" ? "text-bull" : a.sentiment_label === "negative" ? "text-bear" : "text-ink-400"}`}>{a.sentiment_label} {fmtNum(a.sentiment_score)}</span>}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>

      {score.data && (
        <div className="card">
          <div className="card-title">Full explanation</div>
          <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-200">
            {score.data.explanation.map((l, i) => <li key={i}>{l}</li>)}
          </ol>
        </div>
      )}
    </div>
  );
}

function Breakdown({ s }: { s: ScoreDetail }) {
  const comps = Object.values(s.components).sort((a, b) => b.configured_weight - a.configured_weight);
  return (
    <div className="space-y-2">
      {comps.map((c) => (
        <details key={c.name} className="group">
          <summary className="flex cursor-pointer items-center gap-2 text-sm">
            <span className="w-24 capitalize">{c.name}</span>
            {c.available ? <ScoreBar score={c.score} small /> : <span className="text-xs text-ink-500">unavailable</span>}
            <span className="ml-auto text-xs text-ink-500">w {(c.weight * 100).toFixed(0)}%{!c.available && ` (cfg ${(c.configured_weight * 100).toFixed(0)}%)`}</span>
          </summary>
          <ul className="ml-2 mt-1 space-y-0.5 border-l border-ink-800 pl-3 text-xs text-ink-400">
            {!c.available && <li>{c.note}</li>}
            {c.factors.map((f) => (
              <li key={f.name}><span className="text-ink-200">{f.name.replace(/_/g, " ")}</span> {f.score != null && <span className="font-mono">{f.score.toFixed(0)}</span>} · {f.note}</li>
            ))}
          </ul>
        </details>
      ))}
    </div>
  );
}

function Snapshot({ latest }: { latest: Record<string, number | null> }) {
  const rows: [string, string][] = [
    ["RSI 14", fmtNum(latest.rsi_14, 1)],
    ["MACD / signal", `${fmtNum(latest.macd)} / ${fmtNum(latest.macd_signal)}`],
    ["SMA 20 / 50 / 200", `${fmtNum(latest.sma_20)} / ${fmtNum(latest.sma_50)} / ${fmtNum(latest.sma_200)}`],
    ["ATR 14 (% of price)", `${fmtNum(latest.atr_14)} (${fmtNum(latest.atr_pct)}%)`],
    ["Volume vs 20d avg", `${fmtNum(latest.volume_ratio)}x`],
    ["Realized vol 20d", fmtPct(latest.realized_vol_20, 0)],
    ["From 52w high / low", `${fmtNum(latest.pct_from_52w_high, 1)}% / ${fmtNum(latest.pct_from_52w_low, 1)}%`],
    ["Stretch from SMA20", `${fmtNum(latest.stretch_atr, 1)} ATR`],
    ["Bollinger", `${fmtNum(latest.bb_lower)} – ${fmtNum(latest.bb_upper)}`],
  ];
  return (
    <table className="w-full text-xs">
      <tbody>{rows.map(([k, v]) => <tr key={k}><td className="py-0.5 text-ink-400">{k}</td><td className="py-0.5 text-right font-mono text-ink-100">{v}</td></tr>)}</tbody>
    </table>
  );
}

function FundamentalsCard({ f }: { f: Fundamentals }) {
  const pct = (k: string) => k.endsWith("growth_yoy") || k.endsWith("margin") || k === "roe" || k.endsWith("yield");
  const order = ["revenue_growth_yoy", "eps_growth_yoy", "gross_margin", "operating_margin", "net_margin", "roe", "debt_to_equity",
    "current_ratio", "fcf_per_share", "fcf_yield", "pe_ratio", "price_to_book", "price_to_sales", "peg", "dividend_yield"];
  return (
    <div className="space-y-3 text-xs">
      <div className="text-ink-400">
        Last period {f.last_period_end ?? "n/a"} · filed {f.last_filing_date ?? "n/a"} · {f.quarters_available} quarters
        {f.next_earnings && <> · next earnings <span className="text-watch">{f.next_earnings}</span></>}
      </div>
      {f.notes.length > 0 && <div className="text-ink-500">{f.notes.join(" · ")}</div>}
      <table className="w-full">
        <tbody>
          {order.filter((k) => k in f.factors).map((k) => (
            <tr key={k}><td className="py-0.5 text-ink-400">{f.labels[k] ?? k}</td>
              <td className="py-0.5 text-right font-mono text-ink-100">{f.factors[k] == null ? "n/a" : pct(k) ? fmtPct(f.factors[k], 1) : fmtNum(f.factors[k], 2)}</td></tr>
          ))}
        </tbody>
      </table>
      {f.quarters.length > 0 && (
        <>
          <div className="card-title mt-2">Recent quarters</div>
          <table className="w-full">
            <thead><tr className="text-ink-500"><th className="text-left">Period</th><th className="text-right">Revenue</th><th className="text-right">Net inc.</th><th className="text-right">EPS</th></tr></thead>
            <tbody>{f.quarters.slice().reverse().map((q) => (
              <tr key={q.period_end}><td>{q.period_end}</td><td className="text-right font-mono">{fmtBig(q.revenue)}</td><td className="text-right font-mono">{fmtBig(q.net_income)}</td><td className="text-right font-mono">{fmtNum(q.eps)}</td></tr>
            ))}</tbody>
          </table>
        </>
      )}
    </div>
  );
}
