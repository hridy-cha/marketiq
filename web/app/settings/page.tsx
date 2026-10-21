"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { api, fetcher } from "@/lib/api";
import { ErrorBox, Loading } from "@/components/ui";

interface Config {
  scoring: { weights: Record<string, number>; [k: string]: unknown };
  signals: { thresholds: Record<string, number>; risk_flags: Record<string, number>; rules: Record<string, unknown> };
  files: { scoring: string; signals: string };
}

const DESCRIPTIONS: Record<string, string> = {
  trend: "SMA 20/50/200 alignment, distance from SMA200, SMA50 slope, position in the 52-week range",
  momentum: "RSI vs ideal band, MACD histogram, 20/60-day returns relative to the benchmark",
  volume: "OBV confirmation of the price move, today's volume vs the prior 20-day average",
  volatility: "ATR as % of price, 20-day realized volatility, drawdown from the 52-week high (lower risk scores higher)",
  fundamentals: "Universe percentile ranks of growth, margins, ROE, leverage, liquidity, FCF and valuation",
  sentiment: "Recency-weighted news sentiment (needs stored news)",
  ml: "Probability of a positive forward return from the trained model (only if it passes the validation gate)",
};

export default function SettingsPage() {
  const { data, error, isLoading, mutate } = useSWR<Config>("/api/config", fetcher);
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { if (data) setWeights(Object.fromEntries(Object.entries(data.scoring.weights).map(([k, v]) => [k, Math.round(v * 100)]))); }, [data]);
  const total = Object.values(weights).reduce((a, b) => a + b, 0);

  const save = async () => {
    try {
      await api("/api/config/weights", { method: "PUT", body: JSON.stringify({ weights }) });
      setMsg("Saved. Weights apply on the next scan.");
      mutate();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-ink-400">Scoring weights are stored in <code>{data?.files.scoring}</code>; signal rules in <code>{data?.files.signals}</code>. Weights are normalized to sum to 1; components with no data are dropped per ticker and the rest renormalized.</p>
      </div>
      {isLoading && <Loading />}
      {error && <ErrorBox error={error} />}
      {data && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="card">
            <div className="card-title">Component weights (relative)</div>
            <div className="space-y-3">
              {Object.keys(DESCRIPTIONS).map((k) => (
                <div key={k} className="flex items-center gap-3 text-sm">
                  <span className="w-28 capitalize">{k}</span>
                  <input type="range" min={0} max={50} value={weights[k] ?? 0} onChange={(e) => setWeights({ ...weights, [k]: Number(e.target.value) })} className="flex-1" />
                  <span className="w-14 text-right font-mono">{total ? ((weights[k] ?? 0) / total * 100).toFixed(0) : 0}%</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-3">
              <button className="btn btn-primary" onClick={save} disabled={total === 0}>Save weights</button>
              {msg && <span className="text-xs text-ink-400">{msg}</span>}
            </div>
            <ul className="mt-4 space-y-1 text-xs text-ink-500">
              {Object.entries(DESCRIPTIONS).map(([k, v]) => <li key={k}><span className="capitalize text-ink-300">{k}:</span> {v}</li>)}
            </ul>
          </div>
          <div className="card">
            <div className="card-title">Signal rules (read-only here; edit config/signals.yaml)</div>
            <table className="w-full text-xs"><tbody>
              {Object.entries(data.signals.thresholds).map(([k, v]) => <tr key={k}><td className="py-0.5 text-ink-400">threshold · {k}</td><td className="py-0.5 text-right font-mono">{v}</td></tr>)}
              {Object.entries(data.signals.risk_flags).map(([k, v]) => <tr key={k}><td className="py-0.5 text-ink-400">risk flag · {k}</td><td className="py-0.5 text-right font-mono">{v}</td></tr>)}
            </tbody></table>
            <pre className="mt-3 overflow-auto rounded bg-ink-950 p-2 text-xs text-ink-300">{JSON.stringify(data.signals.rules, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
