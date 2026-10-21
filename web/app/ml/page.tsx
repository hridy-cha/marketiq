"use client";

import useSWR from "swr";
import { fetcher, fmtNum, fmtPct } from "@/lib/api";
import { Empty, ErrorBox, Loading } from "@/components/ui";

interface Fold { fold: number; train_end: string; test_start: string; test_end: string; n_train: number; n_test: number; accuracy: number; balanced_accuracy: number; auc: number | null; top_decile_precision: number | null; top_decile_mean_fwd_return: number | null; all_mean_fwd_return: number; base_rate: number }
interface Models { latest: null | { name: string; version: string; backend: string; horizon_days: number; threshold: number; train_start: string; train_end: string; n_train_rows: number; walk_forward: { folds: Fold[]; pooled: Record<string, number>; feature_importance: Record<string, number> } }; gate: { passed: boolean; message: string }; records: unknown[] }
interface Preds { available: boolean; horizon_days?: number; status: string; gate_status?: string; used_in_scoring?: boolean; rows: { symbol: string; as_of: string; prob_up: number | null }[] }

export default function MLPage() {
  const models = useSWR<Models>("/api/ml/models", fetcher);
  const preds = useSWR<Preds>("/api/ml/predictions", fetcher);
  const m = models.data?.latest;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold">ML direction model</h1>
        <p className="text-sm text-ink-400">Gradient-boosted classifier predicting whether the forward return over the horizon is positive, evaluated walk-forward with a gap so no label overlaps the test window. A model only feeds the score if it passes the validation gate in <code>config/scoring.yaml</code>.</p>
      </div>
      {models.isLoading && <Loading />}
      {models.error && <ErrorBox error={models.error} />}
      {models.data && !m && <Empty>No trained model. Run <code>marketiq train</code>.</Empty>}
      {m && (
        <>
          <div className={`card border ${models.data!.gate.passed ? "border-bull/40" : "border-watch/40"}`}>
            <div className="card-title">Validation gate</div>
            <div className={`text-sm ${models.data!.gate.passed ? "text-bull" : "text-watch"}`}>{models.data!.gate.message}</div>
            <div className="mt-2 text-xs text-ink-400">{m.name} {m.version} · {m.backend} · horizon {m.horizon_days} trading days · threshold {m.threshold} · trained on {m.n_train_rows.toLocaleString()} rows {m.train_start} → {m.train_end}</div>
          </div>
          <div className="card overflow-x-auto p-0">
            <div className="card-title p-4 pb-2">Walk-forward folds (out of sample)</div>
            <table className="table-base">
              <thead><tr><th>Fold</th><th>Train end</th><th>Test window</th><th>n test</th><th>Accuracy</th><th>Bal. acc.</th><th>AUC</th><th>Top-decile precision</th><th>Top-decile fwd return</th><th>All-rows fwd return</th><th>Base rate</th></tr></thead>
              <tbody>
                {m.walk_forward.folds.map((f) => (
                  <tr key={f.fold}><td>{f.fold}</td><td>{f.train_end}</td><td>{f.test_start} → {f.test_end}</td><td>{f.n_test}</td><td>{fmtNum(f.accuracy, 3)}</td><td>{fmtNum(f.balanced_accuracy, 3)}</td><td>{fmtNum(f.auc, 3)}</td><td>{fmtNum(f.top_decile_precision, 3)}</td><td>{fmtPct(f.top_decile_mean_fwd_return, 2)}</td><td>{fmtPct(f.all_mean_fwd_return, 2)}</td><td>{fmtNum(f.base_rate, 3)}</td></tr>
                ))}
                <tr className="font-semibold"><td colSpan={3}>Pooled</td><td>{m.walk_forward.pooled.n_test_rows}</td><td>{fmtNum(m.walk_forward.pooled.accuracy, 3)}</td><td>{fmtNum(m.walk_forward.pooled.balanced_accuracy, 3)}</td><td>{fmtNum(m.walk_forward.pooled.auc, 3)}</td><td>{fmtNum(m.walk_forward.pooled.top_decile_precision, 3)}</td><td>{fmtPct(m.walk_forward.pooled.top_decile_mean_fwd_return, 2)}</td><td>{fmtPct(m.walk_forward.pooled.all_mean_fwd_return, 2)}</td><td>{fmtNum(m.walk_forward.pooled.base_rate, 3)}</td></tr>
              </tbody>
            </table>
            <p className="p-4 text-xs text-ink-500">An AUC near 0.5 and balanced accuracy near 0.5 mean the model is not separating up from down moves out of sample. Compare the top-decile forward return with the all-rows forward return: if the top decile does not beat it, the ranking carries no information.</p>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="card">
              <div className="card-title">Feature importance (normalized)</div>
              <table className="w-full text-xs"><tbody>
                {Object.entries(m.walk_forward.feature_importance).slice(0, 15).map(([k, v]) => (
                  <tr key={k}><td className="py-0.5 text-ink-400">{k}</td><td className="w-40 py-0.5"><div className="h-1.5 rounded bg-ink-800"><div className="h-full rounded bg-accent" style={{ width: `${v * 100 * 4}%` }} /></div></td><td className="py-0.5 text-right font-mono">{(v * 100).toFixed(1)}%</td></tr>
                ))}
              </tbody></table>
            </div>
            <div className="card">
              <div className="card-title">Current predictions {preds.data?.available && !preds.data.used_in_scoring && <span className="ml-2 rounded bg-watch/20 px-1 text-watch">not used in scoring (gate failed)</span>}</div>
              {preds.isLoading && <Loading />}
              {preds.data && !preds.data.available && <Empty>{preds.data.status}</Empty>}
              {preds.data?.available && (
                <table className="w-full text-xs"><thead><tr className="text-ink-500"><th className="text-left">Symbol</th><th className="text-right">P(up over {preds.data.horizon_days}d)</th></tr></thead><tbody>
                  {preds.data.rows.map((r) => <tr key={r.symbol}><td>{r.symbol}</td><td className="text-right font-mono">{fmtNum(r.prob_up, 3)}</td></tr>)}
                </tbody></table>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
