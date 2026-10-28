// Typed client for the MarketIQ API. All calls go through the Next.js rewrite (/api -> FastAPI).

export type Signal = "BUY" | "SELL" | "HOLD" | "WATCH";

export interface RankingRow {
  rank: number;
  symbol: string;
  name: string;
  sector: string;
  score: number;
  components: Record<string, number | null>;
  signal: Signal | null;
  rule: string | null;
  risk_flags: string[];
  close: number | null;
  ret_1: number | null;
  ret_20: number | null;
  rsi_14: number | null;
  trend: string | null;
  recent_patterns: string[];
}

export interface Rankings {
  as_of: string | null;
  rows: RankingRow[];
  message?: string;
}

export interface Bar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adj_close: number;
  volume: number;
}

export interface Factor {
  name: string;
  value: number | null;
  score: number | null;
  note: string;
}

export interface Component {
  name: string;
  score: number | null;
  weight: number;
  configured_weight: number;
  available: boolean;
  factors: Factor[];
  note: string;
}

export interface PatternHit {
  name: string;
  fired: boolean;
  direction: "bullish" | "bearish" | "neutral";
  last_fired: string | null;
  bars_ago: number | null;
  reason: string;
  recent: boolean;
}

export interface ScoreDetail {
  symbol: string;
  as_of: string;
  score: number;
  components: Record<string, Component>;
  explanation: string[];
  patterns: Record<string, PatternHit>;
  indicators: Record<string, number | null>;
  signal: { signal: Signal; rule: string; reasons: string[]; risk_flags: string[] } | null;
  history: { as_of: string; score: number }[];
}

export interface Indicators {
  symbol: string;
  as_of: string;
  dates: string[];
  series: Record<string, (number | null)[]>;
  latest: Record<string, number | null>;
  patterns: Record<string, PatternHit>;
}

export interface Fundamentals {
  symbol: string;
  as_of: string;
  last_period_end: string | null;
  last_filing_date: string | null;
  quarters_available: number;
  factors: Record<string, number | null>;
  ttm: Record<string, number | null>;
  notes: string[];
  labels: Record<string, string>;
  next_earnings: string | null;
  quarters: { period_end: string; filing_date: string | null; revenue: number | null; net_income: number | null; eps: number | null }[];
}

export interface NewsResponse {
  symbol: string;
  aggregate: { score: number | null; article_count: number; positive: number; negative: number; neutral: number; available: boolean; reason: string };
  articles: { published_at: string; title: string; summary: string; url: string; publisher: string; sentiment_label: string | null; sentiment_score: number | null; source: string }[];
}

export interface WatchlistItem {
  symbol: string;
  note: string;
  added_by: string;
  created_at: string;
  score: number | null;
  signal: Signal | null;
  close: number | null;
}

export interface BacktestSummary {
  id: number;
  name: string;
  kind: string;
  start_date: string;
  end_date: string;
  metrics: Record<string, number | string | null>;
  created_at: string;
  n_trades: number;
}

export interface BacktestDetail extends BacktestSummary {
  config: Record<string, unknown>;
  equity_curve: { date: string; equity: number; cash?: number; exposure?: number; benchmark: number | null }[];
  trades: { symbol: string; side: string; date: string; price: number; quantity: number; notional: number; commission: number; reason: string; pnl: number | null; hold_days: number | null }[];
}

export interface Portfolio {
  account: string;
  cash: number;
  invested: number;
  equity: number;
  starting_cash: number;
  total_return: number;
  positions: { symbol: string; quantity: number; avg_cost: number; last_price: number; market_value: number; unrealized_pnl: number; unrealized_pnl_pct: number }[];
  trades: { symbol: string; side: string; quantity: number; price: number; commission: number; trade_date: string; reason: string }[];
  equity_history: { as_of: string; equity: number; cash: number }[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!res.ok) {
    let detail: string | null = null;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* not JSON: the Next.js proxy could not reach the backend */
    }
    if (detail == null && res.status >= 500) {
      detail = "The MarketIQ backend is not running. Start it with MarketIQ.command (or `marketiq serve`), then refresh this page.";
    }
    throw new ApiError(res.status, detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const fetcher = <T,>(path: string) => api<T>(path);

export const fmtPct = (v: number | null | undefined, digits = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(digits)}%`);
export const fmtNum = (v: number | null | undefined, digits = 2) => (v == null ? "n/a" : v.toFixed(digits));
export const fmtMoney = (v: number | null | undefined) =>
  v == null ? "n/a" : v.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
export const fmtBig = (v: number | null | undefined) => {
  if (v == null) return "n/a";
  const a = Math.abs(v);
  if (a >= 1e12) return `${(v / 1e12).toFixed(2)}T`;
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  return v.toFixed(0);
};
