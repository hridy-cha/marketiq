"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { api, fetcher, fmtNum, type WatchlistItem } from "@/lib/api";
import { Empty, ErrorBox, Loading, ScoreBar, SignalBadge } from "@/components/ui";

export default function WatchlistPage() {
  const { data, error, isLoading, mutate } = useSWR<{ items: WatchlistItem[] }>("/api/watchlist", fetcher);
  const [symbol, setSymbol] = useState("");
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const add = async () => {
    try {
      await api("/api/watchlist", { method: "POST", body: JSON.stringify({ symbol, note }) });
      setSymbol(""); setNote(""); setMsg(null);
      mutate();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };
  const remove = async (s: string) => {
    await api(`/api/watchlist/${s}`, { method: "DELETE" });
    mutate();
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Watchlist</h1>
      <div className="card flex flex-wrap items-center gap-2">
        <input className="input w-28" placeholder="Symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
        <input className="input w-80" placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
        <button className="btn btn-primary" onClick={add} disabled={!symbol}>Add</button>
        {msg && <span className="text-xs text-bear">{msg}</span>}
      </div>
      {isLoading && <Loading />}
      {error && <ErrorBox error={error} />}
      {data && data.items.length === 0 && <Empty>Nothing on the watchlist yet. Add a symbol from the universe.</Empty>}
      {data && data.items.length > 0 && (
        <div className="card p-0">
          <table className="table-base">
            <thead><tr><th>Symbol</th><th>Score</th><th>Signal</th><th>Close</th><th>Note</th><th>Added</th><th></th></tr></thead>
            <tbody>
              {data.items.map((i) => (
                <tr key={i.symbol}>
                  <td><Link href={`/stocks/${i.symbol}`} className="font-semibold text-accent hover:underline">{i.symbol}</Link></td>
                  <td><ScoreBar score={i.score} /></td>
                  <td><SignalBadge signal={i.signal} /></td>
                  <td>{fmtNum(i.close)}</td>
                  <td className="text-ink-300">{i.note}</td>
                  <td className="text-xs text-ink-500">{i.created_at.slice(0, 10)} · {i.added_by}</td>
                  <td><button className="btn" onClick={() => remove(i.symbol)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
