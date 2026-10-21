"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  CrosshairMode,
  IChartApi,
  ISeriesApi,
  LineStyle,
  createChart,
  type Time,
} from "lightweight-charts";
import type { Bar, Indicators } from "@/lib/api";

const COLORS = {
  bg: "#15181d",
  grid: "#22262e",
  text: "#a9b0bf",
  up: "#1f9d55",
  down: "#d64545",
  sma20: "#f2c14e",
  sma50: "#3b6ff0",
  sma200: "#c084fc",
  bb: "rgba(169,176,191,0.35)",
  macd: "#3b6ff0",
  signal: "#f2c14e",
  rsi: "#60a5fa",
};

function pairs(dates: string[], values: (number | null)[]) {
  const out: { time: Time; value: number }[] = [];
  for (let i = 0; i < dates.length; i++) {
    const v = values[i];
    if (v != null && Number.isFinite(v)) out.push({ time: dates[i] as Time, value: v });
  }
  return out;
}

function baseOptions(height: number, width: number) {
  return {
    height,
    width,
    // Pin the locale: lightweight-charts formats with Intl using navigator.language, which can be an
    // invalid tag on some systems and would otherwise throw during rendering.
    localization: { locale: "en-US" },
    layout: { background: { type: ColorType.Solid, color: COLORS.bg }, textColor: COLORS.text, fontSize: 11 },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderColor: COLORS.grid },
    timeScale: { borderColor: COLORS.grid, timeVisible: false },
    crosshair: { mode: CrosshairMode.Normal },
  };
}

export interface ChartOverlays {
  sma20: boolean;
  sma50: boolean;
  sma200: boolean;
  bollinger: boolean;
}

export default function PriceChart({
  bars,
  indicators,
  overlays,
}: {
  bars: Bar[];
  indicators: Indicators | null;
  overlays: ChartOverlays;
}) {
  const priceRef = useRef<HTMLDivElement>(null);
  const volRef = useRef<HTMLDivElement>(null);
  const rsiRef = useRef<HTMLDivElement>(null);
  const macdRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!priceRef.current || !volRef.current || !rsiRef.current || !macdRef.current || bars.length === 0) return;
    const charts: IChartApi[] = [];
    const width = priceRef.current.clientWidth || 800;
    const price = createChart(priceRef.current, baseOptions(360, width));
    const vol = createChart(volRef.current, baseOptions(100, width));
    const rsi = createChart(rsiRef.current, baseOptions(120, width));
    const macd = createChart(macdRef.current, baseOptions(120, width));
    charts.push(price, vol, rsi, macd);

    const candles = price.addCandlestickSeries({
      upColor: COLORS.up, downColor: COLORS.down, borderVisible: false, wickUpColor: COLORS.up, wickDownColor: COLORS.down,
    });
    candles.setData(bars.map((b) => ({ time: b.date as Time, open: b.open, high: b.high, low: b.low, close: b.close })));

    const volume = vol.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "right" });
    volume.setData(bars.map((b, i) => ({
      time: b.date as Time, value: b.volume,
      color: i > 0 && b.close >= bars[i - 1].close ? "rgba(31,157,85,0.6)" : "rgba(214,69,69,0.6)",
    })));

    const lines: ISeriesApi<"Line">[] = [];
    if (indicators) {
      const { dates, series } = indicators;
      const addLine = (chart: IChartApi, key: string, color: string, width: 1 | 2 = 1, style = LineStyle.Solid) => {
        if (!series[key]) return;
        const s = chart.addLineSeries({ color, lineWidth: width, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        s.setData(pairs(dates, series[key]));
        lines.push(s);
      };
      if (overlays.sma20) addLine(price, "sma_20", COLORS.sma20);
      if (overlays.sma50) addLine(price, "sma_50", COLORS.sma50);
      if (overlays.sma200) addLine(price, "sma200" in series ? "sma200" : "sma_200", COLORS.sma200, 2);
      if (overlays.bollinger) {
        addLine(price, "bb_upper", COLORS.bb, 1, LineStyle.Dotted);
        addLine(price, "bb_lower", COLORS.bb, 1, LineStyle.Dotted);
      }
      const volSma = vol.addLineSeries({ color: COLORS.sma20, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
      volSma.setData(pairs(dates, series["volume_sma_20"] || []));

      const rsiLine = rsi.addLineSeries({ color: COLORS.rsi, lineWidth: 1, priceLineVisible: false });
      rsiLine.setData(pairs(dates, series["rsi_14"] || []));
      for (const lvl of [30, 50, 70]) {
        rsiLine.createPriceLine({ price: lvl, color: lvl === 50 ? COLORS.grid : "rgba(217,162,27,0.5)", lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: "" });
      }
      rsi.priceScale("right").applyOptions({ autoScale: false });
      rsiLine.applyOptions({ autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }) });

      const hist = macd.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
      const h = series["macd_hist"] || [];
      hist.setData(pairs(dates, h).map((p) => ({ ...p, color: p.value >= 0 ? "rgba(31,157,85,0.7)" : "rgba(214,69,69,0.7)" })));
      const macdLine = macd.addLineSeries({ color: COLORS.macd, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
      macdLine.setData(pairs(dates, series["macd"] || []));
      const sigLine = macd.addLineSeries({ color: COLORS.signal, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
      sigLine.setData(pairs(dates, series["macd_signal"] || []));
    }

    // Sync the visible range across panels.
    let syncing = false;
    for (const c of charts) {
      c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (syncing || !range) return;
        syncing = true;
        for (const other of charts) if (other !== c) other.timeScale().setVisibleLogicalRange(range);
        syncing = false;
      });
    }
    price.timeScale().setVisibleLogicalRange({ from: Math.max(0, bars.length - 180), to: bars.length + 2 });

    const ro = new ResizeObserver(() => {
      const w = priceRef.current?.clientWidth ?? 0;
      for (const c of charts) c.applyOptions({ width: w });
    });
    ro.observe(priceRef.current);
    return () => {
      ro.disconnect();
      for (const c of charts) c.remove();
    };
  }, [bars, indicators, overlays]);

  return (
    <div className="space-y-1">
      <div className="flex gap-4 px-1 text-xs text-ink-400">
        <span><span style={{ color: COLORS.sma20 }}>■</span> SMA20</span>
        <span><span style={{ color: COLORS.sma50 }}>■</span> SMA50</span>
        <span><span style={{ color: COLORS.sma200 }}>■</span> SMA200</span>
        <span><span style={{ color: COLORS.bb }}>■</span> Bollinger</span>
      </div>
      <div ref={priceRef} />
      <div className="px-1 text-xs text-ink-400">Volume (line = 20-day average)</div>
      <div ref={volRef} />
      <div className="px-1 text-xs text-ink-400">RSI 14 (dashed 30 / 70)</div>
      <div ref={rsiRef} />
      <div className="px-1 text-xs text-ink-400">MACD 12/26/9 (histogram = MACD minus signal)</div>
      <div ref={macdRef} />
    </div>
  );
}
