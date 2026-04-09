"use client";

/**
 * Professor-facing interactive demo of the AdaptiveBeta strategy.
 * Runs a full browser-side simulation (~2520 days) and re-computes
 * metrics in <10 ms whenever the user adjusts any parameter slider.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Legend,
} from "recharts";
import {
  CRISIS_DEFINITIONS,
  runSimulation,
  runRealSimulation,
  getCrisisWindow,
  type SimParams,
  type SimResult,
  type SimPoint,
  type RealDemoData,
} from "@/lib/demoSimulation";
import { cn } from "@/lib/utils";

// ─── Default parameters ───────────────────────────────────────────────────────

const DEFAULT_PARAMS: SimParams = {
  vixThreshold:      22,   // matches src/config.py VIX_STRESS_LEVEL (fixed from 25)
  betavolPercentile: 70,   // matches src/config.py THRESHOLD_QUANTILE=0.70 (fixed from 0.55)
  regimeRouting:     true,
  txCostBps:         8,
};

// ─── Animated Counter ─────────────────────────────────────────────────────────

function useAnimatedValue(target: number, decimals = 2): string {
  const [current, setCurrent] = useState(target);
  const raf    = useRef<number | null>(null);
  const prev   = useRef(target);
  const start  = useRef(0);

  useEffect(() => {
    const from    = prev.current;
    const to      = target;
    const dur     = 380;
    start.current = performance.now();

    const step = (now: number) => {
      const t = Math.min(1, (now - start.current) / dur);
      const ease = 1 - Math.pow(1 - t, 3);
      setCurrent(from + (to - from) * ease);
      if (t < 1) raf.current = requestAnimationFrame(step);
      else prev.current = to;
    };

    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [target]);

  return current.toFixed(decimals);
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricTile({
  label,
  value,
  suffix = "",
  decimals = 2,
  good,
  size = "md",
}: {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  good?: boolean;
  size?: "sm" | "md" | "lg";
}) {
  const displayed = useAnimatedValue(value, decimals);
  const color =
    good === true  ? "text-teal-300" :
    good === false ? "text-coral-300" :
                     "text-white";
  const textSize =
    size === "lg" ? "text-3xl md:text-4xl" :
    size === "sm" ? "text-lg" :
                    "text-2xl";

  return (
    <div className="flex flex-col items-center gap-1">
      <span className={cn("font-bold font-mono tabular-nums", textSize, color)}>
        {displayed}{suffix}
      </span>
      <span className="text-xs text-gray-500 text-center leading-tight">{label}</span>
    </div>
  );
}

// ─── Parameter Slider ─────────────────────────────────────────────────────────

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  color = "#1D9E75",
  presets,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  color?: string;
  presets?: { label: string; value: number }[];
}) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-300">{label}</span>
        <span className="text-xs font-mono font-semibold" style={{ color }}>
          {format(value)}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, ${color} ${pct}%, rgba(255,255,255,0.1) ${pct}%)`,
            outline: "none",
          }}
        />
      </div>
      {presets && (
        <div className="flex gap-1.5 flex-wrap">
          {presets.map((p) => (
            <button
              key={p.label}
              onClick={() => onChange(p.value)}
              className={cn(
                "text-xs px-2 py-0.5 rounded border transition-all",
                Math.abs(value - p.value) < step * 0.5
                  ? "border-teal-400/60 text-teal-300 bg-teal-400/10"
                  : "border-navy-600 text-gray-500 hover:border-navy-500"
              )}
            >
              {p.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Signal Pill ──────────────────────────────────────────────────────────────

function SignalPill({ signal }: { signal: string }) {
  if (signal === "min_variance")
    return <span className="inline-block w-2 h-2 rounded-full bg-coral-400" title="MIN_VARIANCE" />;
  if (signal === "rebalance")
    return <span className="inline-block w-2 h-2 rounded-full bg-amber-400" title="REBALANCE" />;
  return <span className="inline-block w-1.5 h-1.5 rounded-full bg-navy-600" title="HOLD" />;
}

// ─── Signal Timeline ──────────────────────────────────────────────────────────

function SignalTimeline({ fullResults }: { fullResults: SimPoint[] }) {
  const step = Math.max(1, Math.floor(fullResults.length / 220));
  const ticks = fullResults.filter((_, i) => i % step === 0);

  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-gray-400">
        Signal Timeline —{" "}
        <span className="text-teal-300">
          {fullResults.filter((r) => r.signal === "hold").length} hold
        </span>
        {" · "}
        <span className="text-amber-300">
          {fullResults.filter((r) => r.signal === "rebalance").length} rebalance
        </span>
        {" · "}
        <span className="text-coral-300">
          {fullResults.filter((r) => r.signal === "min_variance").length} min-var
        </span>
      </p>
      <div className="flex flex-wrap gap-px">
        {ticks.map((r) => (
          <SignalPill key={r.date} signal={r.signal} />
        ))}
      </div>
      {(() => {
        const startY = parseInt(fullResults[0]?.date?.substring(0, 4) ?? "2015");
        const endY   = parseInt(fullResults[fullResults.length - 1]?.date?.substring(0, 4) ?? "2024");
        const labels = Array.from({ length: 6 }, (_, i) =>
          Math.round(startY + (i * (endY - startY)) / 5)
        );
        return (
          <div className="flex justify-between text-xs text-gray-600">
            {labels.map((y) => <span key={y}>{y}</span>)}
          </div>
        );
      })()}
    </div>
  );
}

// ─── Crisis Comparison Panel ──────────────────────────────────────────────────

function CrisisPanel({
  result,
  activeCrisis,
  onSelect,
}: {
  result: SimResult;
  activeCrisis: string | null;
  onSelect: (name: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {Object.entries(CRISIS_DEFINITIONS).map(([name, def]) => {
        const stats  = result.crisisStats[name];
        const active = activeCrisis === name;
        const prot   = stats?.protection ?? 0;
        const isGood = prot >= 0;

        return (
          <button
            key={name}
            onClick={() => onSelect(active ? "" : name)}
            className={cn(
              "text-left p-3 rounded-lg border transition-all duration-200",
              active
                ? "border-teal-400/50 bg-teal-400/8"
                : "border-navy-700 hover:border-navy-600 bg-navy-800/40"
            )}
            style={active ? { borderColor: def.color + "60" } : {}}
          >
            <div className="flex items-center gap-1.5 mb-1.5">
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: def.color }}
              />
              <span className="text-xs font-semibold text-white leading-tight">
                {name}
              </span>
            </div>
            {stats && (
              <div className="space-y-0.5">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Portfolio</span>
                  <span className={stats.portfolioReturn < 0 ? "text-coral-300" : "text-teal-300"}>
                    {stats.portfolioReturn.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">NIFTY50</span>
                  <span className={stats.niftyReturn < 0 ? "text-coral-300" : "text-teal-300"}>
                    {stats.niftyReturn.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Protection</span>
                  <span className={isGood ? "text-teal-300 font-semibold" : "text-coral-300"}>
                    {prot >= 0 ? "+" : ""}{prot.toFixed(1)}%
                  </span>
                </div>
                {stats.minVarDays > 0 && (
                  <div className="text-xs text-gray-600 mt-1">
                    {stats.minVarDays}d min-var · {stats.rebalances} rebalances
                  </div>
                )}
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ─── Chart Tooltip ────────────────────────────────────────────────────────────

function DemoTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string; dataKey: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const portfolioEntry = payload.find((p) => p.dataKey === "portfolio");
  const niftyEntry     = payload.find((p) => p.dataKey === "nifty");
  const item           = payload[0];

  return (
    <div className="bg-navy-800/95 border border-navy-600 rounded-lg p-3 shadow-xl text-xs min-w-[170px] backdrop-blur-sm">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      {portfolioEntry && (
        <div className="flex justify-between gap-4 mb-1">
          <span className="text-teal-300">AdaptiveBeta</span>
          <span className="font-mono font-semibold text-white">{portfolioEntry.value.toFixed(1)}</span>
        </div>
      )}
      {niftyEntry && (
        <div className="flex justify-between gap-4 mb-1">
          <span className="text-amber-300">NIFTY50</span>
          <span className="font-mono text-gray-300">{niftyEntry.value.toFixed(1)}</span>
        </div>
      )}
      {item && "beta" in (item as any) && (
        <div className="flex justify-between gap-4 text-gray-500 mt-1 pt-1 border-t border-navy-700">
          <span>β</span>
          <span className="font-mono">{(item as any).beta}</span>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function InteractiveDemo() {
  const [params,       setParams]       = useState<SimParams>(DEFAULT_PARAMS);
  const [activeCrisis, setActiveCrisis] = useState<string | null>(null);
  const [realData,     setRealData]     = useState<RealDemoData | null>(null);
  const [dataLoading,  setDataLoading]  = useState(true);

  // Attempt to load real backtest data generated by scripts/05_export_demo_data.py
  useEffect(() => {
    fetch("/demo-data.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.source === "real" && Array.isArray(d?.dates) && d.dates.length > 0) {
          setRealData(d as RealDemoData);
          // Sync slider defaults to the real backtest configuration
          setParams((p) => ({
            ...p,
            vixThreshold:      d.meta.vixThreshold ?? p.vixThreshold,
            betavolPercentile: Math.round((d.meta.betavolQuantile ?? 0.70) * 100),
          }));
        }
      })
      .catch(() => {})
      .finally(() => setDataLoading(false));
  }, []);

  const result = useMemo<SimResult>(
    () => realData ? runRealSimulation(realData, params) : runSimulation(params),
    [realData, params]
  );

  const crisisData = useMemo<SimPoint[]>(
    () => (activeCrisis ? getCrisisWindow(result.fullResults, activeCrisis) : []),
    [activeCrisis, result.fullResults]
  );

  const chartData = activeCrisis && crisisData.length > 0 ? crisisData : result.curve;
  const crisisDef = activeCrisis ? CRISIS_DEFINITIONS[activeCrisis] : null;

  const updateParam = useCallback(
    <K extends keyof SimParams>(key: K, value: SimParams[K]) =>
      setParams((p) => ({ ...p, [key]: value })),
    []
  );

  const { metrics, niftyMetrics } = result;

  const cagrDelta  = metrics.cagr  - niftyMetrics.cagr;
  const sharpeDelta = metrics.sharpe - niftyMetrics.sharpe;
  const ddDelta    = metrics.maxDrawdown - niftyMetrics.maxDrawdown;

  // Insight text
  const insight = useMemo(() => {
    const dataLabel = realData ? "real India VIX + betavol data" : "synthetic data";
    if (params.vixThreshold < 20) {
      return `Low VIX threshold: strategy goes defensive very frequently using ${dataLabel}. Lots of min-variance days → lower volatility but also lower upside capture.`;
    } else if (params.vixThreshold > 32) {
      return `High VIX threshold on ${dataLabel}: only extreme crises (like COVID at VIX 80) trigger the override. ADANI and rate hikes may slip through without protection.`;
    } else if (params.betavolPercentile < 60) {
      return `Low betavol percentile on ${dataLabel}: frequent rebalances → higher transaction cost drag. Good adaptability, but cost accumulates over the full period.`;
    } else if (params.betavolPercentile > 80) {
      return `High betavol percentile on ${dataLabel}: conservative triggering. Very few rebalances → low cost, but slower to adapt to changing regimes.`;
    } else if (!params.regimeRouting) {
      return "Regime routing OFF: all rebalances target the same portfolio weights (β=0.82). Misses the bull-market upside from aggressive positioning.";
    }
    const src = realData ? `real backtest (${realData.meta.startDate.substring(0, 4)}–${realData.meta.endDate.substring(0, 4)})` : "10-year simulation";
    return `Balanced configuration on ${src}: VIX override at ${params.vixThreshold} captures major crises, ${params.betavolPercentile}th percentile threshold triggers ${result.rebalanceCount} rebalances.`;
  }, [params, result.rebalanceCount, realData]);

  return (
    <div className="space-y-6">

      {/* ── Top: Params + Big Metrics ── */}
      <div className="grid lg:grid-cols-[300px_1fr] gap-5">

        {/* Parameter Panel */}
        <div
          className="rounded-xl p-5 space-y-5"
          style={{
            background: "linear-gradient(155deg, rgba(15,22,41,0.95) 0%, rgba(20,29,58,0.85) 100%)",
            border: "1px solid rgba(29,158,117,0.22)",
            boxShadow: "0 0 40px rgba(29,158,117,0.06), inset 0 1px 0 rgba(255,255,255,0.04)",
            backdropFilter: "blur(20px)",
          }}
        >
          <div>
            <p className="text-xs font-semibold tracking-widest uppercase text-teal-400 mb-0.5">
              Parameter Controls
            </p>
            <p className="text-xs text-gray-500">
              Adjust any slider — metrics update live
            </p>
            {!dataLoading && (
              <div className={cn(
                "flex items-center gap-1.5 mt-2 px-2 py-1 rounded-md text-xs font-medium",
                realData
                  ? "bg-teal-400/10 border border-teal-400/30 text-teal-300"
                  : "bg-navy-700/50 border border-navy-600/50 text-gray-400"
              )}>
                <span className={cn(
                  "w-1.5 h-1.5 rounded-full flex-shrink-0",
                  realData ? "bg-teal-400 animate-pulse" : "bg-gray-500"
                )} />
                {realData
                  ? `Real Backtest · ${realData.meta.startDate.substring(0, 4)}–${realData.meta.endDate.substring(0, 4)}`
                  : "Synthetic Demo Data"}
              </div>
            )}
          </div>

          <ParamSlider
            label="VIX Override Threshold"
            value={params.vixThreshold}
            min={15} max={40} step={1}
            color="#D85A30"
            format={(v) => `VIX > ${v}`}
            onChange={(v) => updateParam("vixThreshold", v)}
            presets={[
              { label: "Sensitive (18)", value: 18 },
              { label: "Default (22)",   value: 22 },
              { label: "Conservative (30)", value: 30 },
            ]}
          />

          <ParamSlider
            label="BetaVol Percentile Threshold"
            value={params.betavolPercentile}
            min={40} max={90} step={5}
            color="#EF9F27"
            format={(v) => `${v}th pct`}
            onChange={(v) => updateParam("betavolPercentile", v)}
            presets={[
              { label: "Reactive (55)",  value: 55 },
              { label: "Default (70)",   value: 70 },
              { label: "Conservative (80)", value: 80 },
            ]}
          />

          <ParamSlider
            label="Round-Trip Transaction Cost"
            value={params.txCostBps}
            min={4} max={20} step={2}
            color="#7F77DD"
            format={(v) => `${v} bps`}
            onChange={(v) => updateParam("txCostBps", v)}
            presets={[
              { label: "Low (4bp)",  value: 4 },
              { label: "Real (8bp)", value: 8 },
              { label: "High (16bp)", value: 16 },
            ]}
          />

          {/* Regime Routing Toggle */}
          <div className="flex items-center justify-between py-1">
            <div>
              <p className="text-xs font-medium text-gray-300">HMM Regime Routing</p>
              <p className="text-xs text-gray-600">Bull→max-Sharpe, Bear→min-var</p>
            </div>
            <button
              onClick={() => updateParam("regimeRouting", !params.regimeRouting)}
              className={cn(
                "relative w-10 h-5 rounded-full transition-all duration-200 flex-shrink-0",
                params.regimeRouting ? "bg-teal-400" : "bg-navy-600"
              )}
            >
              <span
                className={cn(
                  "absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200",
                  params.regimeRouting ? "left-5" : "left-0.5"
                )}
              />
            </button>
          </div>

          {/* Reset */}
          <button
            onClick={() => { setParams(DEFAULT_PARAMS); setActiveCrisis(null); }}
            className="w-full py-2 text-xs text-gray-400 hover:text-white border border-navy-700 hover:border-navy-600 rounded-lg transition-colors"
          >
            Reset to defaults
          </button>
        </div>

        {/* Metrics + Chart */}
        <div className="space-y-4">

          {/* KPI Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              {
                label: "CAGR",
                value: metrics.cagr,
                suffix: "%",
                decimals: 2,
                good: metrics.cagr > niftyMetrics.cagr,
                vs: `${cagrDelta >= 0 ? "+" : ""}${cagrDelta.toFixed(1)}% vs NIFTY`,
              },
              {
                label: "Sharpe Ratio",
                value: metrics.sharpe,
                decimals: 3,
                good: metrics.sharpe > niftyMetrics.sharpe,
                vs: `${sharpeDelta >= 0 ? "+" : ""}${sharpeDelta.toFixed(3)} vs NIFTY`,
              },
              {
                label: "Max Drawdown",
                value: metrics.maxDrawdown,
                suffix: "%",
                decimals: 1,
                good: metrics.maxDrawdown > niftyMetrics.maxDrawdown,
                vs: `${ddDelta >= 0 ? "+" : ""}${ddDelta.toFixed(1)}% vs NIFTY`,
              },
              {
                label: "Rebalances",
                value: result.rebalanceCount,
                decimals: 0,
                vs: `${result.minVarDays} min-var days`,
              },
            ].map((m) => (
              <div
                key={m.label}
                className={`rounded-xl p-3 text-center transition-all duration-300 ${
                  m.good === true  ? "metric-glow-good" :
                  m.good === false ? "metric-glow-bad"  : ""
                }`}
                style={{
                  background: m.good === true
                    ? "linear-gradient(135deg, rgba(29,158,117,0.1) 0%, rgba(15,22,41,0.8) 100%)"
                    : m.good === false
                    ? "linear-gradient(135deg, rgba(216,90,48,0.08) 0%, rgba(15,22,41,0.8) 100%)"
                    : "rgba(15,22,41,0.6)",
                  border: `1px solid ${
                    m.good === true  ? "rgba(29,158,117,0.35)" :
                    m.good === false ? "rgba(216,90,48,0.3)" :
                                       "rgba(26,37,80,0.8)"
                  }`,
                }}
              >
                <MetricTile
                  label={m.label}
                  value={m.value}
                  suffix={m.suffix}
                  decimals={m.decimals}
                  good={m.good}
                  size="md"
                />
                {m.vs && <p className="text-xs text-gray-600 mt-1">{m.vs}</p>}
              </div>
            ))}
          </div>

          {/* Equity Curve */}
          <div
            className="rounded-xl p-4"
            style={{
              background: "linear-gradient(155deg, rgba(15,22,41,0.9) 0%, rgba(10,15,30,0.95) 100%)",
              border: "1px solid rgba(29,158,117,0.15)",
              boxShadow: "0 0 40px rgba(29,158,117,0.04)",
            }}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs font-semibold text-gray-300">
                  {activeCrisis ? `${activeCrisis} — Deep Dive` : "Equity Curve (base = 100)"}
                </p>
                <p className="text-xs text-gray-600 mt-0.5">
                  {activeCrisis
                    ? "30 days context + full crisis window, re-indexed to 100 at window start"
                    : realData
                    ? `Real backtest · AdaptiveBeta vs NIFTY50 · ${realData.meta.startDate.substring(0, 4)}–${realData.meta.endDate.substring(0, 4)} · ${realData.meta.tradingDays} trading days`
                    : "AdaptiveBeta vs NIFTY50 Buy & Hold · 2015–2024 · Synthetic"}
                </p>
              </div>
              {activeCrisis && (
                <button
                  onClick={() => setActiveCrisis(null)}
                  className="text-xs text-gray-500 hover:text-white border border-navy-700 hover:border-navy-600 px-2 py-1 rounded transition-colors"
                >
                  ✕ Full view
                </button>
              )}
            </div>

            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
                <defs>
                  <linearGradient id="portfolioGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#1D9E75" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#1D9E75" stopOpacity={0}   />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#4B5563", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  interval={Math.floor(chartData.length / 6)}
                />
                <YAxis
                  tick={{ fill: "#4B5563", fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                  width={42}
                  tickFormatter={(v) => v.toFixed(0)}
                />
                <Tooltip content={<DemoTooltip />} />

                {/* Crisis shading (full view only) */}
                {!activeCrisis && (
                  <>
                    <ReferenceArea x1="2020-02-20" x2="2020-05-01" fill="rgba(216,90,48,0.07)"
                      label={{ value: "COVID", fill: "#D85A30", fontSize: 9, position: "insideTop" }} />
                    <ReferenceArea x1="2023-01-25" x2="2023-03-15" fill="rgba(239,159,39,0.07)"
                      label={{ value: "ADANI", fill: "#EF9F27", fontSize: 9, position: "insideTop" }} />
                    <ReferenceArea x1="2018-08-01" x2="2018-11-30" fill="rgba(127,119,221,0.06)"
                      label={{ value: "IL&FS", fill: "#7F77DD", fontSize: 9, position: "insideTop" }} />
                  </>
                )}

                {/* Crisis start line */}
                {activeCrisis && crisisDef && (
                  <ReferenceLine
                    x={crisisDef.start}
                    stroke={crisisDef.color}
                    strokeDasharray="5 3"
                    strokeWidth={1.5}
                    label={{ value: "Crisis start", fill: crisisDef.color, fontSize: 9 }}
                  />
                )}

                <ReferenceLine y={100} stroke="rgba(255,255,255,0.10)" strokeDasharray="4 4" />

                <Area
                  type="monotone"
                  dataKey="portfolio"
                  fill="url(#portfolioGrad)"
                  stroke="#1D9E75"
                  strokeWidth={2.5}
                  dot={false}
                  name="AdaptiveBeta"
                  activeDot={{ r: 4, fill: "#1D9E75" }}
                />
                <Line
                  type="monotone"
                  dataKey="nifty"
                  stroke="#EF9F27"
                  strokeWidth={1.5}
                  dot={false}
                  strokeDasharray="4 3"
                  name="NIFTY50"
                  activeDot={{ r: 3 }}
                />
              </ComposedChart>
            </ResponsiveContainer>

            {/* Legend */}
            <div className="flex items-center gap-4 mt-2 justify-end">
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <div className="w-4 h-0.5 bg-teal-400 rounded" />
                AdaptiveBeta
              </div>
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <div className="w-4 h-0.5 bg-amber-400 rounded" style={{ borderTop: "1px dashed #EF9F27" }} />
                NIFTY50
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Bottom: Signal Timeline + Crisis Selector ── */}
      <div className="grid md:grid-cols-2 gap-5">

        {/* Signal Timeline */}
        <div
          className="rounded-xl p-4"
          style={{
            background: "rgba(15,22,41,0.8)",
            border: "1px solid rgba(26,37,80,0.9)",
          }}
        >
          <p className="text-xs font-semibold text-gray-300 mb-3">
            Signal Activity — 10-year view
          </p>
          <SignalTimeline fullResults={result.fullResults} />
          <div className="flex gap-4 mt-3">
            {[
              { color: "bg-teal-400/20 border-teal-400/30 text-teal-300", label: "HOLD", value: result.holdDays },
              { color: "bg-amber-400/20 border-amber-400/30 text-amber-300", label: "REBALANCE", value: result.rebalanceCount },
              { color: "bg-coral-400/20 border-coral-400/30 text-coral-300", label: "MIN-VAR", value: result.minVarDays },
            ].map(({ color, label, value }) => (
              <div key={label} className={cn("px-2 py-1 rounded border text-center flex-1", color)}>
                <div className="text-sm font-bold font-mono">{value}</div>
                <div className="text-xs opacity-80">{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Crisis Selector */}
        <div
          className="rounded-xl p-4"
          style={{
            background: "rgba(15,22,41,0.8)",
            border: "1px solid rgba(26,37,80,0.9)",
          }}
        >
          <p className="text-xs font-semibold text-gray-300 mb-3">
            Crisis Event Deep-Dive — click to zoom
          </p>
          <CrisisPanel
            result={result}
            activeCrisis={activeCrisis}
            onSelect={(name) => setActiveCrisis((prev) => (prev === name ? null : name))}
          />
        </div>
      </div>

      {/* ── Insight Banner ── */}
      <div
        className="flex items-start gap-3 rounded-xl p-4"
        style={{
          background: "linear-gradient(135deg, rgba(29,158,117,0.07) 0%, rgba(15,22,41,0.9) 100%)",
          border: "1px solid rgba(29,158,117,0.25)",
          boxShadow: "0 0 30px rgba(29,158,117,0.05)",
        }}
      >
        <span className="text-lg flex-shrink-0 mt-0.5">💡</span>
        <div>
          <p className="text-xs font-semibold text-teal-300 mb-0.5">Strategy Insight</p>
          <p className="text-xs text-gray-400 leading-relaxed">{insight}</p>
        </div>
      </div>

      {/* ── VIX vs BetaVol Chart ── */}
      <VixBetavolChart data={result.curve} betavolThreshold={result.betavolThreshold} vixThreshold={params.vixThreshold} />
    </div>
  );
}

// ─── VIX + BetaVol Signal Chart ───────────────────────────────────────────────

function VixBetavolChart({
  data,
  betavolThreshold,
  vixThreshold,
}: {
  data: SimPoint[];
  betavolThreshold: number;
  vixThreshold: number;
}) {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: "linear-gradient(155deg, rgba(15,22,41,0.9) 0%, rgba(10,15,30,0.95) 100%)",
        border: "1px solid rgba(127,119,221,0.15)",
      }}
    >
      <div className="mb-3">
        <p className="text-xs font-semibold text-gray-300">Signal Inputs — VIX & Predicted BetaVol</p>
        <p className="text-xs text-gray-600 mt-0.5">
          When VIX crosses the orange line → MIN_VARIANCE. When BetaVol crosses the amber line → REBALANCE.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#4B5563", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            interval={Math.floor(data.length / 6)}
          />
          <YAxis
            yAxisId="vix"
            orientation="right"
            tick={{ fill: "#4B5563", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={36}
            domain={[0, 90]}
          />
          <YAxis
            yAxisId="bv"
            orientation="left"
            tick={{ fill: "#4B5563", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={36}
            tickFormatter={(v) => v.toFixed(2)}
            domain={[0, 0.45]}
          />
          <Tooltip
            contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8, fontSize: 11 }}
            formatter={(v: number, name: string) =>
              name === "vix" ? [v.toFixed(1), "VIX"] : [v.toFixed(3), "BetaVol"]
            }
          />
          {/* Threshold lines */}
          <ReferenceLine
            yAxisId="vix"
            y={vixThreshold}
            stroke="#D85A30"
            strokeWidth={1.5}
            strokeDasharray="5 3"
            label={{ value: `VIX=${vixThreshold}`, fill: "#D85A30", fontSize: 9, position: "right" }}
          />
          <ReferenceLine
            yAxisId="bv"
            y={betavolThreshold}
            stroke="#EF9F27"
            strokeWidth={1.5}
            strokeDasharray="5 3"
            label={{ value: `BV=${betavolThreshold}`, fill: "#EF9F27", fontSize: 9, position: "left" }}
          />
          <Line
            yAxisId="bv"
            type="monotone"
            dataKey="betavol"
            stroke="#7F77DD"
            strokeWidth={1.2}
            dot={false}
            name="betavol"
          />
          <Line
            yAxisId="vix"
            type="monotone"
            dataKey="vix"
            stroke="#D85A3055"
            strokeWidth={1}
            dot={false}
            name="vix"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
