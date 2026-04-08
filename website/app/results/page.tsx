"use client";

import { motion } from "framer-motion";
import EquityCurveChart from "@/components/EquityCurveChart";
import StrategyTable from "@/components/StrategyTable";
import { strategyMetrics, stressEvents, equityCurveData } from "@/lib/sampleData";
import { computeDrawdown } from "@/lib/utils";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import Link from "next/link";
import { ArrowRight, AlertTriangle } from "lucide-react";

// ─── Drawdown Chart ───────────────────────────────────────────────────────────

function DrawdownChart() {
  const dd = computeDrawdown(equityCurveData.adaptiveBeta);
  const displayData = dd.filter((_, i) => i % 4 === 0);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={displayData}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#6B7280", fontSize: 11 }}
          tickLine={false}
          interval={Math.floor(displayData.length / 6)}
        />
        <YAxis
          tick={{ fill: "#6B7280", fontSize: 11 }}
          tickLine={false}
          tickFormatter={(v) => `${v.toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8 }}
          labelStyle={{ color: "#9CA3AF" }}
          formatter={(v: number) => [`${v.toFixed(1)}%`, "Drawdown"]}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#D85A30"
          fill="rgba(216,90,48,0.12)"
          strokeWidth={1.5}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ─── Stress Bar Chart ─────────────────────────────────────────────────────────

function StressBarChart({
  data,
  title,
}: {
  data: Array<{ label: string; portfolio: number; nifty: number; color: string }>;
  title: string;
}) {
  const barData = data.map((d) => ({
    name: d.label,
    portfolio: d.portfolio,
    nifty: d.nifty,
    color: d.color,
  }));

  return (
    <div>
      <p className="text-xs text-gray-400 mb-2 font-medium">{title}</p>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={barData} layout="vertical" margin={{ left: 80, right: 20 }}>
          <XAxis type="number" tick={{ fill: "#6B7280", fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
          <YAxis type="category" dataKey="name" tick={{ fill: "#D1D5DB", fontSize: 10 }} width={75} />
          <Tooltip
            contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8, fontSize: 11 }}
            formatter={(v: number) => [`${v.toFixed(1)}%`, ""]}
          />
          <Bar dataKey="portfolio" name="Portfolio return" radius={[0, 3, 3, 0]}>
            {barData.map((entry, i) => (
              <Cell key={i} fill={entry.portfolio >= 0 ? "#1D9E75" : "#D85A30"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function ResultsPage() {
  const strategies = Object.entries(strategyMetrics);

  // Best-in-column for KPI highlight
  const bestSharpe = Math.max(...strategies.map(([, s]) => s.sharpe));
  const bestCagr   = Math.max(...strategies.map(([, s]) => s.annualReturn));
  const bestDD     = Math.max(...strategies.map(([, s]) => s.maxDrawdown)); // closest to 0

  return (
    <div className="min-h-screen py-24 px-4">
      <div className="max-w-6xl mx-auto">

        {/* ── Header ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-10"
        >
          <p className="section-label mb-3">Backtest Results</p>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Walk-Forward Out-of-Sample Performance
          </h1>
          <p className="text-gray-400 max-w-2xl text-lg">
            Strict chronological validation — train on 3 years, test on 1 year, roll
            forward. No lookahead. Transaction costs included.
          </p>
          <div className="flex flex-wrap gap-2 mt-5">
            <span className="badge-teal">2015–2025 data</span>
            <span className="badge-purple">6 strategies compared</span>
            <span className="badge-amber">0.08% round-trip cost</span>
            <span className="badge bg-navy-700 text-gray-300 border border-navy-600">49 NIFTY50 stocks</span>
          </div>
        </motion.div>

        {/* ── Honest Finding Banner ── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="mb-8 flex items-start gap-3 bg-amber-400/6 border border-amber-400/25 rounded-xl px-5 py-4"
        >
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-amber-300 mb-1">Honest Research Finding</p>
            <p className="text-xs text-gray-400 leading-relaxed">
              AdaptiveBeta (9.9% CAGR, Sharpe 0.643) <strong className="text-white">underperforms</strong> Equal Weight
              and Momentum-Quality in the walk-forward backtest. Root cause: the VIX override
              appears to have not triggered correctly during the COVID crash, causing worse
              drawdown than NIFTY50. The{" "}
              <strong className="text-white">concept is sound</strong> — the{" "}
              <Link href="/demo" className="text-teal-400 hover:underline">interactive demo</Link>{" "}
              shows proper mechanism behaviour. Key improvement: fix VIX data alignment and raise
              the threshold quantile from 55th to 70th percentile to reduce tx cost drag.
            </p>
          </div>
        </motion.div>

        {/* ── KPI Row ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10"
        >
          {[
            {
              label: "Best CAGR",
              value: `${bestCagr.toFixed(1)}%`,
              sub: "Momentum-Quality",
              badge: "bg-purple-400/15 text-purple-300 border-purple-400/25",
            },
            {
              label: "Best Sharpe",
              value: bestSharpe.toFixed(3),
              sub: "Momentum-Quality",
              badge: "bg-purple-400/15 text-purple-300 border-purple-400/25",
            },
            {
              label: "AdaptiveBeta CAGR",
              value: "9.9%",
              sub: "Needs improvement",
              badge: "bg-amber-400/15 text-amber-300 border-amber-400/25",
            },
            {
              label: "AdaptiveBeta Sharpe",
              value: "0.643",
              sub: "+0.643 — needs tuning",
              badge: "bg-amber-400/15 text-amber-300 border-amber-400/25",
            },
          ].map((item) => (
            <div key={item.label} className="card text-center">
              <div className="text-2xl md:text-3xl font-bold text-white mb-1 font-mono">
                {item.value}
              </div>
              <div className="text-xs text-gray-400 mb-2">{item.label}</div>
              <span className={`badge border text-xs ${item.badge}`}>{item.sub}</span>
            </div>
          ))}
        </motion.div>

        {/* ── Equity Curves ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="card mb-6"
        >
          <div className="mb-5">
            <p className="section-label mb-1">Cumulative Returns</p>
            <h2 className="text-xl font-semibold text-white">
              Equity Curves — All 6 Strategies (base = 100, from 2015)
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Toggle strategies. Shaded regions: COVID crash (Feb–May 2020), ADANI crisis (Jan–Mar 2023).
            </p>
          </div>
          <EquityCurveChart />
        </motion.div>

        {/* ── Drawdown ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card mb-6"
        >
          <div className="mb-4">
            <p className="section-label mb-1">Drawdown</p>
            <h2 className="text-lg font-semibold text-white">AdaptiveBeta Drawdown (2015–2025)</h2>
          </div>
          <DrawdownChart />
        </motion.div>

        {/* ── Strategy Table ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="card mb-10"
        >
          <div className="mb-5">
            <p className="section-label mb-1">Strategy Comparison</p>
            <h2 className="text-xl font-semibold text-white">Full Performance Metrics</h2>
            <p className="text-sm text-gray-400 mt-1">
              Click column headers to sort. Teal = best in column, coral = worst.
            </p>
          </div>
          <StrategyTable />
        </motion.div>

        {/* ── Stress Events ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="mb-6">
            <p className="section-label mb-1">Stress Event Analysis</p>
            <h2 className="text-2xl font-bold text-white">Performance Under Market Stress</h2>
            <p className="text-gray-400 mt-2 max-w-2xl">
              4 Indian market crises analysed. Returns are cumulative over the event window.
              AdaptiveBeta offers partial protection on 3 of 4 events; COVID is the outlier.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-5">
            {Object.entries(stressEvents).map(([event, data]) => (
              <div key={event} className="card">
                <h3 className="font-semibold text-white mb-4 text-sm">{event}</h3>
                <div className="space-y-2.5">
                  {Object.entries(data).map(([stratKey, { return: ret, maxDD }]) => {
                    const s = strategyMetrics[stratKey as keyof typeof strategyMetrics];
                    if (!s) return null;
                    return (
                      <div key={stratKey} className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <div
                            className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{ background: s.color }}
                          />
                          <span className="text-xs text-gray-300 truncate">{s.label}</span>
                        </div>
                        <div className="flex gap-3 text-xs flex-shrink-0">
                          <span className={ret >= 0 ? "text-teal-400 font-mono" : "text-coral-300 font-mono"}>
                            {ret >= 0 ? "+" : ""}{ret.toFixed(1)}%
                          </span>
                          <span className="text-gray-600">DD: {maxDD.toFixed(1)}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Regime Analysis ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="card mt-8"
        >
          <p className="section-label mb-2">Regime Analysis</p>
          <h2 className="text-xl font-semibold text-white mb-6">
            Strategy Behaviour by Market Regime (HMM)
          </h2>
          <div className="grid md:grid-cols-3 gap-5">
            {[
              {
                regime: "Bull Market",
                color: "#1D9E75",
                note: "Max-Sharpe MVO with beta targeting → β=0.90. AdaptiveBeta lags because MVO concentrates into fewer stocks vs equal-weight's broad exposure.",
                icon: "📈",
              },
              {
                regime: "Bear / High-VIX",
                color: "#D85A30",
                note: "VIX override triggers min-variance (β~0.3–0.4). Provides downside protection when working correctly. COVID data suggests a calibration bug in v1.",
                icon: "📉",
              },
              {
                regime: "Transition",
                color: "#EF9F27",
                note: "Risk parity routing. Portfolio beta ~0.75. Frequent rebalances during this phase due to elevated betavol — key source of tx cost drag.",
                icon: "↔️",
              },
            ].map((r) => (
              <div
                key={r.regime}
                className="bg-navy-900/60 rounded-xl p-4 border border-navy-700"
                style={{ borderColor: r.color + "30" }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">{r.icon}</span>
                  <h3 className="font-semibold text-white text-sm">{r.regime}</h3>
                </div>
                <p className="text-xs text-gray-400 leading-relaxed">{r.note}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Demo CTA ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mt-10 card border-teal-400/20 bg-gradient-to-br from-navy-800 to-navy-700 text-center"
        >
          <p className="section-label mb-3">Want to explore?</p>
          <h2 className="text-xl font-bold text-white mb-3">
            Try the Interactive Demo — Adjust Parameters Live
          </h2>
          <p className="text-gray-400 text-sm mb-6 max-w-xl mx-auto">
            See how changing the VIX threshold, betavol percentile, and transaction costs
            would have changed the strategy&apos;s performance on 10 years of market data.
          </p>
          <Link
            href="/demo"
            className="inline-flex items-center gap-2 px-6 py-3 bg-teal-400 hover:bg-teal-300 text-navy-900 font-semibold rounded-lg transition-all"
          >
            Open Strategy Simulator <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
