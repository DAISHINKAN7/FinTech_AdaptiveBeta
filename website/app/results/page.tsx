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
} from "recharts";

function DrawdownChart() {
  const dd = computeDrawdown(equityCurveData.adaptiveBeta);
  const displayData = dd.filter((_, i) => i % 5 === 0);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={displayData}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="date" tick={{ fill: "#6B7280", fontSize: 11 }} tickLine={false} interval={Math.floor(displayData.length / 6)} />
        <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} tickLine={false} tickFormatter={(v) => `${v.toFixed(0)}%`} />
        <Tooltip
          contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8 }}
          labelStyle={{ color: "#9CA3AF" }}
          formatter={(v: number) => [`${v.toFixed(1)}%`, "Drawdown"]}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
        <Area type="monotone" dataKey="drawdown" stroke="#D85A30" fill="rgba(216,90,48,0.15)" strokeWidth={1.5} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default function ResultsPage() {
  return (
    <div className="min-h-screen py-24 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <p className="section-label mb-3">Backtest Results</p>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Walk-Forward Out-of-Sample Performance
          </h1>
          <p className="text-gray-400 max-w-2xl text-lg">
            Strict chronological validation — train on 3 years, test on 1 year, roll forward.
            No lookahead. Transaction costs included. NIFTY50 universe, 2018–2024.
          </p>
          <div className="flex flex-wrap gap-3 mt-6">
            <span className="badge-teal">6 OOS years</span>
            <span className="badge-purple">5 strategies compared</span>
            <span className="badge-amber">0.08% round-trip cost</span>
            <span className="badge bg-navy-700 text-gray-300 border border-navy-600">49 stocks</span>
          </div>
        </motion.div>

        {/* KPI Row */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-12"
        >
          {[
            { label: "AdaptiveBeta Sharpe", value: "1.42", vs: "+0.53 vs NIFTY", good: true },
            { label: "Annual Return", value: "18.4%", vs: "+5.7% vs NIFTY", good: true },
            { label: "Max Drawdown", value: "−18.2%", vs: "vs −38.7% NIFTY", good: true },
            { label: "Calmar Ratio", value: "1.01", vs: "vs 0.33 NIFTY", good: true },
          ].map((item) => (
            <div key={item.label} className="card text-center">
              <div className="text-2xl md:text-3xl font-bold text-white mb-1">{item.value}</div>
              <div className="text-xs text-gray-400 mb-2">{item.label}</div>
              <div className="text-xs text-teal-400 font-medium">{item.vs}</div>
            </div>
          ))}
        </motion.div>

        {/* Equity Curves */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="card mb-8"
        >
          <div className="mb-6">
            <p className="section-label mb-1">Cumulative Returns</p>
            <h2 className="text-xl font-semibold text-white">
              Equity Curves — All Strategies (base=100)
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Toggle strategies below. Shaded regions: COVID crash (Feb–May 2020), ADANI crisis (Jan–Mar 2023).
            </p>
          </div>
          <EquityCurveChart />
        </motion.div>

        {/* Drawdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.25 }}
          className="card mb-8"
        >
          <div className="mb-4">
            <p className="section-label mb-1">Drawdown</p>
            <h2 className="text-lg font-semibold text-white">AdaptiveBeta Drawdown</h2>
          </div>
          <DrawdownChart />
        </motion.div>

        {/* Strategy Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="card mb-12"
        >
          <div className="mb-6">
            <p className="section-label mb-1">Strategy Comparison</p>
            <h2 className="text-xl font-semibold text-white">Full Performance Metrics</h2>
            <p className="text-sm text-gray-400 mt-1">
              Best value in each column highlighted in teal. Worst in coral.
            </p>
          </div>
          <StrategyTable />
        </motion.div>

        {/* Stress Events */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.35 }}
        >
          <div className="mb-6">
            <p className="section-label mb-1">Stress Event Analysis</p>
            <h2 className="text-2xl font-bold text-white">Performance Under Market Stress</h2>
            <p className="text-gray-400 mt-2">
              AdaptiveBeta&apos;s VIX-triggered min-variance mode provides systematic downside protection
              during market crashes.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {Object.entries(stressEvents).map(([event, data]) => (
              <div key={event} className="card">
                <h3 className="font-semibold text-white mb-4">{event}</h3>
                <div className="space-y-3">
                  {Object.entries(data).map(([stratKey, { return: ret, maxDD }]) => {
                    const s = strategyMetrics[stratKey as keyof typeof strategyMetrics];
                    return (
                      <div key={stratKey} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                            style={{ background: s.color }}
                          />
                          <span className="text-sm text-gray-300">{s.label}</span>
                        </div>
                        <div className="flex gap-4 text-sm">
                          <span className={ret >= 0 ? "text-teal-400" : "text-coral-300"}>
                            {ret >= 0 ? "+" : ""}{ret}%
                          </span>
                          <span className="text-gray-500 text-xs self-center">
                            DD: {maxDD}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Regime analysis */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="card mt-8"
        >
          <p className="section-label mb-2">Regime Analysis</p>
          <h2 className="text-xl font-semibold text-white mb-6">
            Performance by Market Regime (HMM Classification)
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                regime: "Bull Market",
                color: "#1D9E75",
                adaptiveBeta: "+22.1%",
                nifty: "+19.8%",
                note: "Max-Sharpe MVO with beta targeting",
              },
              {
                regime: "Bear Market",
                color: "#D85A30",
                adaptiveBeta: "−4.2%",
                nifty: "−18.3%",
                note: "Min-variance defensive mode",
              },
              {
                regime: "Transition / High-VIX",
                color: "#EF9F27",
                adaptiveBeta: "+8.4%",
                nifty: "+6.1%",
                note: "Risk parity with VIX override",
              },
            ].map((r) => (
              <div key={r.regime} className="bg-navy-900/60 rounded-lg p-5 border border-navy-700">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-3 h-3 rounded-full" style={{ background: r.color }} />
                  <h3 className="font-semibold text-white text-sm">{r.regime}</h3>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">AdaptiveBeta</span>
                    <span className="text-teal-300 font-medium">{r.adaptiveBeta}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">NIFTY50</span>
                    <span className="text-gray-300">{r.nifty}</span>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-3 italic">{r.note}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
