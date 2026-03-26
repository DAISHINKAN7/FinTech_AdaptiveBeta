"use client";

import { motion } from "framer-motion";
import PipelineFlow from "@/components/PipelineFlow";
import ThresholdExplainer from "@/components/ThresholdExplainer";
import { modelComparison, shaValue } from "@/lib/sampleData";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" },
  }),
};

function SectionHeader({ label, title, subtitle }: { label: string; title: string; subtitle?: string }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      className="mb-10"
    >
      <motion.p custom={0} variants={fadeUp} className="section-label mb-2">
        {label}
      </motion.p>
      <motion.h2 custom={1} variants={fadeUp} className="text-2xl md:text-3xl font-bold text-white mb-3">
        {title}
      </motion.h2>
      {subtitle && (
        <motion.p custom={2} variants={fadeUp} className="text-gray-400 max-w-2xl">
          {subtitle}
        </motion.p>
      )}
    </motion.div>
  );
}

export default function ResearchPage() {
  return (
    <div className="min-h-screen py-24 px-4">
      <div className="max-w-5xl mx-auto space-y-24">
        {/* ---- Header ---- */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <p className="section-label mb-3">Research Deep-Dive</p>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-5">
            The Science Behind AdaptiveBeta
          </h1>
          <p className="text-gray-400 max-w-2xl text-lg leading-relaxed">
            A systematic walkthrough of the problem, the approach, the data, and the models —
            from first principles to walk-forward results.
          </p>
        </motion.div>

        {/* ---- 1. The Problem ---- */}
        <section>
          <SectionHeader
            label="01 — The Problem"
            title="Why Fixed Beta is Wrong"
            subtitle="Classical CAPM assumes beta is constant. It isn't — and using stale beta estimates for portfolio construction is provably suboptimal."
          />
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4 text-gray-300 leading-relaxed">
              <p>
                Beta (β) measures a stock's sensitivity to market moves. In the CAPM framework,
                it&apos;s assumed to be a fixed constant estimated from historical returns.
              </p>
              <p>
                But empirically, beta is <strong className="text-white">time-varying</strong>.
                During the COVID crash of 2020, RELIANCE.NS beta jumped from ~1.0 to ~1.65 in
                six weeks. During the 2018 IL&FS crisis, financial sector betas rose dramatically
                before the equity market reacted.
              </p>
              <p>
                If your portfolio is constructed assuming stable betas and the regime shifts,
                your actual risk exposure diverges massively from your target — <em>especially</em>
                in the crises where it matters most.
              </p>
            </div>
            <div className="card">
              <h3 className="text-sm font-semibold text-gray-300 mb-4">Beta Volatility Events</h3>
              <div className="space-y-3">
                {[
                  { event: "COVID Crash (2020)", betaShift: "+65%", duration: "6 weeks" },
                  { event: "IL&FS Crisis (2018)", betaShift: "+42%", duration: "8 weeks" },
                  { event: "ADANI Crisis (2023)", betaShift: "+31%", duration: "7 weeks" },
                  { event: "2018 Rate Hike", betaShift: "+28%", duration: "10 weeks" },
                ].map((row) => (
                  <div key={row.event} className="flex items-center justify-between py-2 border-b border-navy-700 last:border-0">
                    <span className="text-sm text-gray-300">{row.event}</span>
                    <div className="text-right">
                      <div className="text-sm font-medium text-coral-300">{row.betaShift} beta shift</div>
                      <div className="text-xs text-gray-500">{row.duration}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ---- 2. Approach ---- */}
        <section>
          <SectionHeader
            label="02 — Our Approach"
            title="Predict → Trigger → Optimise"
            subtitle="A 4-phase pipeline from raw data to adaptive portfolio. Click any stage to learn more."
          />
          <PipelineFlow />
        </section>

        {/* ---- 3. Data ---- */}
        <section>
          <SectionHeader
            label="03 — The Data"
            title="10 Data Sources, Perfectly Aligned"
            subtitle="Every dataset is reindexed to the NIFTY50 trading calendar and forward-filled. Monthly FII/DII data is daily-expanded."
          />
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-navy-700">
                  <th className="text-left py-3 text-gray-400 font-medium">Data Source</th>
                  <th className="text-left py-3 text-gray-400 font-medium">Coverage</th>
                  <th className="text-left py-3 text-gray-400 font-medium">Rows</th>
                  <th className="text-left py-3 text-gray-400 font-medium">Frequency</th>
                  <th className="text-left py-3 text-gray-400 font-medium">Used For</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { source: "NIFTY50 prices", coverage: "2015–2026", rows: "2,744", freq: "Daily", use: "Master calendar + market returns" },
                  { source: "49 stock prices", coverage: "2015–2026", rows: "2,753×51", freq: "Daily", use: "Beta computation, portfolio returns" },
                  { source: "India VIX", coverage: "2015–2026", rows: "2,733", freq: "Daily", use: "VIX override trigger, features" },
                  { source: "USD/INR rate", coverage: "2015–2026", rows: "2,900", freq: "Daily", use: "Macro exposure feature" },
                  { source: "Brent Crude", coverage: "2015–2026", rows: "2,801", freq: "Daily", use: "Commodity exposure feature" },
                  { source: "91d T-bill yield", coverage: "2015–2026", rows: "2,929", freq: "Daily (ff)", use: "Risk-free rate in Sharpe" },
                  { source: "RBI Repo Rate", coverage: "2015–2026", rows: "2,929", freq: "Daily (ff)", use: "Rate environment feature" },
                  { source: "FII Flows", coverage: "2015–2026", rows: "147", freq: "Monthly→Daily", use: "Institutional momentum" },
                  { source: "DII Flows", coverage: "2015–2026", rows: "147", freq: "Monthly→Daily", use: "Domestic buying pressure" },
                  { source: "NIFTY Bank/IT/FMCG", coverage: "2015–2026", rows: "2,749", freq: "Daily", use: "Sector rotation features" },
                ].map((row) => (
                  <tr key={row.source} className="border-b border-navy-800 hover:bg-navy-800/40">
                    <td className="py-3 text-white font-medium">{row.source}</td>
                    <td className="py-3 text-gray-400">{row.coverage}</td>
                    <td className="py-3 text-gray-400 font-mono">{row.rows}</td>
                    <td className="py-3">
                      <span className={`badge ${row.freq === "Daily" ? "badge-teal" : row.freq === "Monthly→Daily" ? "badge-amber" : "badge-purple"}`}>
                        {row.freq}
                      </span>
                    </td>
                    <td className="py-3 text-gray-400 text-xs">{row.use}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ---- 4. Feature Engineering ---- */}
        <section>
          <SectionHeader
            label="04 — Features"
            title="SHAP Feature Importance"
            subtitle="Which features actually drive beta volatility predictions? SHAP values from the XGBoost model reveal the signal hierarchy."
          />
          <div className="card">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={shaValue}
                layout="vertical"
                margin={{ left: 140, right: 30, top: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#6B7280", fontSize: 11 }} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fill: "#D1D5DB", fontSize: 12 }}
                  tickLine={false}
                  width={135}
                />
                <Tooltip
                  contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8 }}
                  formatter={(v: number) => [v.toFixed(3), "Mean |SHAP|"]}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {shaValue.map((entry, i) => (
                    <Cell
                      key={entry.feature}
                      fill={i === 0 ? "#1D9E75" : i < 3 ? "#5DCAA5" : "#3A6B59"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* ---- 5. Models ---- */}
        <section>
          <SectionHeader
            label="05 — Models"
            title="Model Comparison"
            subtitle="Four approaches to predicting beta volatility. LSTM wins on MAE and directional accuracy."
          />
          <div className="overflow-x-auto card">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-navy-700">
                  <th className="text-left py-3 text-gray-400">Model</th>
                  <th className="text-left py-3 text-gray-400">MAE ↓</th>
                  <th className="text-left py-3 text-gray-400">Direction Acc ↑</th>
                  <th className="text-left py-3 text-gray-400">Notes</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.map((row, i) => (
                  <tr
                    key={row.model}
                    className={`border-b border-navy-800 ${i === 3 ? "bg-teal-400/5" : ""}`}
                  >
                    <td className="py-3 font-medium text-white">
                      {row.model}
                      {i === 3 && (
                        <span className="ml-2 badge-teal text-xs">Proposed</span>
                      )}
                    </td>
                    <td className={`py-3 font-mono ${i === 3 ? "text-teal-300 font-semibold" : "text-gray-300"}`}>
                      {row.mae.toFixed(4)}
                    </td>
                    <td className={`py-3 font-mono ${i === 3 ? "text-teal-300 font-semibold" : "text-gray-300"}`}>
                      {row.directionAcc ? `${(row.directionAcc * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-3 text-gray-400 text-xs">{row.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* ---- 6. Threshold Trigger ---- */}
        <section>
          <SectionHeader
            label="06 — The Key Innovation"
            title="Interactive Threshold Trigger"
            subtitle="Drag the threshold slider to see how many rebalances get triggered. The 75th percentile balances responsiveness against transaction cost drag."
          />
          <ThresholdExplainer />
        </section>
      </div>
    </div>
  );
}
