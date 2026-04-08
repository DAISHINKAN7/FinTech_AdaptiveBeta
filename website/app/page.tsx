"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";
import HeroChart from "@/components/HeroChart";
import MetricsCard from "@/components/MetricsCard";
import { strategyMetrics } from "@/lib/sampleData";
import {
  ArrowRight, Github, BookOpen, ChevronDown,
  Sliders, Zap, TrendingUp, Shield, Brain, Activity,
} from "lucide-react";

// ── Animated counter ──────────────────────────────────────────────────────────
function useCounter(target: number, duration = 1400, active = false): number {
  const [v, setV] = useState(0);
  const raf    = useRef<number | null>(null);
  const t0     = useRef<number | null>(null);
  useEffect(() => {
    if (!active) return;
    t0.current = null;
    const step = (ts: number) => {
      if (t0.current === null) t0.current = ts;
      const p    = Math.min((ts - t0.current) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setV(ease * target);
      if (p < 1) raf.current = requestAnimationFrame(step);
    };
    if (raf.current) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [target, duration, active]);
  return v;
}

function AnimatedStat({
  value, suffix, label, decimals = 0, delay = 0, color = "text-teal-400",
}: {
  value: number; suffix: string; label: string;
  decimals?: number; delay?: number; color?: string;
}) {
  const ref    = useRef(null);
  const inView = useInView(ref, { once: true });
  const count  = useCounter(value, 1600, inView);
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay, duration: 0.6 }}
      className="text-center group cursor-default select-none"
    >
      <div className={`text-4xl md:text-5xl lg:text-6xl font-black font-mono mb-2 transition-colors duration-300 group-hover:text-white`}>
        <span className="text-white">{count.toFixed(decimals)}</span>
        <span className={color + " text-2xl md:text-3xl"}>{suffix}</span>
      </div>
      <div className="text-sm text-gray-500 font-medium tracking-wide">{label}</div>
    </motion.div>
  );
}

// ── Floating badge ─────────────────────────────────────────────────────────────
function FloatingBadge({
  value, label, sub, color, yRange, delay = 0,
  side,
}: {
  value: string; label: string; sub: string; color: string;
  yRange?: [number, number]; delay?: number; side: "left" | "right";
}) {
  return (
    <motion.div
      className={`absolute ${side === "left" ? "left-4 xl:left-10" : "right-4 xl:right-10"} hidden xl:block z-10`}
      style={{ top: side === "left" ? "32%" : "24%" }}
      animate={{ y: yRange ?? [0, -12, 0] }}
      transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay }}
    >
      <div
        className="card-glass px-4 py-3 rounded-xl min-w-[130px]"
        style={{ boxShadow: `0 0 28px ${color}28, 0 0 60px ${color}0A` }}
      >
        <div className="text-2xl font-black font-mono mb-0.5" style={{ color }}>
          {value}
        </div>
        <div className="text-xs text-gray-300 font-medium">{label}</div>
        <div className="text-xs text-gray-600 mt-0.5">{sub}</div>
      </div>
    </motion.div>
  );
}

// ── Signal ticker ─────────────────────────────────────────────────────────────
const TICKERS = [
  { label: "VIX",    value: "19.2",  color: "text-amber-400" },
  { label: "Signal", value: "HOLD",  color: "text-teal-400"  },
  { label: "Regime", value: "BULL",  color: "text-purple-400" },
  { label: "β̂(20d)", value: "0.872", color: "text-white"     },
  { label: "BetaVol",value: "0.114", color: "text-teal-300"  },
  { label: "Fold",   value: "6/6",   color: "text-gray-300"  },
];

function SignalTicker() {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setIdx((i) => (i + 1) % TICKERS.length), 2200);
    return () => clearInterval(t);
  }, []);
  const tick = TICKERS[idx];
  return (
    <div className="hidden md:flex items-center gap-2 text-xs font-mono bg-navy-800/70 border border-navy-700/60 rounded-full px-3 py-1.5 backdrop-blur-sm min-w-[220px]">
      <span className="text-teal-400 animate-pulse text-[8px]">●</span>
      <span className="text-gray-500">{tick.label}:</span>
      <motion.span
        key={idx}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className={`font-semibold ${tick.color}`}
      >
        {tick.value}
      </motion.span>
      <span className="text-gray-700 ml-auto text-[9px] tracking-wider">LIVE SIM</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────

const fadeUp = {
  hidden:  { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1, y: 0,
    transition: { delay: i * 0.12, duration: 0.55, ease: "easeOut" },
  }),
};

export default function HomePage() {
  return (
    <div className="min-h-screen">

      {/* ══════════════════════════════════════════════════════════════════
          HERO
      ══════════════════════════════════════════════════════════════════ */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-24 pb-20 overflow-hidden">

        {/* Grid background */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.35]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(29,158,117,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.07) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        {/* Radial fade over grid */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 80% 70% at 50% 50%, transparent 20%, #0A0F1E 75%)",
          }}
        />

        {/* Glowing orbs */}
        <motion.div
          className="absolute left-[8%] top-[22%] w-[480px] h-[480px] rounded-full pointer-events-none"
          animate={{ scale: [1, 1.12, 1], opacity: [0.08, 0.18, 0.08] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
          style={{ background: "radial-gradient(circle, rgba(29,158,117,0.9), transparent)", filter: "blur(80px)" }}
        />
        <motion.div
          className="absolute right-[6%] bottom-[28%] w-[420px] h-[420px] rounded-full pointer-events-none"
          animate={{ scale: [1, 1.15, 1], opacity: [0.06, 0.14, 0.06] }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut", delay: 2.5 }}
          style={{ background: "radial-gradient(circle, rgba(127,119,221,0.9), transparent)", filter: "blur(80px)" }}
        />
        <motion.div
          className="absolute right-[35%] top-[18%] w-[280px] h-[280px] rounded-full pointer-events-none"
          animate={{ scale: [1, 1.2, 1], opacity: [0.04, 0.09, 0.04] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          style={{ background: "radial-gradient(circle, rgba(239,159,39,0.9), transparent)", filter: "blur(70px)" }}
        />

        {/* Floating metric badges (xl+ only) */}
        <FloatingBadge
          value="0.893" label="Best Sharpe"    sub="MomentumQuality"
          color="#1D9E75" delay={0}   side="left"
        />
        <motion.div
          className="absolute right-4 xl:right-10 bottom-[34%] hidden xl:block z-10"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 5.5, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        >
          <div className="card-glass px-4 py-3 rounded-xl min-w-[130px] glow-purple">
            <div className="text-2xl font-black font-mono text-purple-300 mb-0.5">17.47%</div>
            <div className="text-xs text-gray-300 font-medium">Best CAGR</div>
            <div className="text-xs text-gray-600 mt-0.5">10Y OOS Backtest</div>
          </div>
        </motion.div>
        <motion.div
          className="absolute right-4 xl:right-10 top-[26%] hidden xl:block z-10"
          animate={{ y: [0, -8, 0] }}
          transition={{ duration: 6.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        >
          <div className="card-glass px-4 py-3 rounded-xl min-w-[130px] glow-amber">
            <div className="text-2xl font-black font-mono text-amber-300 mb-0.5">−38.2%</div>
            <div className="text-xs text-gray-300 font-medium">Max Drawdown</div>
            <div className="text-xs text-gray-600 mt-0.5">AdaptiveBeta</div>
          </div>
        </motion.div>

        {/* Badge row */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="mb-7 flex flex-wrap items-center gap-3 justify-center"
        >
          <span className="badge-teal">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse-slow" />
            M.Tech Thesis · SIT Pune · 2026
          </span>
          <SignalTicker />
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.12 }}
          className="text-5xl md:text-7xl lg:text-[5.2rem] font-black text-center max-w-5xl leading-[1.07] mb-6 tracking-tight"
        >
          <span className="text-white">Beta is not constant.</span>
          <br />
          <span className="shimmer-text">Neither should your portfolio be.</span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.28 }}
          className="text-lg md:text-xl text-gray-400 text-center max-w-2xl mb-12 leading-relaxed"
        >
          An LSTM deep learning system that predicts future beta volatility and triggers
          adaptive portfolio rebalancing — only when systematic risk is about to shift.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.42 }}
          className="flex flex-wrap gap-4 justify-center mb-16"
        >
          <Link
            href="/demo"
            className="group inline-flex items-center gap-2 px-7 py-3.5 font-bold rounded-xl transition-all duration-200"
            style={{
              background: "linear-gradient(135deg, #1D9E75, #5DCAA5)",
              color: "#060B16",
              boxShadow: "0 0 32px rgba(29,158,117,0.4), 0 4px 16px rgba(29,158,117,0.3)",
            }}
          >
            <Zap className="w-4 h-4 transition-transform duration-200 group-hover:scale-110" />
            Try Live Demo
          </Link>
          <Link
            href="/research"
            className="inline-flex items-center gap-2 px-6 py-3.5 bg-navy-700/80 hover:bg-navy-700 text-white font-medium rounded-xl border border-navy-600 hover:border-teal-400/40 transition-all duration-200 backdrop-blur-sm"
          >
            Explore the Research <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="https://github.com/daishinkan7/ai_fintech"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3.5 text-gray-400 hover:text-white font-medium rounded-xl border border-navy-700 hover:border-navy-600 transition-all duration-200"
          >
            <Github className="w-4 h-4" /> GitHub
          </a>
        </motion.div>

        {/* Hero chart */}
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.75, delay: 0.55 }}
          className="w-full max-w-4xl"
        >
          <div
            className="rounded-2xl overflow-hidden border"
            style={{
              background: "rgba(15,22,41,0.85)",
              borderColor: "rgba(29,158,117,0.2)",
              boxShadow: "0 0 60px rgba(29,158,117,0.08), 0 20px 60px rgba(0,0,0,0.5)",
            }}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-navy-700/60">
              <div>
                <p className="section-label mb-0.5">Beta Dynamics</p>
                <p className="text-sm text-gray-500">
                  Rolling 30d / 60d / 120d beta for RELIANCE.NS with VIX overlay
                </p>
              </div>
              <span className="badge-teal">RELIANCE.NS</span>
            </div>
            <div className="p-4">
              <HeroChart />
            </div>
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.4 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 text-gray-600"
        >
          <span className="text-xs tracking-wider">Scroll to explore</span>
          <ChevronDown className="w-4 h-4 animate-bounce" />
        </motion.div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          KEY STATS — animated counting numbers
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-20 px-4 border-y border-navy-700/50 relative overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 60% 100% at 50% 50%, rgba(29,158,117,0.04) 0%, transparent 70%)",
          }}
        />
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4 relative">
            {/* Dividers (desktop) */}
            {[1,2,3].map((i) => (
              <div key={i} className="hidden md:block absolute top-1/2 -translate-y-1/2 w-px h-12 bg-navy-700/80"
                style={{ left: `${i * 25}%` }} />
            ))}
            <AnimatedStat value={49}   suffix=" stocks" label="NIFTY50 Universe"     color="text-teal-400"   delay={0}   />
            <AnimatedStat value={10}   suffix=" years"  label="2015–2025 Data"       color="text-purple-400" delay={0.1} />
            <AnimatedStat value={6}    suffix=" strats" label="Compared OOS"         color="text-amber-400"  delay={0.2} />
            <AnimatedStat value={17.5} suffix="%"       label="Best CAGR (MomQual)"  color="text-teal-400"   delay={0.3} decimals={1} />
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          THE INNOVATION
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-28 px-4">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.p custom={0} variants={fadeUp} className="section-label mb-3">The Innovation</motion.p>
            <motion.h2 custom={1} variants={fadeUp} className="text-3xl md:text-5xl font-black text-white mb-5 tracking-tight">
              The Threshold Trigger
            </motion.h2>
            <motion.p custom={2} variants={fadeUp} className="text-gray-400 max-w-2xl mx-auto text-lg">
              Instead of rebalancing every month (wasteful) or every day (too expensive),
              AdaptiveBeta only acts when the LSTM predicts beta instability is about to spike.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Predict",
                color: "#1D9E75",
                icon: <Brain className="w-6 h-6" />,
                desc: "LSTM predicts 20-day forward beta volatility using India VIX, macro flows, RBI rates, FII/DII data, and rolling multi-window betas across 49 stocks.",
              },
              {
                step: "02",
                title: "Trigger",
                color: "#EF9F27",
                icon: <Zap className="w-6 h-6" />,
                desc: "Rebalance ONLY when predicted betavol exceeds calibrated threshold (70th percentile) or VIX > 22 for crisis override. ~65–75% of days are HOLD.",
              },
              {
                step: "03",
                title: "Optimise",
                color: "#7F77DD",
                icon: <TrendingUp className="w-6 h-6" />,
                desc: "Route to max-Sharpe MVO (bull), min-variance (bear), or risk parity (transition) based on HMM regime classification. Sector caps prevent concentration.",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15, duration: 0.55 }}
                className="rounded-2xl border p-6 transition-all duration-300 hover:-translate-y-1"
                style={{
                  background: "rgba(15,22,41,0.7)",
                  borderColor: item.color + "28",
                  boxShadow: `0 0 30px ${item.color}0A`,
                }}
              >
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center mb-5"
                  style={{ background: item.color + "18", color: item.color }}
                >
                  {item.icon}
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-mono text-gray-600">{item.step}</span>
                  <h3 className="text-lg font-bold" style={{ color: item.color }}>
                    {item.title}
                  </h3>
                </div>
                <p className="text-gray-400 text-sm leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          LIVE DEMO CTA
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-24 px-4 relative overflow-hidden">
        {/* Gradient background */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 80% 60% at 20% 50%, rgba(29,158,117,0.06) 0%, transparent 60%), radial-gradient(ellipse 60% 60% at 80% 50%, rgba(127,119,221,0.05) 0%, transparent 60%)",
          }}
        />
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-teal-400/30 to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-navy-700/80 to-transparent" />

        <div className="max-w-5xl mx-auto relative">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.65 }}
            >
              <span className="badge bg-teal-400/12 text-teal-300 border border-teal-400/22 mb-5 inline-flex">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
                Interactive Demo
              </span>
              <h2 className="text-3xl md:text-4xl font-black text-white mt-3 mb-5 leading-tight tracking-tight">
                See the strategy in action —{" "}
                <span className="shimmer-text">live, in your browser</span>
              </h2>
              <p className="text-gray-400 leading-relaxed mb-8 text-lg">
                Adjust the VIX threshold, beta volatility percentile, and transaction costs
                with sliders and watch the equity curve, signal timeline, and performance
                metrics update <strong className="text-white">instantly</strong>.
                Deep-dive into COVID, ADANI, IL&amp;FS, or the 2018 rate hike.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  href="/demo"
                  className="inline-flex items-center gap-2 px-6 py-3 font-bold rounded-xl transition-all duration-200"
                  style={{
                    background: "linear-gradient(135deg, #1D9E75, #5DCAA5)",
                    color: "#060B16",
                    boxShadow: "0 0 24px rgba(29,158,117,0.35)",
                  }}
                >
                  <Sliders className="w-4 h-4" /> Open Simulator
                </Link>
                <Link
                  href="/results"
                  className="inline-flex items-center gap-2 px-5 py-3 text-gray-300 hover:text-white border border-navy-700 hover:border-navy-600 rounded-xl transition-all"
                >
                  <BookOpen className="w-4 h-4" /> Real Results
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.65, delay: 0.1 }}
              className="grid grid-cols-2 gap-3"
            >
              {[
                { label: "VIX threshold slider",  icon: "🎚️", desc: "15–40 range, see crisis responses live" },
                { label: "BetaVol percentile",    icon: "📊", desc: "50–90th pct, tune rebalance frequency" },
                { label: "Regime routing toggle", icon: "🔀", desc: "Bull→max-Sharpe, Bear→min-var routing" },
                { label: "4 crisis deep-dives",   icon: "🔍", desc: "Zoom into COVID, ADANI, IL&FS, Rate Hike" },
                { label: "Live signal timeline",  icon: "⏱️", desc: "Every day: HOLD / REBALANCE / MIN-VAR" },
                { label: "Real-time metrics",     icon: "📈", desc: "CAGR, Sharpe, MaxDD animate as you drag" },
              ].map((f, i) => (
                <motion.div
                  key={f.label}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.15 + i * 0.07 }}
                  className="rounded-xl border border-navy-700/80 p-4 text-center transition-all duration-200 hover:border-teal-400/25 hover:bg-navy-800/60"
                  style={{ background: "rgba(15,22,41,0.6)" }}
                >
                  <div className="text-2xl mb-2">{f.icon}</div>
                  <div className="text-xs font-semibold text-white mb-1">{f.label}</div>
                  <div className="text-xs text-gray-600 leading-tight">{f.desc}</div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          RESULTS PREVIEW
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-28 px-4" style={{ background: "rgba(15,22,41,0.4)" }}>
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-14"
          >
            <motion.p custom={0} variants={fadeUp} className="section-label mb-3">Results</motion.p>
            <motion.h2 custom={1} variants={fadeUp} className="text-3xl md:text-5xl font-black text-white mb-5 tracking-tight">
              Walk-Forward Backtest (2015–2025)
            </motion.h2>
            <motion.p custom={2} variants={fadeUp} className="text-gray-400 max-w-xl text-lg">
              Out-of-sample performance — 6 strategies, no lookahead, realistic tx costs.
              Sharpe Ratio shown below.
            </motion.p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {Object.entries(strategyMetrics).map(([key, s], i) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.96, y: 16 }}
                whileInView={{ opacity: 1, scale: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08, duration: 0.45 }}
              >
                <MetricsCard
                  label={s.label}
                  value={s.sharpe}
                  subValue={`${s.annualReturn}% CAGR`}
                  delta={key !== "buyHoldNifty" ? s.sharpe - strategyMetrics.buyHoldNifty.sharpe : undefined}
                  deltaLabel="vs NIFTY50"
                  color={s.color}
                  highlight={key === "momentumQuality"}
                  metricName="Sharpe Ratio"
                />
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 }}
            className="text-center mt-10"
          >
            <Link
              href="/results"
              className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-300 font-semibold transition-colors"
            >
              View full results, equity curves &amp; stress analysis <ArrowRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          METHODOLOGY
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-28 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-14 items-center">
            <motion.div
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.65 }}
            >
              <p className="section-label mb-3">Methodology</p>
              <h2 className="text-3xl md:text-4xl font-black text-white mb-6 leading-tight tracking-tight">
                India-specific signals that mainstream models ignore
              </h2>
              <ul className="space-y-3.5 text-gray-400">
                {[
                  "India VIX (not US VIX) — local fear gauge",
                  "RBI repo rate & 91-day T-bill spread",
                  "FII / DII monthly flow data",
                  "USD/INR and Brent crude oil exposure",
                  "NIFTY Bank / IT / FMCG sector rotation signals",
                  "Rolling multi-window beta (30d, 60d, 120d)",
                ].map((item) => (
                  <li key={item} className="flex items-start gap-3">
                    <span className="text-teal-400 mt-0.5 flex-shrink-0 font-bold">✓</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Link
                  href="/research"
                  className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-300 font-semibold transition-colors"
                >
                  Deep-dive into the methodology <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.65, delay: 0.1 }}
              className="grid grid-cols-2 gap-4"
            >
              {[
                { label: "Data Sources",       value: "10",       icon: "🗄️", color: "text-teal-300" },
                { label: "Features / Stock",   value: "~35",      icon: "⚙️", color: "text-purple-300" },
                { label: "Training Period",    value: "3Y folds", icon: "📅", color: "text-amber-300" },
                { label: "OOS Test Years",     value: "2018–25",  icon: "🔍", color: "text-teal-300" },
                { label: "Walk-Forward Folds", value: "6",        icon: "🔄", color: "text-purple-300" },
                { label: "LSTM Parameters",    value: "~45K",     icon: "🧠", color: "text-amber-300" },
              ].map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 }}
                  className="rounded-xl border border-navy-700/80 p-4 text-center transition-all duration-200 hover:border-teal-400/20 hover:-translate-y-0.5"
                  style={{ background: "rgba(15,22,41,0.7)" }}
                >
                  <div className="text-2xl mb-2">{item.icon}</div>
                  <div className={`text-xl font-black ${item.color} font-mono`}>{item.value}</div>
                  <div className="text-xs text-gray-500 mt-1">{item.label}</div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════
          CTA
      ══════════════════════════════════════════════════════════════════ */}
      <section className="py-28 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 28 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.65 }}
            className="rounded-2xl border p-10 relative overflow-hidden"
            style={{
              background: "linear-gradient(135deg, rgba(15,22,41,0.9) 0%, rgba(20,29,58,0.8) 100%)",
              borderColor: "rgba(29,158,117,0.22)",
              boxShadow: "0 0 60px rgba(29,158,117,0.07)",
            }}
          >
            {/* Decorative glow */}
            <div
              className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-1 rounded-full"
              style={{ background: "linear-gradient(90deg, transparent, rgba(29,158,117,0.5), transparent)" }}
            />
            <p className="section-label mb-5">Open Source</p>
            <h2 className="text-2xl md:text-3xl font-black text-white mb-4 tracking-tight">
              Full pipeline on GitHub
            </h2>
            <p className="text-gray-400 mb-9 leading-relaxed">
              4 Python scripts, complete src package, Streamlit dashboard, and this website.
              MIT licensed — fork, extend, build on it.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <a
                href="https://github.com/daishinkan7/ai_fintech"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 font-bold rounded-xl transition-all duration-200"
                style={{
                  background: "linear-gradient(135deg, #1D9E75, #5DCAA5)",
                  color: "#060B16",
                  boxShadow: "0 0 24px rgba(29,158,117,0.3)",
                }}
              >
                <Github className="w-4 h-4" /> View on GitHub
              </a>
              <Link
                href="/architecture"
                className="inline-flex items-center gap-2 px-6 py-3 bg-navy-700 hover:bg-navy-600 text-white font-medium rounded-xl border border-navy-600 transition-all"
              >
                System Architecture
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
