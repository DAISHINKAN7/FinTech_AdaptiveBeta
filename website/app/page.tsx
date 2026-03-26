"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, useInView } from "framer-motion";
import HeroChart from "@/components/HeroChart";
import MetricsCard from "@/components/MetricsCard";
import { strategyMetrics } from "@/lib/sampleData";
import { ArrowRight, Github, BookOpen, ChevronDown } from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.55, ease: "easeOut" },
  }),
};

function StatItem({
  value,
  label,
  suffix = "",
  delay = 0,
}: {
  value: string;
  label: string;
  suffix?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay, duration: 0.5 }}
      className="text-center"
    >
      <div className="text-3xl md:text-4xl font-bold text-white mb-1">
        {value}
        <span className="text-teal-400">{suffix}</span>
      </div>
      <div className="text-sm text-gray-400">{label}</div>
    </motion.div>
  );
}

export default function HomePage() {
  const [scrolled, setScrolled] = useState(false);
  const statsRef = useRef(null);
  const statsInView = useInView(statsRef, { once: true });

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 80);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen">
      {/* ------------------------------------------------------------------ */}
      {/* HERO SECTION */}
      {/* ------------------------------------------------------------------ */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-20 pb-16 overflow-hidden">
        {/* Animated background glow */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse at 50% 60%, rgba(29,158,117,0.12) 0%, transparent 65%)",
          }}
        />

        {/* Label */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-6"
        >
          <span className="badge-teal">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse-slow" />
            M.Tech Thesis · SIT Pune · 2026
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, delay: 0.1 }}
          className="text-4xl md:text-6xl lg:text-7xl font-bold text-center max-w-5xl leading-tight mb-6"
        >
          Beta is not constant.{" "}
          <span className="gradient-text">Neither should your portfolio be.</span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.25 }}
          className="text-lg md:text-xl text-gray-400 text-center max-w-2xl mb-10 leading-relaxed"
        >
          An LSTM deep learning system that predicts future beta volatility and triggers
          adaptive portfolio rebalancing — only when systematic risk is about to shift.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.38 }}
          className="flex flex-wrap gap-4 justify-center mb-16"
        >
          <Link
            href="/research"
            className="inline-flex items-center gap-2 px-6 py-3 bg-teal-400 hover:bg-teal-300 text-navy-900 font-semibold rounded-lg transition-all duration-200 hover:shadow-lg hover:shadow-teal-400/25"
          >
            Explore the Research <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="https://github.com/daishinkan7/ai_fintech"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 bg-navy-700 hover:bg-navy-600 text-white font-medium rounded-lg border border-navy-600 hover:border-teal-400/40 transition-all duration-200"
          >
            <Github className="w-4 h-4" /> View the Code
          </a>
          <Link
            href="/results"
            className="inline-flex items-center gap-2 px-6 py-3 text-gray-300 hover:text-white font-medium rounded-lg border border-navy-700 hover:border-navy-600 transition-all duration-200"
          >
            <BookOpen className="w-4 h-4" /> See Results
          </Link>
        </motion.div>

        {/* Hero Chart */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="w-full max-w-4xl"
        >
          <div className="card border-navy-600 overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="section-label mb-1">Live Beta Dynamics</p>
                <p className="text-sm text-gray-400">
                  Rolling 30d / 60d / 120d beta for RELIANCE.NS with VIX overlay
                </p>
              </div>
              <span className="badge-teal">RELIANCE.NS</span>
            </div>
            <HeroChart />
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 text-gray-500"
        >
          <span className="text-xs">Scroll to explore</span>
          <ChevronDown className="w-4 h-4 animate-bounce" />
        </motion.div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* KEY STATS */}
      {/* ------------------------------------------------------------------ */}
      <section ref={statsRef} className="py-20 px-4 border-y border-navy-700/60">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            animate={statsInView ? { opacity: 1 } : {}}
            transition={{ duration: 0.5 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            <StatItem value="49" suffix=" stocks" label="NIFTY50 Universe" delay={0} />
            <StatItem value="10" suffix=" years" label="2015–2024 Data" delay={0.1} />
            <StatItem value="4" suffix=" models" label="Compared" delay={0.2} />
            <StatItem value="1.42" label="Best Sharpe Ratio" delay={0.3} />
          </motion.div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* THE INNOVATION */}
      {/* ------------------------------------------------------------------ */}
      <section className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.p custom={0} variants={fadeUp} className="section-label mb-3">
              The Innovation
            </motion.p>
            <motion.h2
              custom={1}
              variants={fadeUp}
              className="text-3xl md:text-4xl font-bold text-white mb-4"
            >
              The Threshold Trigger
            </motion.h2>
            <motion.p custom={2} variants={fadeUp} className="text-gray-400 max-w-2xl mx-auto">
              Instead of rebalancing every month (wasteful) or every day (too expensive),
              AdaptiveBeta only acts when the LSTM predicts that beta instability is about to spike.
            </motion.p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: "Predict",
                color: "teal",
                icon: "🧠",
                desc: "LSTM predicts 20-day forward beta volatility using VIX, macro flows, and rolling beta features.",
              },
              {
                title: "Trigger",
                color: "amber",
                icon: "⚡",
                desc: "Rebalance ONLY when predicted betavol > 75th percentile threshold (or VIX > 25 for crisis override).",
              },
              {
                title: "Optimise",
                color: "purple",
                icon: "🎯",
                desc: "Route to max-Sharpe MVO (bull), min-variance (bear), or risk parity (transition) based on HMM regime.",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15, duration: 0.5 }}
                className="card-hover"
              >
                <div className="text-3xl mb-4">{item.icon}</div>
                <h3
                  className={`text-lg font-semibold mb-2 ${
                    item.color === "teal"
                      ? "text-teal-300"
                      : item.color === "amber"
                      ? "text-amber-300"
                      : "text-purple-300"
                  }`}
                >
                  {item.title}
                </h3>
                <p className="text-gray-400 text-sm leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* RESULTS PREVIEW */}
      {/* ------------------------------------------------------------------ */}
      <section className="py-24 px-4 bg-navy-800/40">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.p custom={0} variants={fadeUp} className="section-label mb-3">
              Results
            </motion.p>
            <motion.h2
              custom={1}
              variants={fadeUp}
              className="text-3xl md:text-4xl font-bold text-white mb-4"
            >
              Walk-Forward Backtest (2018–2024)
            </motion.h2>
            <motion.p custom={2} variants={fadeUp} className="text-gray-400">
              Out-of-sample performance vs. four competing strategies. No lookahead bias.
            </motion.p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {Object.entries(strategyMetrics).map(([key, s], i) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, scale: 0.97 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
              >
                <MetricsCard
                  label={s.label}
                  value={s.sharpe}
                  subValue={`${s.annualReturn}% annual return`}
                  delta={
                    key !== "buyHoldNifty"
                      ? s.sharpe - strategyMetrics.buyHoldNifty.sharpe
                      : undefined
                  }
                  deltaLabel="vs NIFTY50"
                  color={s.color}
                  highlight={key === "adaptiveBeta"}
                  metricName="Sharpe Ratio"
                />
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.5 }}
            className="text-center mt-10"
          >
            <Link
              href="/results"
              className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-300 font-medium transition-colors"
            >
              View full results and equity curves <ArrowRight className="w-4 h-4" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* DATA & METHODOLOGY TEASER */}
      {/* ------------------------------------------------------------------ */}
      <section className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <p className="section-label mb-3">Methodology</p>
              <h2 className="text-3xl font-bold text-white mb-5">
                India-specific signals that mainstream models ignore
              </h2>
              <ul className="space-y-3 text-gray-400">
                {[
                  "India VIX (not US VIX) — local fear gauge",
                  "RBI repo rate & 91-day T-bill spread",
                  "FII / DII monthly flow data",
                  "USD/INR and Brent crude exposure",
                  "NIFTY Bank / IT / FMCG sector rotation",
                  "Rolling multi-window beta (30d, 60d, 120d)",
                ].map((item) => (
                  <li key={item} className="flex items-start gap-2.5">
                    <span className="text-teal-400 mt-0.5 flex-shrink-0">✓</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Link
                  href="/research"
                  className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-300 font-medium transition-colors"
                >
                  Deep-dive into the methodology <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="grid grid-cols-2 gap-4"
            >
              {[
                { label: "Data Sources", value: "10", icon: "🗄️" },
                { label: "Features / Stock", value: "~35", icon: "⚙️" },
                { label: "Training Period", value: "2015–21", icon: "📅" },
                { label: "OOS Test Period", value: "2018–24", icon: "🔍" },
                { label: "Walk-Forward Folds", value: "6", icon: "🔄" },
                { label: "LSTM Parameters", value: "~45K", icon: "🧠" },
              ].map((item) => (
                <div key={item.label} className="card text-center p-4">
                  <div className="text-2xl mb-2">{item.icon}</div>
                  <div className="text-xl font-bold text-white">{item.value}</div>
                  <div className="text-xs text-gray-400 mt-1">{item.label}</div>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* CTA SECTION */}
      {/* ------------------------------------------------------------------ */}
      <section className="py-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="card border-teal-400/20 bg-gradient-to-br from-navy-800 to-navy-700"
          >
            <p className="section-label mb-4">Open Source</p>
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
              Full pipeline on GitHub
            </h2>
            <p className="text-gray-400 mb-8">
              4 Colab notebooks, complete Python src package, Streamlit dashboard, and this website.
              MIT licensed — fork, extend, and build on it.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <a
                href="https://github.com/daishinkan7/ai_fintech"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 bg-teal-400 hover:bg-teal-300 text-navy-900 font-semibold rounded-lg transition-all duration-200"
              >
                <Github className="w-4 h-4" /> View on GitHub
              </a>
              <Link
                href="/architecture"
                className="inline-flex items-center gap-2 px-6 py-3 bg-navy-700 hover:bg-navy-600 text-white font-medium rounded-lg border border-navy-600 transition-all duration-200"
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
