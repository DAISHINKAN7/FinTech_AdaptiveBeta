"use client";

import { motion } from "framer-motion";
import PipelineFlow from "@/components/PipelineFlow";
import { pipelineStages } from "@/lib/sampleData";

export default function ArchitecturePage() {
  return (
    <div className="min-h-screen py-24 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-16"
        >
          <p className="section-label mb-3">System Architecture</p>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-5">
            How AdaptiveBeta Works
          </h1>
          <p className="text-gray-400 max-w-2xl text-lg">
            A modular, end-to-end Python pipeline from raw market data to adaptive portfolio weights,
            with a Next.js portfolio site and Streamlit dashboard for monitoring.
          </p>
        </motion.div>

        {/* Pipeline Flow */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-16"
        >
          <h2 className="text-xl font-semibold text-white mb-6">End-to-End Pipeline</h2>
          <PipelineFlow showDetails />
        </motion.div>

        {/* Stage Detail Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-16"
        >
          <h2 className="text-xl font-semibold text-white mb-6">Pipeline Stages</h2>
          <div className="grid md:grid-cols-2 gap-5">
            {pipelineStages.map((stage, i) => (
              <motion.div
                key={stage.id}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="card-hover"
              >
                <div className="flex items-start gap-4">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center text-lg flex-shrink-0"
                    style={{ background: `${stage.color}20`, border: `1px solid ${stage.color}40` }}
                  >
                    {i + 1}
                  </div>
                  <div>
                    <h3 className="font-semibold text-white mb-1">{stage.label}</h3>
                    <p className="text-sm text-gray-400 mb-2">{stage.description}</p>
                    <p className="text-xs font-mono text-gray-500">{stage.detail}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Project Structure */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-16"
        >
          <h2 className="text-xl font-semibold text-white mb-6">Repository Structure</h2>
          <div className="card font-mono text-sm text-gray-300 overflow-x-auto">
            <pre className="whitespace-pre leading-7">
{`adaptivebeta/
├── notebooks/
│   ├── 01_feature_engineering.ipynb    ← Load data, compute betas
│   ├── 02_beta_prediction_models.ipynb ← Train LSTM, XGBoost, Kalman
│   ├── 03_portfolio_optimisation.ipynb ← Threshold + optimisers
│   └── 04_backtesting.ipynb            ← Walk-forward + results
│
├── src/
│   ├── config.py              ← All constants, tickers, hyperparams
│   ├── data/
│   │   ├── loader.py          ← Load & align all datasets
│   │   └── validator.py       ← Data quality checks
│   ├── features/
│   │   ├── returns.py         ← Log returns
│   │   ├── beta.py            ← Rolling OLS beta + betavol
│   │   └── macro.py           ← VIX, macro, flow features
│   ├── models/
│   │   ├── lstm.py            ← BetaLSTM + training loop
│   │   ├── xgboost_model.py   ← XGBoost + SHAP
│   │   ├── kalman.py          ← Kalman filter beta
│   │   └── hmm_regime.py      ← HMM regime classifier
│   ├── portfolio/
│   │   ├── signal.py          ← Threshold trigger + routing
│   │   ├── optimiser.py       ← MVO, min-var, risk parity
│   │   └── constraints.py     ← Weight cleaning + validation
│   ├── backtest/
│   │   ├── engine.py          ← WalkForwardBacktester
│   │   ├── metrics.py         ← Performance metrics + stress
│   │   └── transaction_costs.py
│   └── utils/
│       ├── logging.py         ← Structured logging
│       └── plotting.py        ← Matplotlib/Plotly charts
│
├── dashboard.py               ← Streamlit results dashboard
├── requirements.txt           ← All Python dependencies
│
└── website/                   ← This Next.js portfolio site
    ├── app/                   ← App Router pages
    ├── components/            ← Reusable React components
    └── lib/                   ← Utilities + sample data`}
            </pre>
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-xl font-semibold text-white mb-6">Technology Stack</h2>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-5">
            {[
              {
                category: "ML Pipeline",
                color: "#1D9E75",
                items: ["PyTorch (LSTM)", "XGBoost + SHAP", "PyKalman", "hmmlearn", "scikit-learn"],
              },
              {
                category: "Portfolio Optimisation",
                color: "#7F77DD",
                items: ["PyPortfolioOpt (MVO)", "riskfolio-lib (RP)", "cvxpy (constraints)", "Ledoit-Wolf cov"],
              },
              {
                category: "Data & Features",
                color: "#EF9F27",
                items: ["pandas + NumPy", "Google Drive API", "NIFTY50 (NSE)", "India VIX, RBI data"],
              },
              {
                category: "Backtesting",
                color: "#D85A30",
                items: ["Walk-forward CV", "quantstats", "Custom cost model", "Stress event analysis"],
              },
              {
                category: "Dashboard",
                color: "#5DCAA5",
                items: ["Streamlit", "Plotly", "Google Colab", "ngrok (sharing)"],
              },
              {
                category: "Website",
                color: "#AFA9EC",
                items: ["Next.js 14", "Tailwind CSS", "Framer Motion", "Recharts", "TypeScript"],
              },
            ].map((stack) => (
              <div
                key={stack.category}
                className="card"
                style={{ borderColor: `${stack.color}25` }}
              >
                <div
                  className="text-xs font-semibold tracking-wider uppercase mb-3"
                  style={{ color: stack.color }}
                >
                  {stack.category}
                </div>
                <ul className="space-y-1.5">
                  {stack.items.map((item) => (
                    <li key={item} className="flex items-center gap-2 text-sm text-gray-300">
                      <span style={{ color: stack.color }}>›</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
