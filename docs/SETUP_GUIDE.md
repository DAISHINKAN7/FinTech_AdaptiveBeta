# AdaptiveBeta — Complete Setup & Usage Guide
 
This guide walks you through every step to get AdaptiveBeta running from scratch, including the Python ML pipeline, the Plotly Dash dashboard (`app.py`), and the Next.js website.
 
---
 
## Table of Contents
 
1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Data Preparation](#4-data-preparation)
5. [Running the ML Pipeline](#5-running-the-ml-pipeline)
   - [Script 01 — Feature Engineering](#script-01--feature-engineering)
   - [Script 02 — Model Training](#script-02--model-training)
   - [Script 03 — Portfolio Optimisation](#script-03--portfolio-optimisation)
   - [Script 04 — Walk-Forward Backtest](#script-04--walk-forward-backtest)
   - [One-Shot Pipeline Runner](#one-shot-pipeline-runner)
6. [Dashboard — Plotly Dash (app.py)](#6-dashboard--plotly-dash-apppy)
7. [Dashboard — Streamlit (dashboard.py)](#7-dashboard--streamlit-dashboardpy)
8. [Next.js Website](#8-nextjs-website)
9. [Understanding the Outputs](#9-understanding-the-outputs)
10. [Troubleshooting](#10-troubleshooting)
 
---
 
## 1. Prerequisites
 
### Python
- Python **3.10+** recommended (tested on 3.11)
- `pip` or `conda` package manager
 
### Node.js (for website only)
- Node.js **18+** and npm **9+**
- Install from [nodejs.org](https://nodejs.org)
 
### Hardware
- **Minimum**: Any modern CPU — full pipeline takes ~2–4 hours
- **Recommended**: Apple M-series (MPS) or NVIDIA GPU (CUDA) — reduces training to ~10 minutes
- **RAM**: 8 GB minimum, 16 GB recommended (stacked_features.csv is ~93 MB)
 
---
 
## 2. Clone the Repository
 
```bash
git clone https://github.com/daishinkan7/fintech_adaptivebeta.git
cd FinTech_AdaptiveBeta
```
 
---
 
## 3. Python Environment Setup
 
### Step 3.1 — Create a virtual environment
 
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
 
# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
 
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```
 
You should see `(venv)` in your terminal prompt.
 
### Step 3.2 — Install all dependencies
 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
 
This installs the full stack: PyTorch, XGBoost, PyPortfolioOpt, CVXPY, pykalman, hmmlearn, Plotly, Dash, Streamlit, and more.
 
**Expected install time:** 3–8 minutes depending on internet speed.
 
### Step 3.3 — (Optional) GPU support
 
**NVIDIA CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
 
**Apple Silicon (MPS — auto-detected):**
No extra steps needed. PyTorch 2.1+ detects MPS automatically.
 
### Step 3.4 — Verify installation
 
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
python -c "import xgboost, pypfopt, dash, streamlit; print('All key packages OK')"
```
 
---
 
## 4. Data Preparation
 
### Required directory structure
 
Create a `data/` folder at the project root with the following layout:
 
```
FinTech_AdaptiveBeta/
└── data/
    ├── stocks/
    │   ├── all_stocks_prices.csv
    │   └── all_stocks_volume.csv
    ├── market/
    │   ├── nifty50.csv
    │   └── india_vix.csv
    ├── macro/
    │   ├── usdinr.csv
    │   ├── crude_oil.csv
    │   ├── risk_free_rate_91d_daily.csv
    │   └── repo_rate_daily.csv
    ├── flows/
    │   ├── fii_flows.csv
    │   └── dii_flows.csv
    └── sector/
        ├── nifty_bank.csv
        ├── nifty_it.csv
        └── nifty_fmcg.csv
```
 
### CSV format requirements
 
| File | Columns | Frequency | Notes |
|---|---|---|---|
| `all_stocks_prices.csv` | `date`, `RELIANCE.NS`, `TCS.NS`, ... (49 tickers) | Daily | Close prices |
| `all_stocks_volume.csv` | `date`, same 49 tickers | Daily | Volume |
| `nifty50.csv` | `date`, `Open`, `High`, `Low`, `Close`, `Volume` | Daily | Index |
| `india_vix.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | India VIX |
| `usdinr.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | FX rate |
| `crude_oil.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | WTI crude |
| `risk_free_rate_91d_daily.csv` | `date`, `risk_free_rate` | Daily | 91-day T-bill yield |
| `repo_rate_daily.csv` | `date`, `repo_rate` | Daily | RBI repo rate |
| `fii_flows.csv` | `date`, `fii_net_investment` | Monthly | INR crores |
| `dii_flows.csv` | `date`, `dii_net_investment` | Monthly | INR crores |
| `nifty_bank.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | Sector index |
| `nifty_it.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | Sector index |
| `nifty_fmcg.csv` | `date`, `Open`, `High`, `Low`, `Close` | Daily | Sector index |
 
**All `date` columns must be parseable** (e.g. `2015-01-02`, `02-01-2015`, `Jan 2, 2015`).
 
**Minimum required** to run the pipeline: `stocks/all_stocks_prices.csv` + `market/nifty50.csv`.
All other files degrade gracefully with zero-fill if missing.
 
### Data sources (where to get the data)
 
- **Stocks, market indices:** NSE/BSE via `yfinance`, Quandl, or Bloomberg
- **India VIX:** NSE website or `yfinance` ticker `^INDIAVIX`
- **USD/INR, Crude Oil:** Investing.com, Yahoo Finance
- **Risk-free rate, Repo rate:** RBI website (DBIE portal)
- **FII/DII flows:** SEBI/NSE monthly publications, moneycontrol.com
- **Sector indices:** NSE (NIFTY Bank: `^NSEBANK`, IT: `^CNXIT`, FMCG: `^CNXFMCG`)
 
---
 
## 5. Running the ML Pipeline
 
The pipeline has **4 sequential stages**. Always run them in order (01 → 02 → 03 → 04) since each stage depends on outputs from the previous one.
 
### Script 01 — Feature Engineering
 
**What it does:**
- Loads all raw CSV data and aligns to the NIFTY50 trading calendar
- Computes log returns for all 49 stocks
- Computes rolling OLS beta (30d, 60d, 120d windows) and beta volatility
- Builds market-level feature matrix (24 features: VIX, macro, rates, flows, sectors)
- Builds per-stock feature matrices with momentum, RSI, MA distance
- Stacks all features into one 118,719 × 44 matrix
- Saves all outputs to `features/`
 
**Run:**
```bash
python scripts/01_feature_engineering.py
```
 
**Expected output:**
```
Feature engineering complete in 9.5 seconds.
```
 
**Outputs written:**
- `features/stacked_features.csv` (118,719 rows × 44 cols, ~93 MB)
- `features/market_features.csv`
- `features/beta{30,60,120}d.csv`
- `features/betavol_{30,60,120}d.csv`
- `features/target_betavol_20d_ahead.csv`
- `results/feature_overview.png`
 
---
 
### Script 02 — Model Training
 
**What it does:**
- Loads `stacked_features.csv`, scales features, splits train/test at 2021-12-31
- **Model A — XGBoost**: trains gradient boosting with early stopping, computes SHAP importance
- **Model B — Kalman Filter**: fits state-space EM for time-varying beta for all 49 stocks
- **Model C — LSTM**: trains 2-layer LSTM (128 hidden, 230,785 params) with HuberLoss, AdamW, early stopping
- **Model D — HMM**: fits 3-state Hidden Markov Model for bull/bear/transition regime labels
- Calibrates the betavol rebalancing threshold (Q70 of XGBoost training predictions)
- Saves all model artifacts to `models/`
 
**Run:**
```bash
# Recommended: auto-detect device (CUDA > MPS > CPU)
python scripts/02_train_models.py --device auto
 
# Skip Kalman (saves ~5–20 min depending on hardware)
python scripts/02_train_models.py --device auto --fast
 
# Force CPU
python scripts/02_train_models.py --device cpu
 
# Apple Silicon (explicit MPS)
python scripts/02_train_models.py --device mps
```
 
**CLI flags:**
 
| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | Hardware: `auto`, `cuda`, `mps`, `cpu` |
| `--epochs N` | 50 | Max LSTM training epochs (early stopping may fire sooner) |
| `--fast` | off | Skip Kalman filter (saves time; Kalman strategy in backtest will use fallback) |
| `--no-xgb` | off | Skip XGBoost + SHAP computation |
 
**Expected runtime:**
- Apple MPS: ~8–10 minutes (Kalman: ~5 min, LSTM: ~3 min)
- NVIDIA CUDA: ~5–8 minutes
- CPU only: ~2–4 hours (Kalman is the bottleneck)
 
**Expected output:**
```
XGBoost — MAE: 0.0391 | Direction Acc: 0.765
LSTM — MAE: 0.0661 | Direction Acc: 0.504  (early stopped at epoch 12)
Kalman — MAE: 0.0749
Saved signal_config.json: threshold=0.1794 (model=xgboost)
Model training complete in ~499 seconds.
```
 
**Outputs written:**
- `models/xgb_model.pkl`
- `models/lstm_best.pt`
- `models/scaler.pkl`
- `models/feature_cols.txt`
- `models/signal_config.json`
- `features/kalman_betas.csv`
- `features/hmm_regimes.csv`
- `results/shap_importance.png`
- `results/lstm_training_curves.png`
- `results/hmm_regimes.png`
 
---
 
### Script 03 — Portfolio Optimisation Validation
 
**What it does:**
- Loads `signal_config.json` and beta data
- Runs all three optimiser modes (Max-Sharpe, Min-Variance, Risk-Parity) at the training cutoff date
- Computes signal frequency distribution over the training period
- Saves weights and comparison tables
 
**Run:**
```bash
python scripts/03_portfolio_optimisation.py
```
 
**Expected output:**
```
max_sharpe_beta_constrained  → 11 stocks, portfolio_beta=0.850
min_variance                 → 21 stocks, portfolio_beta=0.602
Signal: HOLD 59.6% | REBALANCE 26.9% | MIN_VARIANCE 13.4%
```
 
**Outputs written:**
- `features/optimiser_weights_max_sharpe_beta_constrained.csv`
- `features/optimiser_weights_min_variance.csv`
- `features/optimiser_weights_risk_parity.csv`
- `results/optimiser_comparison.csv`
- `results/signal_frequency.csv`
- `results/signal_threshold_demo.png`
- `results/optimiser_weights_chart.png`
 
---
 
### Script 04 — Walk-Forward Backtest
 
**What it does:**
- Runs a strict walk-forward backtest (3-year train, 1-year test, 1-year step)
- Tests all 6 strategies: AdaptiveBeta, Static CAPM MVO, Kalman MVO, Equal Weight, Buy & Hold NIFTY50, Momentum-Quality
- Applies 0.08% round-trip transaction costs on each rebalance
- Computes performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Volatility
- Analyses 4 stress events: COVID Crash, ADANI Crisis, 2018 Rate Hike, IL&FS Crisis
- Generates all charts
 
**Run:**
```bash
python scripts/04_backtest.py
 
# Skip Kalman strategy (if you ran --fast in step 02)
python scripts/04_backtest.py --fast
```
 
**Expected runtime:** ~25–30 minutes
 
**Expected output:**
```
CAGR(%)   Sharpe  Sortino  MaxDD(%)   Calmar    Vol(%)
AdaptiveBeta          9.24    0.785    1.047    -17.36    0.532    11.26
Static CAPM MVO      11.38    0.587    0.716    -35.33    0.322    18.35
Kalman-Beta MVO      10.92    0.561    0.676    -35.44    0.308    18.48
Equal Weight         15.86    0.864    0.992    -38.23    0.415    17.04
Buy & Hold NIFTY50   12.16    0.668    0.781    -38.44    0.316    17.19
Momentum-Quality     17.47    0.893    1.028    -38.51    0.454    18.03
```
 
**Outputs written:**
- `results/performance_metrics.csv`
- `results/stress_analysis.csv`
- `results/strategy_returns.csv`
- `results/adaptive_beta_returns.csv` (and one per strategy)
- `results/equity_curves.png`
- `results/drawdown_curves.png`
- `results/rolling_sharpe.png`
- `results/stress_heatmap.png`
- `results/monthly_returns_heatmap_adaptive_beta.png`
- `results/monthly_returns_heatmap_momentum_quality.png`
 
---
 
### One-Shot Pipeline Runner
 
Run the entire pipeline with a single command:
 
```bash
python run_pipeline.py
```
 
**Options:**
 
| Flag | Description |
|---|---|
| `--device auto` | Hardware selection (default: auto) |
| `--fast` | Skip Kalman filter |
| `--start N` | Resume from script N (1–4) |
| `--epochs N` | Override LSTM epochs |
 
**Example — resume from backtest only:**
```bash
python run_pipeline.py --start 4
```
 
---
 
## 6. Dashboard — Plotly Dash (app.py)
 
`app.py` is a full interactive dashboard built with **Plotly Dash** and **Dash Bootstrap Components**.
 
### Prerequisites
 
Make sure the full pipeline has been run at least once (you need `results/performance_metrics.csv`, `results/strategy_returns.csv`, etc.).
 
### Run
 
```bash
# From the project root with venv activated:
python app.py
```
 
The dashboard starts at **http://localhost:8050**
 
### Features
 
- **Overview Tab**: Key performance metrics cards (CAGR, Sharpe, MaxDD, Calmar)
- **Equity Curves Tab**: Interactive cumulative wealth chart for all 6 strategies
- **Drawdown Tab**: Underwater equity chart, identifying stress periods
- **Rolling Sharpe Tab**: 1-year rolling Sharpe ratio for all strategies
- **Stress Events Tab**: Heatmap and bar charts of stress event performance
- **Monthly Returns Tab**: Calendar return heatmap for AdaptiveBeta
- **Signals Tab**: Betavol vs threshold visualisation, signal frequency pie chart
 
### Customisation
 
Edit `app.py` to:
- Change the port: `app.run(debug=True, port=8050)` → change `8050`
- Change the theme: `dbc.themes.DARKLY` → any Bootstrap theme
- Add/remove tabs in the `dbc.Tabs` section
 
---
 
## 7. Dashboard — Streamlit (dashboard.py)
 
An alternative dashboard using **Streamlit**.
 
### Run
 
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```
 
### Features
 
Similar to the Dash dashboard but in Streamlit's component model.
 
---
 
## 8. Next.js Website
 
A production-quality static website showcasing the research with interactive charts.
 
### Prerequisites
 
- Node.js 18+ and npm 9+
- Check: `node --version` and `npm --version`
 
### Step 8.1 — Navigate to website directory
 
```bash
cd website
```
 
### Step 8.2 — Install Node dependencies
 
```bash
npm install
```
 
**Expected install time:** 1–3 minutes. This installs Next.js 14, Tailwind CSS, Framer Motion, Recharts, and Lucide icons.
 
### Step 8.3 — Run development server
 
```bash
npm run dev
```
 
Open **http://localhost:3000** in your browser.
 
### Step 8.4 — Build for production (optional)
 
```bash
npm run build
npm run start
# Production server at http://localhost:3000
```
 
### Step 8.5 — Static export (optional)
 
```bash
npm run build
# Output in website/out/ directory (deploy to any static host)
```
 
### Website Pages
 
| Route | Description |
|---|---|
| `/` | Hero page with animated performance stats |
| `/research` | Academic framing, problem statement, methodology |
| `/results` | Full performance tables, charts, stress events |
| `/demo` | Interactive portfolio simulator |
| `/architecture` | System architecture diagram, module breakdown |
 
### Updating website data
 
The website uses hardcoded sample data in `website/lib/sampleData.ts`. To update with latest backtest results:
 
1. Open `website/lib/sampleData.ts`
2. Update `strategyMetrics` object with values from `results/performance_metrics.csv`
3. Update `stressEvents` with values from `results/stress_analysis.csv`
4. Save and the dev server hot-reloads automatically
 
---
 
## 9. Understanding the Outputs
 
### Feature files (`features/`)
 
| File | Description |
|---|---|
| `stacked_features.csv` | 118,719 rows (49 stocks × 2,424 dates). Columns: `date`, `ticker`, 42 feature columns |
| `market_features.csv` | 2,743 rows, 24 market-level features (VIX, macro, sectors, flows) |
| `beta60d.csv` | 2,743 rows × 49 columns — 60-day rolling OLS beta per stock |
| `betavol_60d.csv` | 2,743 rows × 49 columns — 60-day rolling beta volatility per stock |
| `target_betavol_20d_ahead.csv` | 2,625 rows × 49 columns — 20-day forward shifted betavol (LSTM target) |
| `kalman_betas.csv` | 2,743 rows × 49 columns — Kalman-filtered time-varying beta |
| `hmm_regimes.csv` | 2,743 rows — daily regime labels: `bull`, `bear`, `transition` |
| `optimiser_weights_*.csv` | Portfolio weights for each optimisation mode |
 
### Model files (`models/`)
 
| File | Description |
|---|---|
| `xgb_model.pkl` | Fitted XGBoost model (used for signal generation) |
| `lstm_best.pt` | Best LSTM checkpoint saved by early stopping |
| `scaler.pkl` | `sklearn.preprocessing.StandardScaler` fitted on training data only |
| `feature_cols.txt` | Ordered list of 42 feature names (order matters for scaler) |
| `signal_config.json` | `{"betavol_threshold": 0.1794, "vix_threshold": 22.0, "best_model": "xgboost"}` |
 
### Results files (`results/`)
 
| File | Description |
|---|---|
| `performance_metrics.csv` | CAGR, Sharpe, Sortino, MaxDD, Calmar, Vol for all 6 strategies |
| `stress_analysis.csv` | Cumulative return and MaxDD for each strategy in each stress event |
| `strategy_returns.csv` | Daily returns for all strategies (wide format, dates as index) |
| `shap_importance.csv` | XGBoost SHAP mean absolute values per feature |
| `optimiser_comparison.csv` | Expected return/vol/Sharpe for each optimiser mode |
| `signal_frequency.csv` | Count and % for HOLD / REBALANCE / MIN_VARIANCE signals |
 
---
 
## 10. Troubleshooting
 
### `ModuleNotFoundError: No module named 'xxx'`
 
Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```
 
### `FileNotFoundError: data/stocks/all_stocks_prices.csv`
 
The `data/` directory is not included in the repository. You must supply your own CSV files. See [Section 4 — Data Preparation](#4-data-preparation).
 
### `FileNotFoundError: features/stacked_features.csv`
 
You need to run Script 01 first:
```bash
python scripts/01_feature_engineering.py
```
 
### LSTM training is very slow
 
```bash
# Check what device PyTorch detected:
python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
 
# Force MPS on Apple Silicon:
python scripts/02_train_models.py --device mps
 
# Or skip Kalman and reduce epochs:
python scripts/02_train_models.py --fast --epochs 20
```
 
### `The problem doesn't have a solution with actual input parameters` (Script 03)
 
This is a known warning from the CVXPY risk-parity optimiser when the covariance matrix is ill-conditioned. The code falls back to Min-Variance automatically — this is expected behaviour and does not break anything.
 
### `npm install` fails (website)
 
```bash
# Clear npm cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```
 
### Dash app shows no charts
 
Run the full pipeline first (all 4 scripts) to generate the required `results/` CSV files. The Dash app reads from disk on startup.
 
### Memory errors on large machines
 
The stacked features CSV is ~93 MB. If you're on a machine with <8 GB RAM:
```bash
# Edit src/config.py and reduce TICKERS list
# Or run with Python's garbage collection:
python -X faulthandler scripts/04_backtest.py
```
 
---
 
*For questions or issues, open a GitHub issue in the repository.*