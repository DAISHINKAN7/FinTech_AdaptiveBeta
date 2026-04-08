"""Central configuration for the AdaptiveBeta project.

All constants, paths, tickers, and hyperparameters live here.

Data root is resolved in this order:
  1. Environment variable ADAPTIVEBETA_ROOT (set this if you want to override)
  2. Google Drive path (if running in Colab and Drive is mounted)
  3. Local: project_root/data/  (default for local runs)

To run locally, put your data under:
  adaptivebeta/data/stocks/all_stocks_prices.csv  etc.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adaptivebeta/

def _resolve_root() -> Path:
    """Return the data root directory."""
    # 1. Explicit env override
    env = os.environ.get("ADAPTIVEBETA_ROOT")
    if env:
        return Path(env)
    # 2. Google Colab + Drive
    colab_path = Path("/content/drive/MyDrive/AI_Finance_Project")
    if colab_path.exists():
        return colab_path
    # 3. Local default  ← where YOU put your data
    return _PROJECT_ROOT


ROOT: Path = _resolve_root()

# Sub-directories
DATA_DIR:     Path = ROOT / "data"        # raw CSVs live here (locally)
FEATURES_DIR: Path = ROOT / "features"   # computed features saved here
MODELS_DIR:   Path = ROOT / "models"     # trained model checkpoints
RESULTS_DIR:  Path = ROOT / "results"    # backtest outputs

# For Colab compatibility keep the string form too
DRIVE_ROOT: str = str(ROOT)

# Create output dirs if they don't exist yet
for _d in [FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw data file paths  (relative to DATA_DIR)
# ---------------------------------------------------------------------------
RAW_FILES: dict[str, str] = {
    "prices":       "stocks/all_stocks_prices.csv",
    "volume":       "stocks/all_stocks_volume.csv",
    "nifty50":      "market/nifty50.csv",
    "india_vix":    "market/india_vix.csv",
    "usdinr":       "macro/usdinr.csv",
    "crude_oil":    "macro/crude_oil.csv",
    "risk_free_rate": "macro/risk_free_rate_91d_daily.csv",
    "repo_rate":    "macro/repo_rate_daily.csv",
    "fii_flows":    "flows/fii_flows.csv",
    "dii_flows":    "flows/dii_flows.csv",
    "nifty_bank":   "sector/nifty_bank.csv",
    "nifty_it":     "sector/nifty_it.csv",
    "nifty_fmcg":   "sector/nifty_fmcg.csv",
}

# ---------------------------------------------------------------------------
# Stock universe — 49 tickers (TMPV.NS excluded: only 87 rows post-demerger)
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TECHM.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DRREDDY.NS",
    "DIVISLAB.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS", "BRITANNIA.NS", "GRASIM.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "UPL.NS", "BPCL.NS", "IOC.NS", "HINDALCO.NS",
]

# Late IPO tickers — pre-IPO NaN rows expected; filled with 0 for returns
LATE_IPO_TICKERS: list[str] = ["HDFCLIFE.NS", "SBILIFE.NS"]

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
BETA_WINDOWS:        list[int] = [30, 60, 120]
TARGET_HORIZON:      int  = 20      # predict betavol 20 trading days ahead
BETAVOL_BASE_WINDOW: int  = 60      # 60d betavol as prediction target

# ---------------------------------------------------------------------------
# Threshold & regime constants
# ---------------------------------------------------------------------------
THRESHOLD_QUANTILE: float = 0.70    # 70th pct of training betavol (was 0.55 — bug: comment said 75th but value was 55th)
VIX_STRESS_LEVEL:   float = 22.0    # VIX override level (lowered from 25: fires earlier in COVID-style events)

# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
TRAIN_END:            str = "2021-12-31"
TEST_START:           str = "2022-01-01"
BACKTEST_START_YEAR:  int = 2015
BACKTEST_END_YEAR:    int = 2025
TRAIN_YEARS:          int = 3
TEST_YEARS:           int = 1

# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------
BROKERAGE: float = 0.0005   # 0.05% per trade
SLIPPAGE:  float = 0.0003   # 0.03% market impact

# ---------------------------------------------------------------------------
# Momentum strategy
# ---------------------------------------------------------------------------
MOMENTUM_TOP_N: int = 15
# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
MAX_WEIGHT:          float = 0.15
MIN_WEIGHT:          float = 0.01
DEFAULT_BETA_TARGET: float = 0.85

# ---------------------------------------------------------------------------
# LSTM hyperparameters
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN: int   = 30
LSTM_HIDDEN:  int   = 128
LSTM_LAYERS:  int   = 2
LSTM_DROPOUT: float = 0.2
LSTM_BATCH:   int   = 256
LSTM_EPOCHS:  int   = 50
LSTM_LR:      float = 1e-3
LSTM_PATIENCE:int   = 10

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
STRATEGIES: list[str] = [
    "adaptive_beta",
    "static_capm_mvo",
    "kalman_mvo",
    "equal_weight",
    "buy_hold_nifty",
]

# ---------------------------------------------------------------------------
# Stress events
# ---------------------------------------------------------------------------
STRESS_EVENTS: dict[str, tuple[str, str]] = {
    "COVID Crash":    ("2020-02-01", "2020-05-01"),
    "ADANI Crisis":   ("2023-01-25", "2023-03-15"),
    "2018 Rate Hike": ("2018-09-01", "2018-11-30"),
    "IL&FS Crisis":   ("2018-08-01", "2018-10-31"),
}
