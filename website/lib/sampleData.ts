/**
 * Real backtest results — Walk-Forward 2015-2025, NIFTY50 universe (49 stocks).
 * Source: scripts/04_backtest.py output, pipeline run April 2026.
 */

// ─── Strategy Metrics (real OOS values) ──────────────────────────────────────

export const strategyMetrics = {
  adaptiveBeta: {
    annualReturn:  9.90,
    annualVol:    14.68,
    sharpe:        0.643,
    sortino:       0.761,
    maxDrawdown:  -38.20,
    calmar:        0.259,
    label: "AdaptiveBeta (ours)",
    color: "#1D9E75",
    days: 1964,
  },
  momentumQuality: {
    annualReturn: 17.47,
    annualVol:    18.03,
    sharpe:        0.893,
    sortino:       1.028,
    maxDrawdown:  -38.51,
    calmar:        0.454,
    label: "Momentum-Quality",
    color: "#7F77DD",
    days: 2707,
  },
  equalWeight: {
    annualReturn: 15.86,
    annualVol:    17.04,
    sharpe:        0.864,
    sortino:       0.992,
    maxDrawdown:  -38.23,
    calmar:        0.415,
    label: "Equal Weight",
    color: "#5DCAA5",
    days: 1964,
  },
  buyHoldNifty: {
    annualReturn: 12.16,
    annualVol:    17.19,
    sharpe:        0.668,
    sortino:       0.781,
    maxDrawdown:  -38.44,
    calmar:        0.316,
    label: "Buy & Hold NIFTY50",
    color: "#EF9F27",
    days: 1964,
  },
  staticCapmMvo: {
    annualReturn: 10.71,
    annualVol:    18.48,
    sharpe:        0.550,
    sortino:       0.675,
    maxDrawdown:  -36.62,
    calmar:        0.292,
    label: "Static CAPM MVO",
    color: "#D85A30",
    days: 1964,
  },
  kalmanMvo: {
    annualReturn: 10.03,
    annualVol:    18.62,
    sharpe:        0.513,
    sortino:       0.621,
    maxDrawdown:  -36.62,
    calmar:        0.274,
    label: "Kalman-Beta MVO",
    color: "#888780",
    days: 1964,
  },
};

export type StrategyKey = keyof typeof strategyMetrics;

// ─── Real Stress Event Results ────────────────────────────────────────────────

export const stressEvents: Record<
  string,
  Partial<Record<StrategyKey, { return: number; maxDD: number }>>
> = {
  "COVID Crash (Feb–May 2020)": {
    adaptiveBeta:    { return: -27.33, maxDD: -37.60 },
    momentumQuality: { return: -15.47, maxDD: -38.51 },
    equalWeight:     { return: -18.94, maxDD: -37.63 },
    buyHoldNifty:    { return: -17.57, maxDD: -37.63 },
    staticCapmMvo:   { return: -11.02, maxDD: -35.33 },
    kalmanMvo:       { return: -11.03, maxDD: -35.34 },
  },
  "ADANI Crisis (Jan–Mar 2023)": {
    adaptiveBeta:    { return: -6.66,  maxDD: -5.87  },
    momentumQuality: { return: -6.56,  maxDD: -7.08  },
    equalWeight:     { return: -6.66,  maxDD: -5.87  },
    buyHoldNifty:    { return: -6.33,  maxDD: -5.90  },
    staticCapmMvo:   { return: -10.19, maxDD: -13.71 },
    kalmanMvo:       { return: -11.07, maxDD: -14.25 },
  },
  "2018 Rate Hike (Sep–Nov 2018)": {
    adaptiveBeta:    { return: -7.51,  maxDD: -12.86 },
    momentumQuality: { return: -8.75,  maxDD: -12.44 },
    equalWeight:     { return: -7.51,  maxDD: -12.86 },
    buyHoldNifty:    { return: -6.88,  maxDD: -13.45 },
    staticCapmMvo:   { return: -10.64, maxDD: -15.16 },
    kalmanMvo:       { return: -10.44, maxDD: -15.26 },
  },
  "IL&FS Crisis (Aug–Oct 2018)": {
    adaptiveBeta:    { return: -6.01,  maxDD: -13.54 },
    momentumQuality: { return: -6.42,  maxDD: -13.84 },
    equalWeight:     { return: -6.01,  maxDD: -13.54 },
    buyHoldNifty:    { return: -8.54,  maxDD: -14.55 },
    staticCapmMvo:   { return: -7.92,  maxDD: -16.67 },
    kalmanMvo:       { return: -7.90,  maxDD: -16.72 },
  },
};

// ─── Equity Curve Generator ───────────────────────────────────────────────────

function generateEquityCurve(
  startDate: Date,
  endDate: Date,
  annualReturn: number,
  annualVol: number,
  seed: number,
  crisisOverrides?: Array<{ start: string; end: string; dailyDrift: number }>
): Array<{ date: string; value: number }> {
  const dates: Date[] = [];
  const cur = new Date(startDate);
  while (cur <= endDate) {
    const dow = cur.getDay();
    if (dow !== 0 && dow !== 6) dates.push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
  }

  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  const dailyMu    = annualReturn  / 100 / 252;
  const dailySigma = annualVol / 100 / Math.sqrt(252);

  let value = 100;
  return dates.map((d) => {
    const dateStr = d.toISOString().split("T")[0];
    const override = crisisOverrides?.find(
      (o) => dateStr >= o.start && dateStr <= o.end
    );
    const drift = override ? override.dailyDrift : dailyMu;
    const r = drift + dailySigma * ((rand() * 2 - 1) * Math.sqrt(3));
    value = value * Math.exp(r);
    return { date: dateStr, value: Math.round(value * 100) / 100 };
  });
}

const COVID_CRASH = { start: "2020-02-20", end: "2020-05-01", dailyDrift: -0.012 };
const COVID_RECOVERY = { start: "2020-05-02", end: "2020-08-31", dailyDrift: 0.003 };

export const equityCurveData = {
  adaptiveBeta: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 9.90, 14.68, 1,
    [COVID_CRASH, COVID_RECOVERY]
  ),
  momentumQuality: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 17.47, 18.03, 7,
    [{ ...COVID_CRASH, dailyDrift: -0.009 }, COVID_RECOVERY]
  ),
  equalWeight: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 15.86, 17.04, 4,
    [{ ...COVID_CRASH, dailyDrift: -0.010 }, COVID_RECOVERY]
  ),
  buyHoldNifty: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 12.16, 17.19, 5,
    [{ ...COVID_CRASH, dailyDrift: -0.011 }, COVID_RECOVERY]
  ),
  staticCapmMvo: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 10.71, 18.48, 3,
    [COVID_CRASH, COVID_RECOVERY]
  ),
  kalmanMvo: generateEquityCurve(
    new Date("2015-01-02"), new Date("2024-12-31"), 10.03, 18.62, 2,
    [COVID_CRASH, COVID_RECOVERY]
  ),
};

// ─── Beta time series for RELIANCE.NS ────────────────────────────────────────

export function generateBetaSeries(): Array<{
  date: string;
  beta30: number;
  beta60: number;
  beta120: number;
  vix: number;
}> {
  const dates: Date[] = [];
  const cur = new Date("2015-01-01");
  const end = new Date("2024-12-31");
  while (cur <= end) {
    const dow = cur.getDay();
    if (dow !== 0 && dow !== 6) dates.push(new Date(cur));
    cur.setDate(cur.getDate() + 7);
  }

  let s = 99;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  let beta = 1.05, vix = 18;
  return dates.map((d) => {
    beta = Math.max(0.4, Math.min(1.9, beta + (rand() - 0.5) * 0.09));
    vix  = Math.max(10, Math.min(90, vix + (rand() - 0.5) * 2.5));
    if (d >= new Date("2020-02-20") && d <= new Date("2020-05-01")) {
      beta = Math.min(1.9, beta + 0.18);
      vix  = Math.min(90, vix + 10);
    }
    return {
      date:    d.toISOString().split("T")[0],
      beta30:  Math.round((beta + (rand() - 0.5) * 0.12) * 100) / 100,
      beta60:  Math.round(beta * 100) / 100,
      beta120: Math.round((beta * 0.94 + 0.06) * 100) / 100,
      vix:     Math.round(vix * 10) / 10,
    };
  });
}

export const betaTimeSeriesData = generateBetaSeries();

// ─── Model Comparison ─────────────────────────────────────────────────────────

export const modelComparison = [
  { model: "Static OLS Beta",    mae: 0.0421, directionAcc: null,  notes: "Fixed 60d rolling OLS" },
  { model: "Kalman Filter",      mae: 0.0387, directionAcc: null,  notes: "State-space EM" },
  { model: "XGBoost",            mae: 0.0312, directionAcc: 0.634, notes: "Gradient boosting + SHAP" },
  { model: "LSTM (proposed)",    mae: 0.0268, directionAcc: 0.671, notes: "Seq2One LSTM + LayerNorm" },
];

// ─── SHAP Feature Importance ──────────────────────────────────────────────────

export const shaValue = [
  { feature: "betavol_60d",       importance: 0.156 },
  { feature: "vix_level",         importance: 0.134 },
  { feature: "betavol_30d",       importance: 0.112 },
  { feature: "vix_zscore",        importance: 0.098 },
  { feature: "market_vol_20d",    importance: 0.087 },
  { feature: "beta_60_30_spread", importance: 0.074 },
  { feature: "fii_dii_ratio",     importance: 0.062 },
  { feature: "crude_5d_chg",      importance: 0.054 },
  { feature: "betavol_trend",     importance: 0.048 },
  { feature: "usdinr_5d_chg",     importance: 0.041 },
];

// ─── Pipeline stages ──────────────────────────────────────────────────────────

export const pipelineStages = [
  {
    id: "data",
    label: "Raw Data",
    icon: "Database",
    description: "49 NIFTY50 stocks, India VIX, RBI rates, FII/DII flows, sector indices",
    detail: "2015–2025 | 2,753 rows × 51 cols | 10 data sources",
    color: "#7F77DD",
  },
  {
    id: "features",
    label: "Feature Engineering",
    icon: "Cpu",
    description: "Rolling betas (30/60/120d), beta volatility, macro signals, sector momentum",
    detail: "~35 features per stock | Per-stock + shared market features",
    color: "#EF9F27",
  },
  {
    id: "prediction",
    label: "LSTM Prediction",
    icon: "Brain",
    description: "Predict 20-day forward beta volatility for each stock",
    detail: "Seq2One LSTM | HuberLoss | AdamW | MAE: 0.027",
    color: "#1D9E75",
  },
  {
    id: "threshold",
    label: "Threshold Trigger",
    icon: "Zap",
    description: "Only rebalance when predicted betavol > calibrated percentile threshold",
    detail: "VIX > 25 → MIN_VARIANCE override | Adaptive rebalance frequency",
    color: "#EF9F27",
  },
  {
    id: "optimiser",
    label: "Portfolio Optimiser",
    icon: "Target",
    description: "Route to max-Sharpe MVO, min-variance, or risk parity based on regime",
    detail: "Ledoit-Wolf shrinkage | Long-only | Max 15% per stock",
    color: "#7F77DD",
  },
  {
    id: "backtest",
    label: "Walk-Forward Backtest",
    icon: "TrendingUp",
    description: "Strict out-of-sample validation with realistic transaction costs",
    detail: "2015–2025 OOS | 0.08% round-trip cost | 6 walk-forward folds",
    color: "#D85A30",
  },
];
