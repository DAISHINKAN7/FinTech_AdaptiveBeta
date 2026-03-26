/**
 * Sample data for the AdaptiveBeta portfolio website.
 * These are realistic values based on the M.Tech thesis results.
 * Replace with actual values from results/performance_metrics.csv after running the pipeline.
 */

export const strategyMetrics = {
  adaptiveBeta: {
    annualReturn: 18.4,
    annualVol: 13.8,
    sharpe: 1.42,
    sortino: 2.1,
    maxDrawdown: -18.2,
    calmar: 1.01,
    label: "AdaptiveBeta (ours)",
    color: "#1D9E75",
  },
  kalmanMvo: {
    annualReturn: 15.8,
    annualVol: 15.0,
    sharpe: 1.19,
    sortino: 1.6,
    maxDrawdown: -22.1,
    calmar: 0.71,
    label: "Kalman-Beta MVO",
    color: "#EF9F27",
  },
  staticCapmMvo: {
    annualReturn: 14.1,
    annualVol: 15.5,
    sharpe: 1.08,
    sortino: 1.4,
    maxDrawdown: -24.6,
    calmar: 0.57,
    label: "Static CAPM MVO",
    color: "#7F77DD",
  },
  equalWeight: {
    annualReturn: 13.2,
    annualVol: 16.5,
    sharpe: 0.94,
    sortino: 1.1,
    maxDrawdown: -31.4,
    calmar: 0.42,
    label: "Equal Weight",
    color: "#888780",
  },
  buyHoldNifty: {
    annualReturn: 12.7,
    annualVol: 16.8,
    sharpe: 0.89,
    sortino: 0.9,
    maxDrawdown: -38.7,
    calmar: 0.33,
    label: "Buy & Hold NIFTY50",
    color: "#D85A30",
  },
};

export type StrategyKey = keyof typeof strategyMetrics;

/** Generate a smooth equity curve for a strategy starting at 100. */
function generateEquityCurve(
  startDate: Date,
  endDate: Date,
  annualReturn: number,
  annualVol: number,
  seed: number
): Array<{ date: string; value: number }> {
  const dates: Date[] = [];
  const cur = new Date(startDate);
  while (cur <= endDate) {
    const day = cur.getDay();
    if (day !== 0 && day !== 6) dates.push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
  }

  // Simple seeded pseudo-random
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  const dailyMu = annualReturn / 100 / 252;
  const dailySigma = annualVol / 100 / Math.sqrt(252);

  let value = 100;
  return dates.map((d) => {
    const r = dailyMu + dailySigma * (rand() * 2 - 1) * Math.sqrt(3);
    // COVID crash injection
    if (d >= new Date("2020-02-20") && d <= new Date("2020-03-23")) {
      const covidDrop = seed === 1 ? -0.018 : -0.025;
      value = value * Math.exp(covidDrop + dailySigma * (rand() * 2 - 1));
    } else if (d >= new Date("2023-01-25") && d <= new Date("2023-03-15")) {
      const adaniDrop = seed === 1 ? -0.005 : -0.008;
      value = value * Math.exp(adaniDrop + r);
    } else {
      value = value * Math.exp(r);
    }
    return { date: d.toISOString().split("T")[0], value: Math.round(value * 100) / 100 };
  });
}

export const equityCurveData = {
  adaptiveBeta: generateEquityCurve(new Date("2018-01-02"), new Date("2024-12-31"), 18.4, 13.8, 1),
  kalmanMvo: generateEquityCurve(new Date("2018-01-02"), new Date("2024-12-31"), 15.8, 15.0, 2),
  staticCapmMvo: generateEquityCurve(new Date("2018-01-02"), new Date("2024-12-31"), 14.1, 15.5, 3),
  equalWeight: generateEquityCurve(new Date("2018-01-02"), new Date("2024-12-31"), 13.2, 16.5, 4),
  buyHoldNifty: generateEquityCurve(new Date("2018-01-02"), new Date("2024-12-31"), 12.7, 16.8, 5),
};

/** Beta time series for RELIANCE.NS, 2015–2024 */
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
    const day = cur.getDay();
    if (day !== 0 && day !== 6) dates.push(new Date(cur));
    cur.setDate(cur.getDate() + 7); // weekly for readability
  }

  let s = 99;
  const rand = () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };

  let beta = 1.05;
  let vix = 18;
  return dates.map((d) => {
    beta = Math.max(0.4, Math.min(1.8, beta + (rand() - 0.5) * 0.08));
    vix = Math.max(10, Math.min(90, vix + (rand() - 0.5) * 3));
    if (d >= new Date("2020-02-20") && d <= new Date("2020-05-01")) {
      beta = Math.min(1.8, beta + 0.15);
      vix = Math.min(90, vix + 8);
    }
    return {
      date: d.toISOString().split("T")[0],
      beta30: Math.round((beta + (rand() - 0.5) * 0.1) * 100) / 100,
      beta60: Math.round(beta * 100) / 100,
      beta120: Math.round((beta * 0.95 + 0.05) * 100) / 100,
      vix: Math.round(vix * 10) / 10,
    };
  });
}

export const betaTimeSeriesData = generateBetaSeries();

export const modelComparison = [
  { model: "Static OLS Beta", mae: 0.0421, directionAcc: null, notes: "Fixed 60d rolling OLS" },
  { model: "Kalman Filter", mae: 0.0387, directionAcc: null, notes: "State-space EM" },
  { model: "XGBoost", mae: 0.0312, directionAcc: 0.634, notes: "Gradient boosting + SHAP" },
  { model: "LSTM (proposed)", mae: 0.0268, directionAcc: 0.671, notes: "Seq2One LSTM + LayerNorm" },
];

export const shaValue = [
  { feature: "betavol_60", importance: 0.156 },
  { feature: "vix_level", importance: 0.134 },
  { feature: "betavol_30", importance: 0.112 },
  { feature: "vix_zscore", importance: 0.098 },
  { feature: "market_vol_20d", importance: 0.087 },
  { feature: "beta_60_30_spread", importance: 0.074 },
  { feature: "fii_dii_ratio", importance: 0.062 },
  { feature: "crude_5d_chg", importance: 0.054 },
  { feature: "betavol_trend", importance: 0.048 },
  { feature: "usdinr_5d_chg", importance: 0.041 },
];

export const stressEvents = {
  "COVID Crash (Feb–May 2020)": {
    adaptiveBeta: { return: -12.4, maxDD: -18.2 },
    kalmanMvo: { return: -18.7, maxDD: -24.1 },
    staticCapmMvo: { return: -21.3, maxDD: -27.8 },
    equalWeight: { return: -28.4, maxDD: -35.2 },
    buyHoldNifty: { return: -32.1, maxDD: -38.7 },
  },
  "ADANI Crisis (Jan–Mar 2023)": {
    adaptiveBeta: { return: -3.2, maxDD: -5.1 },
    kalmanMvo: { return: -5.8, maxDD: -8.4 },
    staticCapmMvo: { return: -6.1, maxDD: -9.2 },
    equalWeight: { return: -7.4, maxDD: -11.3 },
    buyHoldNifty: { return: -5.1, maxDD: -7.8 },
  },
};

export const pipelineStages = [
  {
    id: "data",
    label: "Raw Data",
    icon: "Database",
    description: "49 NIFTY50 stocks, India VIX, RBI rates, FII/DII flows, sector indices",
    detail: "2015–2024 | 2,753 rows × 51 cols | 10 data sources",
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
    description: "Only rebalance when predicted betavol > 75th percentile threshold",
    detail: "VIX > 25 → MIN_VARIANCE override | ~40% fewer trades",
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
    detail: "2018–2024 OOS | 0.08% round-trip cost | 6 test years",
    color: "#D85A30",
  },
];
