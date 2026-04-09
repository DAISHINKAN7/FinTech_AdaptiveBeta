/**
 * Browser-side portfolio simulation engine for the AdaptiveBeta professor demo.
 *
 * Generates 2520 days of synthetic-but-realistic market data (2015-2024) with
 * correctly-embedded crisis periods (COVID, ADANI, IL&FS, Rate Hike), then
 * re-runs the AdaptiveBeta algorithm in the browser when the user adjusts
 * parameters.  All data is deterministic (seeded PRNG).
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export type Signal = "hold" | "rebalance" | "min_variance";

export interface DayData {
  date: string;
  marketReturn: number;       // daily log return (NIFTY50-like)
  vixLevel: number;           // India VIX
  predictedBetavol: number;   // synthetic LSTM prediction
  alphaSignal: number;        // cross-sectional alpha available from optimisation
  regime: "bull" | "bear" | "transition";
}

export interface SimParams {
  vixThreshold: number;       // VIX override level — default 25
  betavolPercentile: number;  // 50–95, default 55
  regimeRouting: boolean;     // HMM regime-aware optimizer routing
  txCostBps: number;          // round-trip tx cost in bps — default 8
}

export interface SimPoint {
  date: string;
  portfolio: number;   // indexed to 100 at start
  nifty: number;
  signal: Signal;
  vix: number;
  betavol: number;
  threshold: number;
  beta: number;
}

export interface SimMetrics {
  cagr: number;
  sharpe: number;
  maxDrawdown: number;
  sortino: number;
  calmar: number;
  annualVol: number;
  totalReturn: number;
}

export interface CrisisStats {
  portfolioReturn: number;
  niftyReturn: number;
  portfolioMaxDD: number;
  niftyMaxDD: number;
  rebalances: number;
  minVarDays: number;
  protection: number; // how much better (negative = worse) vs nifty
}

export interface SimResult {
  curve: SimPoint[];
  fullResults: SimPoint[];
  metrics: SimMetrics;
  niftyMetrics: SimMetrics;
  rebalanceCount: number;
  minVarDays: number;
  holdDays: number;
  betavolThreshold: number;
  crisisStats: Record<string, CrisisStats>;
}

// ─── Seeded PRNG (Mulberry32) ─────────────────────────────────────────────────

function mkRand(seed: number) {
  let s = seed >>> 0;
  return (): number => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 0xffffffff;
  };
}

function normalRand(rand: () => number): number {
  // Box-Muller
  const u1 = Math.max(1e-10, rand());
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Trading Calendar ─────────────────────────────────────────────────────────

function tradingDays(start: string, end: string): string[] {
  const days: string[] = [];
  const cur = new Date(start);
  const endD = new Date(end);
  while (cur <= endD) {
    const dow = cur.getDay();
    if (dow !== 0 && dow !== 6) days.push(cur.toISOString().split("T")[0]);
    cur.setDate(cur.getDate() + 1);
  }
  return days;
}

// ─── Crisis Definitions ───────────────────────────────────────────────────────

export const CRISIS_DEFINITIONS: Record<string, { start: string; end: string; label: string; color: string }> = {
  "COVID Crash":   { start: "2020-02-20", end: "2020-05-01",  label: "COVID Crash\nFeb–May 2020",   color: "#D85A30" },
  "ADANI Crisis":  { start: "2023-01-25", end: "2023-03-15",  label: "ADANI Crisis\nJan–Mar 2023",  color: "#EF9F27" },
  "IL&FS Crisis":  { start: "2018-08-01", end: "2018-10-31",  label: "IL&FS Crisis\nAug–Oct 2018",  color: "#7F77DD" },
  "2018 Rate Hike":{ start: "2018-09-01", end: "2018-11-30",  label: "Rate Hike\nSep–Nov 2018",     color: "#5DCAA5" },
};

// ─── Synthetic Market Data Generation ────────────────────────────────────────

function generateMarketData(): DayData[] {
  const rand  = mkRand(42);
  const rn    = () => normalRand(rand);

  const dates = tradingDays("2015-01-02", "2024-12-31");

  // NIFTY50-like: CAGR ~12%, ann vol ~17%
  const dailyMu    = 0.12 / 252;
  const dailySigma = 0.17 / Math.sqrt(252);

  let vix     = 16.0;
  let betavol = 0.08;

  return dates.map((date) => {
    // ── Crisis detection ──
    const isCOVIDPeak = date >= "2020-03-05"  && date <= "2020-03-23";
    const isCOVID     = date >= "2020-02-20"  && date <= "2020-05-01";
    const isADANI     = date >= "2023-01-25"  && date <= "2023-03-15";
    const isILFS      = date >= "2018-08-01"  && date <= "2018-10-31";
    const isRateHike  = date >= "2018-09-01"  && date <= "2018-11-30";

    // ── VIX mean-reversion with crisis spikes ──
    const vixTargets: [boolean, number][] = [
      [isCOVIDPeak, 78],
      [isCOVID,     48],
      [isADANI,     27],
      [isILFS,      23],
      [isRateHike,  21],
    ];
    const vixTarget = vixTargets.find(([cond]) => cond)?.[1] ?? 16;
    vix = vix * 0.91 + vixTarget * 0.09 + (rand() - 0.49) * 1.4;
    vix = Math.max(10, Math.min(90, vix));

    // ── Market return ──
    let ret = dailyMu + dailySigma * rn();
    if (isCOVIDPeak) ret += -0.022 + dailySigma * 1.8 * rn();
    else if (isCOVID)    ret += -0.010;
    else if (isADANI)    ret += -0.004;
    else if (isILFS || isRateHike) ret += -0.003;

    // ── BetaVol correlated with VIX ──
    const vixNorm = (vix - 10) / 80;
    betavol = 0.035 + vixNorm * 0.38 + (rand() - 0.5) * 0.018;
    betavol = Math.max(0.022, Math.min(0.44, betavol));

    // ── Regime ──
    const regime: "bull" | "bear" | "transition" =
      vix > 25 ? "bear" : vix > 18 ? "transition" : "bull";

    // ── Alpha ──
    const alphaBase = regime === "bull" ? 3e-4 : regime === "bear" ? -1e-4 : 1e-4;
    const alphaSignal = alphaBase + (rand() - 0.5) * 2e-4;

    return {
      date,
      marketReturn:      ret,
      vixLevel:          Math.round(vix * 10) / 10,
      predictedBetavol:  Math.round(betavol * 1000) / 1000,
      alphaSignal,
      regime,
    };
  });
}

/** Pre-generated once at module level — deterministic, ~2520 entries */
export const MARKET_DATA: DayData[] = generateMarketData();

// ─── Percentile ───────────────────────────────────────────────────────────────

export function computePercentile(values: number[], pct: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(
    Math.floor((pct / 100) * sorted.length),
    sorted.length - 1
  );
  return sorted[idx];
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

function computeMetrics(results: SimPoint[], field: "portfolio" | "nifty"): SimMetrics {
  if (results.length < 10) {
    return { cagr: 0, sharpe: 0, maxDrawdown: 0, sortino: 0, calmar: 0, annualVol: 0, totalReturn: 0 };
  }

  const vals   = results.map((r) => r[field]);
  const n      = vals.length;
  const years  = n / 252;
  const start  = vals[0];
  const end    = vals[n - 1];

  const dailyRets: number[] = [];
  for (let i = 1; i < n; i++) dailyRets.push((vals[i] - vals[i - 1]) / vals[i - 1]);

  const meanRet  = dailyRets.reduce((a, b) => a + b, 0) / dailyRets.length;
  const variance = dailyRets.reduce((a, b) => a + (b - meanRet) ** 2, 0) / dailyRets.length;
  const annVol   = Math.sqrt(variance * 252) * 100;

  const totalReturn = (end / start - 1) * 100;
  const cagr        = (Math.pow(end / start, 1 / years) - 1) * 100;
  const rfRate      = 6.5; // India risk-free ~6.5%
  const sharpe      = annVol > 0 ? (cagr - rfRate) / annVol : 0;

  const downside    = dailyRets.filter((r) => r < 0);
  const downVar     = downside.length > 0 ? downside.reduce((a, b) => a + b ** 2, 0) / downside.length : 0;
  const downVol     = Math.sqrt(downVar * 252) * 100;
  const sortino     = downVol > 0 ? (cagr - rfRate) / downVol : 0;

  let peak = vals[0], maxDD = 0;
  for (const v of vals) {
    if (v > peak) peak = v;
    const dd = (v - peak) / peak * 100;
    if (dd < maxDD) maxDD = dd;
  }
  const calmar = maxDD !== 0 ? cagr / Math.abs(maxDD) : 0;

  return {
    cagr:        Math.round(cagr * 100) / 100,
    sharpe:      Math.round(sharpe * 1000) / 1000,
    sortino:     Math.round(sortino * 1000) / 1000,
    maxDrawdown: Math.round(maxDD * 100) / 100,
    calmar:      Math.round(calmar * 1000) / 1000,
    annualVol:   Math.round(annVol * 100) / 100,
    totalReturn: Math.round(totalReturn * 100) / 100,
  };
}

// ─── Crisis stats ─────────────────────────────────────────────────────────────

function crisisStats(full: SimPoint[], name: string): CrisisStats {
  const def = CRISIS_DEFINITIONS[name];
  if (!def) return { portfolioReturn: 0, niftyReturn: 0, portfolioMaxDD: 0, niftyMaxDD: 0, rebalances: 0, minVarDays: 0, protection: 0 };

  const window = full.filter((r) => r.date >= def.start && r.date <= def.end);
  if (!window.length) return { portfolioReturn: 0, niftyReturn: 0, portfolioMaxDD: 0, niftyMaxDD: 0, rebalances: 0, minVarDays: 0, protection: 0 };

  const pS = window[0].portfolio, pE = window[window.length - 1].portfolio;
  const nS = window[0].nifty,    nE = window[window.length - 1].nifty;
  const pRet = (pE / pS - 1) * 100;
  const nRet = (nE / nS - 1) * 100;

  let pPeak = pS, pDD = 0, nPeak = nS, nDD = 0;
  let rebalances = 0, minVarDays = 0;
  for (const r of window) {
    if (r.portfolio > pPeak) pPeak = r.portfolio;
    if (r.nifty > nPeak)     nPeak = r.nifty;
    const pD = (r.portfolio - pPeak) / pPeak * 100;
    const nD = (r.nifty     - nPeak) / nPeak * 100;
    if (pD < pDD) pDD = pD;
    if (nD < nDD) nDD = nD;
    if (r.signal === "rebalance")    rebalances++;
    if (r.signal === "min_variance") minVarDays++;
  }

  return {
    portfolioReturn: Math.round(pRet * 100) / 100,
    niftyReturn:     Math.round(nRet * 100) / 100,
    portfolioMaxDD:  Math.round(pDD * 100) / 100,
    niftyMaxDD:      Math.round(nDD * 100) / 100,
    rebalances,
    minVarDays,
    protection:      Math.round((pRet - nRet) * 100) / 100,
  };
}

// ─── Main simulation ──────────────────────────────────────────────────────────

export function runSimulation(params: SimParams): SimResult {
  const { vixThreshold, betavolPercentile, regimeRouting, txCostBps } = params;
  const data = MARKET_DATA;

  // Calibrate betavol threshold on first 60% of data (training window)
  const trainN   = Math.floor(data.length * 0.6);
  const trainVols = data.slice(0, trainN).map((d) => d.predictedBetavol);
  const betavolThreshold = computePercentile(trainVols, betavolPercentile);

  let portfolioValue = 100;
  let niftyValue     = 100;
  let currentBeta    = 0.85;
  let lastRebalIdx   = -10;

  const fullResults: SimPoint[] = [];

  for (let i = 0; i < data.length; i++) {
    const day = data[i];

    // ── Signal ──
    let signal: Signal = "hold";
    if (day.vixLevel > vixThreshold) {
      signal = "min_variance";
    } else if (day.predictedBetavol > betavolThreshold && i - lastRebalIdx >= 5) {
      signal = "rebalance";
    }

    // ── Beta update ──
    if (signal === "min_variance") {
      // Reduce exposure gradually when VIX spikes
      currentBeta = Math.max(0.26, currentBeta * 0.87 - 0.015);
    } else if (signal === "rebalance") {
      if (regimeRouting) {
        currentBeta = day.regime === "bull" ? 0.90 : day.regime === "bear" ? 0.65 : 0.80;
      } else {
        currentBeta = 0.82;
      }
      lastRebalIdx = i;
    } else {
      // Slow drift back toward 0.82 when holding
      if (currentBeta < 0.82) currentBeta = Math.min(0.82, currentBeta + 0.004);
    }

    // ── Cost ──
    const cost = signal !== "hold" ? (txCostBps / 10000) * 0.45 : 0;

    // ── Alpha ──
    const alpha =
      signal === "rebalance"    ? day.alphaSignal * 3.2  :
      signal === "min_variance" ? day.alphaSignal * 0.3  :
                                  day.alphaSignal * 1.1;

    // ── Portfolio return ──
    const portfolioReturn = currentBeta * day.marketReturn + alpha - cost;
    portfolioValue *= 1 + portfolioReturn;
    niftyValue     *= 1 + day.marketReturn;

    fullResults.push({
      date:      day.date,
      portfolio: Math.round(portfolioValue * 100) / 100,
      nifty:     Math.round(niftyValue * 100) / 100,
      signal,
      vix:       day.vixLevel,
      betavol:   day.predictedBetavol,
      threshold: betavolThreshold,
      beta:      Math.round(currentBeta * 1000) / 1000,
    });
  }

  const metrics      = computeMetrics(fullResults, "portfolio");
  const niftyMetrics = computeMetrics(fullResults, "nifty");

  const cStats: Record<string, CrisisStats> = {};
  for (const name of Object.keys(CRISIS_DEFINITIONS)) {
    cStats[name] = crisisStats(fullResults, name);
  }

  // Downsample to ~620 points for chart rendering
  const step  = Math.max(1, Math.floor(fullResults.length / 620));
  const curve = fullResults.filter((_, i) => i % step === 0 || i === fullResults.length - 1);

  const rebalanceCount = fullResults.filter((r) => r.signal === "rebalance").length;
  const minVarDays     = fullResults.filter((r) => r.signal === "min_variance").length;

  return {
    curve,
    fullResults,
    metrics,
    niftyMetrics,
    rebalanceCount,
    minVarDays,
    holdDays: fullResults.length - rebalanceCount - minVarDays,
    betavolThreshold: Math.round(betavolThreshold * 1000) / 1000,
    crisisStats: cStats,
  };
}

/** Extract full-resolution data for a specific crisis window (+30 days context) */
export function getCrisisWindow(fullResults: SimPoint[], crisisName: string): SimPoint[] {
  const def = CRISIS_DEFINITIONS[crisisName];
  if (!def || !fullResults.length) return [];

  const preStart = new Date(def.start);
  preStart.setDate(preStart.getDate() - 30);
  const preStartStr = preStart.toISOString().split("T")[0];

  // Re-index both curves to 100 at the crisis window start (first data point)
  const raw = fullResults.filter((r) => r.date >= preStartStr && r.date <= def.end);
  if (!raw.length) return raw;

  const pBase = raw[0].portfolio;
  const nBase = raw[0].nifty;
  return raw.map((r) => ({
    ...r,
    portfolio: Math.round((r.portfolio / pBase) * 100 * 100) / 100,
    nifty:     Math.round((r.nifty     / nBase) * 100 * 100) / 100,
  }));
}

// ─── Real Demo Data Types ─────────────────────────────────────────────────────

/**
 * Shape of website/public/demo-data.json generated by scripts/05_export_demo_data.py.
 * When this file exists the browser demo switches from synthetic to real backtest data.
 */
export interface RealDemoData {
  source: "real";
  meta: {
    generatedAt: string;
    tradingDays: number;
    startDate: string;
    endDate: string;
    vixThreshold: number;
    betavolQuantile: number;
    betavolThreshold: number;
    strategies: string[];
  };
  dates: string[];
  vix: number[];
  betavol: number[];
  regime: string[];
  signals: string[];
  returns: Record<string, number[]>;
  equityCurves: Record<string, number[]>;
  metrics: Record<string, {
    cagr: number;
    sharpe: number;
    sortino: number;
    maxDrawdown: number;
    calmar: number;
    annualVol: number;
    totalReturn: number;
    tradingDays: number;
  }>;
  betavolThresholds: Record<string, number>;
  crisisStats: Record<string, {
    portfolioReturn: number;
    niftyReturn: number;
    portfolioMaxDD: number;
    niftyMaxDD: number;
    protection: number;
    rebalances: number;
    minVarDays: number;
  }>;
}

// ─── Real Data Simulation ─────────────────────────────────────────────────────

/**
 * When demo-data.json exists (run scripts/05_export_demo_data.py to generate it),
 * this replaces runSimulation() with a version that uses:
 *   - Real India VIX levels  → drives min_variance signal
 *   - Real portfolio betavol → drives rebalance signal
 *   - Real backtest equity curves (adaptive_beta + buy_hold_nifty)
 *
 * Adjusting sliders re-computes which days WOULD HAVE fired each signal at the
 * chosen thresholds, while the equity curve stays anchored to the actual backtest
 * returns. This lets users explore parameter sensitivity on real market data.
 */
export function runRealSimulation(data: RealDemoData, params: SimParams): SimResult {
  const { vixThreshold, betavolPercentile } = params;
  const n = data.dates.length;
  if (n === 0) return runSimulation(params);

  // Compute betavol threshold from real distribution at the chosen percentile
  const cleanBv = data.betavol.filter((v) => v > 0 && isFinite(v));
  const betavolThreshold = computePercentile(cleanBv.length > 0 ? cleanBv : data.betavol, betavolPercentile);

  // Real equity curves from backtest (already indexed to 100)
  const adaptiveCurve = data.equityCurves["adaptive_beta"]  ?? [];
  const niftyCurve    = data.equityCurves["buy_hold_nifty"] ?? [];

  const fullResults: SimPoint[] = [];
  let lastRebalIdx = -10;

  for (let i = 0; i < n; i++) {
    const vix    = data.vix[i]     ?? 18;
    const bv     = data.betavol[i] ?? 0.12;
    const rawReg = data.regime[i]  ?? "transition";
    const regime: "bull" | "bear" | "transition" =
      rawReg === "bull" || rawReg === "bear" ? rawReg : "transition";

    // Re-derive signal with current slider thresholds
    let signal: Signal = "hold";
    if (vix > vixThreshold) {
      signal = "min_variance";
    } else if (bv > betavolThreshold && i - lastRebalIdx >= 5) {
      signal = "rebalance";
      lastRebalIdx = i;
    }

    fullResults.push({
      date:      data.dates[i],
      portfolio: adaptiveCurve[i] ?? 100,
      nifty:     niftyCurve[i]    ?? 100,
      signal,
      vix,
      betavol:   bv,
      threshold: betavolThreshold,
      beta:      0.82,
    });
  }

  const metrics      = computeMetrics(fullResults, "portfolio");
  const niftyMetrics = computeMetrics(fullResults, "nifty");

  const cStats: Record<string, CrisisStats> = {};
  for (const name of Object.keys(CRISIS_DEFINITIONS)) {
    cStats[name] = crisisStats(fullResults, name);
  }

  const step  = Math.max(1, Math.floor(fullResults.length / 620));
  const curve = fullResults.filter((_, i) => i % step === 0 || i === fullResults.length - 1);

  const rebalanceCount = fullResults.filter((r) => r.signal === "rebalance").length;
  const minVarDays     = fullResults.filter((r) => r.signal === "min_variance").length;

  return {
    curve,
    fullResults,
    metrics,
    niftyMetrics,
    rebalanceCount,
    minVarDays,
    holdDays: n - rebalanceCount - minVarDays,
    betavolThreshold: Math.round(betavolThreshold * 1000) / 1000,
    crisisStats: cStats,
  };
}
