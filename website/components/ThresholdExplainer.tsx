"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
  ScatterChart,
  ComposedChart,
  Area,
} from "recharts";
import { betaTimeSeriesData } from "@/lib/sampleData";
import { downsample } from "@/lib/utils";
import { cn } from "@/lib/utils";

/** Generate synthetic portfolio betavol from the beta series. */
function generateBetaVolSeries() {
  const data = downsample(betaTimeSeriesData, 2);
  let runningAvg = 0.08;
  return data.map((d, i) => {
    const noise = (Math.sin(i * 0.3) * 0.02 + Math.cos(i * 0.15) * 0.015);
    runningAvg = Math.max(0.03, Math.min(0.35, runningAvg + noise * 0.1 + (d.vix / 1000)));
    const isStress = d.date >= "2020-02-01" && d.date <= "2020-05-15";
    const vol = isStress ? runningAvg + 0.15 : runningAvg;
    return {
      date: d.date,
      betavol: Math.round(Math.max(0.02, Math.min(0.40, vol)) * 1000) / 1000,
    };
  });
}

const BETAVOL_DATA = generateBetaVolSeries();
const ALL_VALUES = BETAVOL_DATA.map((d) => d.betavol);
const Q25 = ALL_VALUES.sort((a, b) => a - b)[Math.floor(ALL_VALUES.length * 0.25)];
const Q75 = ALL_VALUES.sort((a, b) => a - b)[Math.floor(ALL_VALUES.length * 0.75)];
const MIN_VAL = Math.min(...ALL_VALUES);
const MAX_VAL = Math.max(...ALL_VALUES);

export default function ThresholdExplainer() {
  const [threshold, setThreshold] = useState(Q75);

  const { rebalanceDates, holdCount, rebalanceCount } = useMemo(() => {
    const triggers = BETAVOL_DATA.filter((d) => d.betavol > threshold);
    return {
      rebalanceDates: triggers.map((d) => d.date),
      rebalanceCount: triggers.length,
      holdCount: BETAVOL_DATA.length - triggers.length,
    };
  }, [threshold]);

  const chartData = BETAVOL_DATA.map((d) => ({
    ...d,
    triggered: d.betavol > threshold ? d.betavol : null,
  }));

  const tPct = ((threshold - MIN_VAL) / (MAX_VAL - MIN_VAL)) * 100;

  return (
    <div className="card">
      {/* Counters */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-teal-400">{rebalanceCount}</div>
          <div className="text-xs text-gray-400">Rebalances Triggered</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-300">{holdCount}</div>
          <div className="text-xs text-gray-400">Hold Days</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-amber-400">
            {((rebalanceCount / BETAVOL_DATA.length) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Trigger Rate</div>
        </div>
      </div>

      {/* Chart */}
      <div className="mb-5" style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#6B7280", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              interval={Math.floor(chartData.length / 7)}
            />
            <YAxis
              tick={{ fill: "#6B7280", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={40}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{ background: "#0F1629", border: "1px solid #1A2550", borderRadius: 8, fontSize: 11 }}
              formatter={(v: number) => [v.toFixed(3), ""]}
            />

            {/* Area above threshold */}
            <Area
              type="monotone"
              dataKey="triggered"
              fill="rgba(239,159,39,0.15)"
              stroke="none"
            />

            {/* Betavol line */}
            <Line
              type="monotone"
              dataKey="betavol"
              stroke="#7F77DD"
              strokeWidth={1.5}
              dot={false}
              name="Portfolio BetaVol"
            />

            {/* Threshold reference line */}
            <ReferenceLine
              y={threshold}
              stroke="#EF9F27"
              strokeWidth={2}
              strokeDasharray="6 3"
              label={{ value: `Threshold: ${threshold.toFixed(3)}`, fill: "#EF9F27", fontSize: 10, position: "right" }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Slider */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-500 mb-2">
          <span>More rebalances (lower threshold)</span>
          <span>Fewer rebalances (higher threshold)</span>
        </div>
        <input
          type="range"
          min={MIN_VAL}
          max={MAX_VAL}
          step={0.001}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full accent-amber-400 cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>{MIN_VAL.toFixed(3)}</span>
          <div className="flex gap-3">
            <button
              onClick={() => setThreshold(Q25)}
              className={cn("text-xs px-2 py-0.5 rounded border transition-colors",
                Math.abs(threshold - Q25) < 0.001 ? "border-purple-400 text-purple-300" : "border-navy-700 text-gray-500"
              )}
            >
              Q25
            </button>
            <button
              onClick={() => setThreshold(Q75)}
              className={cn("text-xs px-2 py-0.5 rounded border transition-colors",
                Math.abs(threshold - Q75) < 0.001 ? "border-amber-400 text-amber-300" : "border-navy-700 text-gray-500"
              )}
            >
              Q75 (used)
            </button>
          </div>
          <span>{MAX_VAL.toFixed(3)}</span>
        </div>
      </div>

      <p className="text-xs text-gray-500 text-center">
        Threshold is calibrated on training data only (2015–2021). Applied to 2022–2024 OOS period.
        Orange shading = rebalance zone.
      </p>
    </div>
  );
}
