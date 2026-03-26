"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
} from "recharts";
import { equityCurveData, strategyMetrics } from "@/lib/sampleData";
import { downsample } from "@/lib/utils";
import { cn } from "@/lib/utils";

type StrategyKey = keyof typeof equityCurveData;

const STRATEGIES: { key: StrategyKey; label: string; color: string }[] = [
  { key: "adaptiveBeta", label: "AdaptiveBeta (ours)", color: "#1D9E75" },
  { key: "kalmanMvo", label: "Kalman-Beta MVO", color: "#EF9F27" },
  { key: "staticCapmMvo", label: "Static CAPM MVO", color: "#7F77DD" },
  { key: "equalWeight", label: "Equal Weight", color: "#888780" },
  { key: "buyHoldNifty", label: "Buy & Hold NIFTY50", color: "#D85A30" },
];

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-2.5 shadow-xl text-xs min-w-[160px]">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center justify-between gap-3 mb-1">
          <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: entry.color }} />
            <span className="text-gray-300 truncate max-w-[110px]">{entry.name}</span>
          </span>
          <span className="font-mono font-medium text-white">{entry.value?.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

export default function EquityCurveChart() {
  const [activeStrategies, setActiveStrategies] = useState<Set<StrategyKey>>(
    new Set(STRATEGIES.map((s) => s.key))
  );

  const chartData = useMemo(() => {
    // Merge all strategy data by date, downsample for performance
    const mergedMap = new Map<string, Record<string, number>>();
    for (const { key } of STRATEGIES) {
      for (const { date, value } of equityCurveData[key]) {
        const existing = mergedMap.get(date) ?? { date: date as unknown as number };
        mergedMap.set(date, { ...existing, [key]: Math.round(value * 10) / 10 });
      }
    }
    const all = Array.from(mergedMap.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );
    return downsample(all, 4);
  }, []);

  const toggleStrategy = (key: StrategyKey) => {
    setActiveStrategies((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        if (next.size > 1) next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div>
      {/* Strategy toggles */}
      <div className="flex flex-wrap gap-2 mb-5">
        {STRATEGIES.map(({ key, label, color }) => (
          <button
            key={key}
            onClick={() => toggleStrategy(key)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all",
              activeStrategies.has(key)
                ? "border-transparent"
                : "border-navy-700 text-gray-500 opacity-50"
            )}
            style={
              activeStrategies.has(key)
                ? { background: `${color}20`, borderColor: `${color}50`, color }
                : {}
            }
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: activeStrategies.has(key) ? color : "#4B5563" }}
            />
            {label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6B7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval={Math.floor(chartData.length / 7)}
          />
          <YAxis
            tick={{ fill: "#6B7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={45}
            tickFormatter={(v) => v.toFixed(0)}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* COVID Crash */}
          <ReferenceArea
            x1="2020-02-03"
            x2="2020-05-01"
            fill="rgba(216,90,48,0.08)"
            label={{ value: "COVID", fill: "#D85A30", fontSize: 10, position: "insideTop" }}
          />
          {/* ADANI Crisis */}
          <ReferenceArea
            x1="2023-01-23"
            x2="2023-03-13"
            fill="rgba(239,159,39,0.07)"
            label={{ value: "ADANI", fill: "#EF9F27", fontSize: 10, position: "insideTop" }}
          />

          <ReferenceLine y={100} stroke="rgba(255,255,255,0.12)" strokeDasharray="4 4" />

          {STRATEGIES.map(({ key, label, color }) =>
            activeStrategies.has(key) ? (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={label}
                stroke={color}
                strokeWidth={key === "adaptiveBeta" ? 2.5 : 1.5}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ) : null
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
