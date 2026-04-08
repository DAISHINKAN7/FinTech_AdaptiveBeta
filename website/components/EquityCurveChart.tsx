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
  ReferenceArea,
  ReferenceLine,
} from "recharts";
import { equityCurveData, strategyMetrics } from "@/lib/sampleData";
import { downsample } from "@/lib/utils";
import { cn } from "@/lib/utils";

type StrategyKey = keyof typeof equityCurveData;

const STRATEGIES: { key: StrategyKey; label: string; color: string; dashed?: boolean }[] = [
  { key: "adaptiveBeta",    label: "AdaptiveBeta (ours)",  color: "#1D9E75" },
  { key: "momentumQuality", label: "Momentum-Quality",     color: "#7F77DD" },
  { key: "equalWeight",     label: "Equal Weight",         color: "#5DCAA5" },
  { key: "buyHoldNifty",    label: "Buy & Hold NIFTY50",   color: "#EF9F27", dashed: true },
  { key: "staticCapmMvo",   label: "Static CAPM MVO",      color: "#D85A30", dashed: true },
  { key: "kalmanMvo",       label: "Kalman-Beta MVO",      color: "#888780", dashed: true },
];

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-navy-800/95 border border-navy-600 rounded-lg px-3 py-2.5 shadow-xl text-xs min-w-[175px] backdrop-blur-sm">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      {[...payload]
        .sort((a, b) => b.value - a.value)
        .map((entry) => (
          <div key={entry.name} className="flex items-center justify-between gap-3 mb-1">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: entry.color }} />
              <span className="text-gray-300 truncate max-w-[115px]">{entry.name}</span>
            </span>
            <span className="font-mono font-semibold text-white">{entry.value?.toFixed(1)}</span>
          </div>
        ))}
    </div>
  );
}

export default function EquityCurveChart() {
  const [active, setActive] = useState<Set<StrategyKey>>(
    new Set(STRATEGIES.map((s) => s.key))
  );

  const chartData = useMemo(() => {
    const map = new Map<string, Record<string, number>>();
    for (const { key } of STRATEGIES) {
      for (const { date, value } of equityCurveData[key]) {
        const ex = map.get(date) ?? { date: date as unknown as number };
        map.set(date, { ...ex, [key]: Math.round(value * 10) / 10 });
      }
    }
    const all = Array.from(map.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );
    return downsample(all, 4);
  }, []);

  const toggle = (key: StrategyKey) => {
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(key) && next.size > 1) next.delete(key);
      else next.add(key);
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
            onClick={() => toggle(key)}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium border transition-all",
              active.has(key)
                ? "border-transparent"
                : "border-navy-700 text-gray-500 opacity-40"
            )}
            style={
              active.has(key)
                ? { background: `${color}18`, borderColor: `${color}45`, color }
                : {}
            }
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: active.has(key) ? color : "#4B5563" }}
            />
            {label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
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
            width={46}
            tickFormatter={(v) => v.toFixed(0)}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Crisis shading */}
          <ReferenceArea
            x1="2020-02-03" x2="2020-05-01"
            fill="rgba(216,90,48,0.07)"
            label={{ value: "COVID", fill: "#D85A30", fontSize: 10, position: "insideTop" }}
          />
          <ReferenceArea
            x1="2023-01-23" x2="2023-03-13"
            fill="rgba(239,159,39,0.06)"
            label={{ value: "ADANI", fill: "#EF9F27", fontSize: 10, position: "insideTop" }}
          />
          <ReferenceArea
            x1="2018-08-01" x2="2018-11-30"
            fill="rgba(127,119,221,0.05)"
            label={{ value: "IL&FS", fill: "#7F77DD", fontSize: 10, position: "insideTop" }}
          />

          <ReferenceLine y={100} stroke="rgba(255,255,255,0.10)" strokeDasharray="4 4" />

          {STRATEGIES.map(({ key, label, color, dashed }) =>
            active.has(key) ? (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={label}
                stroke={color}
                strokeWidth={key === "adaptiveBeta" || key === "momentumQuality" ? 2.5 : 1.5}
                strokeDasharray={dashed ? "5 3" : undefined}
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
