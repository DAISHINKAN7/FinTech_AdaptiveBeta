"use client";

import { useMemo } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { betaTimeSeriesData } from "@/lib/sampleData";
import { downsample } from "@/lib/utils";

type TooltipPayload = {
  name: string;
  value: number;
  color: string;
};

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-2.5 shadow-xl text-xs">
      <p className="text-gray-400 mb-1.5">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
            <span className="w-2 h-2 rounded-full" style={{ background: entry.color }} />
            {entry.name}
          </span>
          <span className="font-mono font-medium text-white">{entry.value?.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function HeroChart() {
  // Downsample to keep chart performant
  const chartData = useMemo(() => downsample(betaTimeSeriesData, 3), []);

  return (
    <div className="w-full" style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6B7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval={Math.floor(chartData.length / 8)}
          />
          <YAxis
            yAxisId="beta"
            domain={[0.3, 2.0]}
            tick={{ fill: "#6B7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={35}
          />
          <YAxis
            yAxisId="vix"
            orientation="right"
            domain={[8, 100]}
            tick={{ fill: "#6B7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#9CA3AF", paddingTop: 8 }}
          />

          {/* VIX as subtle area */}
          <Area
            yAxisId="vix"
            type="monotone"
            dataKey="vix"
            name="India VIX"
            stroke="#D85A30"
            fill="rgba(216,90,48,0.06)"
            strokeWidth={1}
            dot={false}
            activeDot={false}
          />

          {/* Beta = 1.0 reference */}
          <ReferenceLine yAxisId="beta" y={1.0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />

          {/* Beta lines */}
          <Line
            yAxisId="beta"
            type="monotone"
            dataKey="beta120"
            name="Beta 120d"
            stroke="#EF9F27"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            yAxisId="beta"
            type="monotone"
            dataKey="beta60"
            name="Beta 60d"
            stroke="#7F77DD"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            yAxisId="beta"
            type="monotone"
            dataKey="beta30"
            name="Beta 30d"
            stroke="#1D9E75"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
