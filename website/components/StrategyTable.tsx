"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { strategyMetrics } from "@/lib/sampleData";
import { ChevronUp, ChevronDown } from "lucide-react";

type MetricKey = "annualReturn" | "annualVol" | "sharpe" | "sortino" | "maxDrawdown" | "calmar";

const columns: { key: MetricKey; label: string; higherIsBetter: boolean; suffix: string }[] = [
  { key: "annualReturn", label: "Annual Return", higherIsBetter: true, suffix: "%" },
  { key: "annualVol", label: "Annual Vol", higherIsBetter: false, suffix: "%" },
  { key: "sharpe", label: "Sharpe Ratio", higherIsBetter: true, suffix: "" },
  { key: "sortino", label: "Sortino Ratio", higherIsBetter: true, suffix: "" },
  { key: "maxDrawdown", label: "Max Drawdown", higherIsBetter: false, suffix: "%" },
  { key: "calmar", label: "Calmar Ratio", higherIsBetter: true, suffix: "" },
];

export default function StrategyTable() {
  const [sortKey, setSortKey] = useState<MetricKey>("sharpe");
  const [sortAsc, setSortAsc] = useState(false);

  const strategies = Object.entries(strategyMetrics);

  // Compute best/worst per column
  const columnStats = Object.fromEntries(
    columns.map((col) => {
      const values = strategies.map(([, s]) => s[col.key]);
      return [col.key, { min: Math.min(...values), max: Math.max(...values) }];
    })
  );

  const sorted = [...strategies].sort(([, a], [, b]) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    return sortAsc ? av - bv : bv - av;
  });

  const handleSort = (key: MetricKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-navy-700">
            <th className="text-left py-3 px-2 text-gray-400 font-medium w-48">Strategy</th>
            {columns.map((col) => (
              <th
                key={col.key}
                className="text-right py-3 px-2 text-gray-400 font-medium cursor-pointer hover:text-white select-none whitespace-nowrap"
                onClick={() => handleSort(col.key)}
              >
                <span className="inline-flex items-center gap-1 justify-end">
                  {col.label}
                  {sortKey === col.key ? (
                    sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronDown className="w-3 h-3 opacity-30" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(([key, s]) => (
            <tr
              key={key}
              className={cn(
                "border-b border-navy-800 transition-colors hover:bg-navy-800/50",
                key === "adaptiveBeta" && "bg-teal-400/5"
              )}
            >
              {/* Strategy name */}
              <td className="py-3 px-2">
                <div className="flex items-center gap-2.5">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: s.color }} />
                  <span className={cn("font-medium", key === "adaptiveBeta" ? "text-white" : "text-gray-300")}>
                    {s.label}
                  </span>
                  {key === "adaptiveBeta" && (
                    <span className="badge-teal text-xs hidden sm:inline-flex">★ best</span>
                  )}
                </div>
              </td>

              {/* Metric cells */}
              {columns.map((col) => {
                const val = s[col.key];
                const { min, max } = columnStats[col.key];
                const isBest = col.higherIsBetter ? val === max : val === min;
                const isWorst = col.higherIsBetter ? val === min : val === max;
                return (
                  <td
                    key={col.key}
                    className={cn(
                      "py-3 px-2 text-right font-mono tabular-nums",
                      isBest && "text-teal-300 font-semibold",
                      isWorst && "text-coral-300",
                      !isBest && !isWorst && "text-gray-300"
                    )}
                  >
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded",
                        isBest && "bg-teal-400/10",
                        isWorst && "bg-coral-400/10"
                      )}
                    >
                      {val > 0 && col.key !== "maxDrawdown" ? "" : ""}
                      {val.toFixed(col.suffix === "%" ? 1 : 2)}{col.suffix}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-600 mt-3">
        Click column headers to sort. Teal = best in column, coral = worst.
      </p>
    </div>
  );
}
