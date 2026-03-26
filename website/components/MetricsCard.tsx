"use client";

import { useEffect, useRef, useState } from "react";
import { useInView } from "framer-motion";
import { cn } from "@/lib/utils";

interface MetricsCardProps {
  label: string;
  value: number;
  metricName?: string;
  subValue?: string;
  delta?: number;
  deltaLabel?: string;
  color?: string;
  highlight?: boolean;
  decimals?: number;
}

function useCountUp(target: number, duration: number, active: boolean): number {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!active) return;
    const startTime = performance.now();
    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(target * eased);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, duration, active]);

  return count;
}

export default function MetricsCard({
  label,
  value,
  metricName = "Sharpe Ratio",
  subValue,
  delta,
  deltaLabel,
  color = "#1D9E75",
  highlight = false,
  decimals = 2,
}: MetricsCardProps) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  const displayValue = useCountUp(value, 1400, inView);

  return (
    <div
      ref={ref}
      className={cn(
        "relative rounded-xl p-5 border transition-all duration-200",
        highlight
          ? "bg-gradient-to-br from-navy-800 to-navy-700 border-teal-400/30 shadow-lg shadow-teal-400/10"
          : "bg-navy-800 border-navy-700 hover:border-navy-600"
      )}
    >
      {highlight && (
        <div className="absolute top-3 right-3">
          <span className="badge-teal text-xs">Best</span>
        </div>
      )}

      {/* Color accent bar */}
      <div
        className="w-8 h-0.5 rounded-full mb-4"
        style={{ background: color }}
      />

      <p className="text-xs text-gray-400 mb-1">{label}</p>

      <div className="flex items-baseline gap-1 mb-1">
        <span className="text-2xl font-bold text-white tabular-nums">
          {displayValue.toFixed(decimals)}
        </span>
        {delta !== undefined && (
          <span
            className={cn(
              "text-xs font-medium ml-1",
              delta >= 0 ? "text-teal-400" : "text-coral-400"
            )}
          >
            {delta >= 0 ? "+" : ""}{delta.toFixed(2)}
          </span>
        )}
      </div>

      {deltaLabel && delta !== undefined && (
        <p className="text-xs text-gray-500 mb-2">{deltaLabel}</p>
      )}

      <p className="text-xs text-gray-400 font-medium">{metricName}</p>

      {subValue && (
        <p className="text-xs text-gray-500 mt-1">{subValue}</p>
      )}
    </div>
  );
}
