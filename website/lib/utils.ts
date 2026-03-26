import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a number as a percentage string. */
export function formatPct(value: number, decimals = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

/** Format a Sharpe/Sortino/Calmar ratio. */
export function formatRatio(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

/** Clamp a value between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Format a date string as MMM YYYY. */
export function formatMonthYear(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-IN", { month: "short", year: "numeric" });
}

/** Compute drawdown series from an equity array. */
export function computeDrawdown(
  equityCurve: Array<{ date: string; value: number }>
): Array<{ date: string; drawdown: number }> {
  let peak = equityCurve[0]?.value ?? 100;
  return equityCurve.map(({ date, value }) => {
    if (value > peak) peak = value;
    const drawdown = ((value - peak) / peak) * 100;
    return { date, drawdown };
  });
}

/** Sample every Nth item from an array for performance. */
export function downsample<T>(arr: T[], factor: number): T[] {
  if (factor <= 1) return arr;
  return arr.filter((_, i) => i % factor === 0 || i === arr.length - 1);
}
