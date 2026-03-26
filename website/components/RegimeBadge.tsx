import { cn } from "@/lib/utils";

type Regime = "bull" | "bear" | "transition" | "high-vol";

interface RegimeBadgeProps {
  regime: Regime;
  active?: boolean;
  size?: "sm" | "md";
}

const REGIME_CONFIG: Record<Regime, { label: string; bg: string; text: string; border: string; dot: string }> = {
  bull: {
    label: "Bull",
    bg: "bg-teal-400/15",
    text: "text-teal-300",
    border: "border-teal-400/25",
    dot: "bg-teal-400",
  },
  bear: {
    label: "Bear",
    bg: "bg-coral-400/15",
    text: "text-coral-300",
    border: "border-coral-400/25",
    dot: "bg-coral-400",
  },
  transition: {
    label: "Transition",
    bg: "bg-gray-400/10",
    text: "text-gray-300",
    border: "border-gray-400/20",
    dot: "bg-gray-400",
  },
  "high-vol": {
    label: "High-VIX",
    bg: "bg-amber-400/15",
    text: "text-amber-300",
    border: "border-amber-400/25",
    dot: "bg-amber-400",
  },
};

export default function RegimeBadge({ regime, active = false, size = "md" }: RegimeBadgeProps) {
  const cfg = REGIME_CONFIG[regime];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        cfg.bg,
        cfg.text,
        cfg.border,
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm"
      )}
    >
      <span
        className={cn(
          "rounded-full flex-shrink-0",
          cfg.dot,
          active ? "animate-pulse-slow" : "",
          size === "sm" ? "w-1.5 h-1.5" : "w-2 h-2"
        )}
      />
      {cfg.label}
    </span>
  );
}
