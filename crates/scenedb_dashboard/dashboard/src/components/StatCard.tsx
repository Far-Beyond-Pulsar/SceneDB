"use client";

interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  color?: "blue" | "green" | "yellow" | "red" | "gray";
}

const colorMap = {
  blue: { value: "text-github-accent", bg: "bg-github-accent/10", bar: "bg-github-accent" },
  green: { value: "text-github-green", bg: "bg-github-green/10", bar: "bg-github-green" },
  yellow: { value: "text-github-yellow", bg: "bg-github-yellow/10", bar: "bg-github-yellow" },
  red: { value: "text-github-red", bg: "bg-github-red/10", bar: "bg-github-red" },
  gray: { value: "text-github-text", bg: "bg-github-border-muted/50", bar: "bg-github-text-muted" },
};

export default function StatCard({ label, value, subtitle, trend, color = "blue" }: StatCardProps) {
  const c = colorMap[color];
  return (
    <div className="card relative overflow-hidden group hover:border-github-border transition-colors">
      <div className={`absolute top-0 left-0 w-full h-0.5 ${c.bar} opacity-60`} />
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs font-medium text-github-text-muted uppercase tracking-wider">
            {label}
          </p>
          <p className={`stat-value ${c.value}`}>
            {typeof value === "number" ? value.toLocaleString() : value}
          </p>
          {subtitle && (
            <p className="text-xs text-github-text-muted">{subtitle}</p>
          )}
        </div>
        <div className={`w-9 h-9 rounded-lg ${c.bg} flex items-center justify-center`}>
          {trend === "up" && (
            <svg className={`w-4 h-4 ${c.value}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <polyline points="18 15 12 9 6 15" />
            </svg>
          )}
          {trend === "down" && (
            <svg className={`w-4 h-4 ${c.value}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
