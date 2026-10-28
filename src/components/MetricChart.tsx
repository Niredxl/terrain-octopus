import type { MetricsSnapshot } from "../api/client";

export function MetricChart({
  title,
  values,
  color,
  format = (value) => value.toFixed(2),
}: {
  title: string;
  values: number[];
  color: string;
  format?: (value: number) => string;
}) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * 100;
    const y = 88 - ((value - min) / range) * 68;
    return `${x},${y}`;
  }).join(" ");
  const latest = values[values.length - 1];
  return (
    <div className="metric-card">
      <div className="metric-card-head"><span>{title}</span><b style={{ color }}>{format(latest)}</b></div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="metric-chart" aria-label={`${title} chart`}>
        <path d="M0 88 H100 M0 54 H100 M0 20 H100" className="chart-grid" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="2.3" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="metric-range"><span>step 0</span><span>step 120k</span></div>
    </div>
  );
}

export function MetricsGrid({ metrics }: { metrics: MetricsSnapshot[] }) {
  return (
    <div className="metrics-grid">
      <MetricChart title="mean reward" values={metrics.map((metric) => metric.reward)} color="#b9f36b" />
      <MetricChart title="policy loss" values={metrics.map((metric) => metric.policyLoss)} color="#66e3ff" />
      <MetricChart title="entropy" values={metrics.map((metric) => metric.entropy)} color="#f7b267" />
      <MetricChart title="success rate" values={metrics.map((metric) => metric.successRate)} color="#b39cff" format={(value) => `${Math.round(value * 100)}%`} />
    </div>
  );
}
