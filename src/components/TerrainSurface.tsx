import type { CSSProperties } from "react";
import type { TerrainSample } from "../api/client";

export function TerrainSurface({ sample }: { sample: TerrainSample }) {
  return (
    <div className="terrain-card" style={{ "--terrain-accent": sample.accent } as CSSProperties}>
      <div className="terrain-card-head">
        <div>
          <strong>{sample.label}</strong>
          <span>{sample.family}</span>
        </div>
        <span className="status-dot" />
      </div>
      <div className="terrain-viewport">
        <svg viewBox="0 0 220 140" role="img" aria-label={`${sample.label} terrain surface`}>
          <defs>
            <linearGradient id={`fill-${sample.id}`} x1="0" x2="1" y1="0" y2="1">
              <stop offset="0" stopColor={sample.accent} stopOpacity=".84" />
              <stop offset="1" stopColor="#0d1d28" stopOpacity=".2" />
            </linearGradient>
          </defs>
          {sample.values.map((row, y) =>
            row.map((height, x) => {
              const px = 16 + x * 20 - y * 7;
              const py = 26 + y * 8 + x * 3 - height * 23;
              const next = sample.values[y]?.[x + 1] ?? height;
              const down = sample.values[y + 1]?.[x] ?? height;
              const points = `${px},${py} ${px + 20},${py + 3 - (next - height) * 18} ${px + 13},${py + 11 + (down - height) * 11} ${px - 7},${py + 8}`;
              return <polygon key={`${x}-${y}`} points={points} fill={`url(#fill-${sample.id})`} opacity={0.55 + height * 0.45} />;
            }),
          )}
        </svg>
      </div>
      <div className="terrain-stats">
        <span><small>MEAN Z</small><b>{sample.stats.elevation}</b></span>
        <span><small>SLOPE</small><b>{sample.stats.slope}</b></span>
        <span><small>ROUGH</small><b>{sample.stats.roughness}</b></span>
      </div>
    </div>
  );
}
