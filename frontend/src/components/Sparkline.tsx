// Sparkline.tsx — a vital's last 30 minutes as a thin line inside its normal band.
// Gaps stay gaps: a missing reading is never bridged into a normal-looking line.
import type { VitalKey } from '../types';

const RANGE: Record<VitalKey, { lo: number; hi: number; normal: [number, number] }> = {
  hr: { lo: 30, hi: 160, normal: [51, 90] },
  rr: { lo: 0, hi: 40, normal: [12, 20] },
};

export default function Sparkline({ kind, t, v, label, width = 220, height = 44 }: {
  kind: VitalKey; t: number[]; v: (number | null)[]; label: string; width?: number; height?: number;
}) {
  const r = RANGE[kind];
  if (t.length < 2) {
    return <div className="hatch rounded-md" style={{ width: '100%', maxWidth: width, height }} role="img" aria-label={`${label}: no history yet`} />;
  }
  const t0 = t[0];
  const span = Math.max(1, t[t.length - 1] - t0);
  const x = (ti: number) => ((ti - t0) / span) * width;
  const y = (vi: number) => height - ((Math.min(r.hi, Math.max(r.lo, vi)) - r.lo) / (r.hi - r.lo)) * height;
  const segs: string[] = [];
  let cur = '';
  v.forEach((vi, i) => {
    if (vi === null || !Number.isFinite(vi)) {
      if (cur) segs.push(cur);
      cur = '';
      return;
    }
    cur += `${cur ? 'L' : 'M'}${x(t[i]).toFixed(1)},${y(vi).toFixed(1)}`;
  });
  if (cur) segs.push(cur);
  const last = [...v].reverse().find((q) => q !== null) ?? null;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="w-full" style={{ maxWidth: width, height }}
         role="img" aria-label={`${label} over the last ${Math.round(span / 60)} minutes${last !== null ? `, latest ${Math.round(last)}` : ''}`}>
      <rect x={0} width={width} y={y(r.normal[1])} height={y(r.normal[0]) - y(r.normal[1])}
            fill="var(--surface-3)" opacity={0.7} />
      {segs.map((d, i) => (
        <path key={i} d={d} fill="none" stroke={kind === 'hr' ? 'var(--hr-line)' : 'var(--rr-line)'}
              strokeWidth={1.75} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      ))}
    </svg>
  );
}
