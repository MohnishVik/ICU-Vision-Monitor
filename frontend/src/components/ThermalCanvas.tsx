// ThermalCanvas.tsx — renders a raw thermal grid (centi-degC) as a heatmap.
//
// Colour is computed here from the temperatures, so the legend always matches
// the frame and hovering reads the real value. Range runs from the 2nd percentile
// to the 3rd-hottest pixel: up to two stuck/hot sensor pixels cannot wash the
// image out, but the face (where breathing is read) keeps its detail instead of
// clipping to white.
import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { INFERNO, INFERNO_CSS } from '../lib/inferno';
import type { ThermalGrid } from '../types';

export interface ThermalMarker { x: number; y: number; label: string; pending?: boolean }

export function robustRange(data: number[]): [number, number] {
  if (!data.length) return [20, 37];
  const s = [...data].sort((a, b) => a - b);
  const lo = s[Math.floor(s.length * 0.02)] / 100;
  // 3rd-hottest pixel: ignores up to two stuck/hot sensor pixels but keeps the face
  // (often < 0.5 % of the frame, so any percentile cap would clip it to white)
  const hi = s[Math.max(0, s.length - 3)] / 100;
  return hi - lo < 1 ? [lo - 0.5, lo + 0.5] : [lo, hi];
}

export function ThermalLegend({ lo, hi }: { lo: number; hi: number }) {
  return (
    <div className="flex items-center gap-3 t-micro text-ink2 tnum" aria-label={`Colour scale from ${lo.toFixed(1)} to ${hi.toFixed(1)} degrees Celsius`}>
      <span>{lo.toFixed(1)} °C</span>
      <span className="h-2 flex-1 rounded-full" style={{ background: INFERNO_CSS }} aria-hidden />
      <span>{hi.toFixed(1)} °C</span>
    </div>
  );
}

export default function ThermalCanvas({ grid, range, roi, markers, onPick, pixelated = false, label }: {
  grid: ThermalGrid;
  range: [number, number];
  roi?: [number, number, number, number] | null;
  markers?: ThermalMarker[];
  onPick?: (x: number, y: number) => void;
  pixelated?: boolean;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number; c: number } | null>(null);
  const { w, h, data } = grid;

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext('2d');
    if (!el || !ctx || data.length !== w * h) return;
    const [lo, hi] = range;
    const img = ctx.createImageData(w, h);
    const span = Math.max(1e-6, hi - lo);
    for (let i = 0; i < w * h; i++) {
      const k = Math.max(0, Math.min(255, Math.round(((data[i] / 100 - lo) / span) * 255))) * 3;
      img.data[i * 4] = INFERNO[k];
      img.data[i * 4 + 1] = INFERNO[k + 1];
      img.data[i * 4 + 2] = INFERNO[k + 2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, w, h, range]);

  const toGrid = (e: PointerEvent): { x: number; y: number } | null => {
    const el = svg.current;
    const m = el?.getScreenCTM();
    if (!el || !m) return null;
    const p = el.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(m.inverse());
    if (q.x < 0 || q.y < 0 || q.x > w || q.y > h) return null;
    return { x: q.x, y: q.y };
  };

  const r = Math.max(w, h) / 50;

  return (
    <div className="relative w-full overflow-hidden rounded-lg bg-black" style={{ aspectRatio: `${w} / ${h}` }}>
      <canvas ref={canvas} width={w} height={h} className="absolute inset-0 h-full w-full"
              style={{ imageRendering: pixelated ? 'pixelated' : 'auto' }} aria-hidden />
      <svg ref={svg} viewBox={`0 0 ${w} ${h}`} className={`absolute inset-0 h-full w-full ${onPick ? 'cursor-crosshair' : ''}`}
           role="img" aria-label={label}
           onPointerMove={(e) => {
             const g = toGrid(e);
             if (!g) return setHover(null);
             const ix = Math.min(w - 1, Math.floor(g.x));
             const iy = Math.min(h - 1, Math.floor(g.y));
             setHover({ x: g.x, y: g.y, c: data[iy * w + ix] / 100 });
           }}
           onPointerLeave={() => setHover(null)}
           onPointerDown={(e) => {
             if (!onPick) return;
             const g = toGrid(e);
             if (g) onPick(g.x, g.y);
           }}>
        {roi && (
          <rect x={roi[0]} y={roi[1]} width={roi[2] - roi[0]} height={roi[3] - roi[1]} fill="none"
                stroke="var(--rr)" strokeWidth={2} vectorEffect="non-scaling-stroke" rx={0.4} />
        )}
        {markers?.map((m, i) => (
          <g key={i} opacity={m.pending ? 0.75 : 1}>
            <circle cx={m.x} cy={m.y} r={r * 1.25} fill="var(--scrub)" stroke="#06262a"
                    strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
            <text x={m.x} y={m.y} dy="0.36em" textAnchor="middle" fontSize={r * 1.5}
                  fontFamily="Barlow Condensed" fontWeight={600} fill="#06262a">{m.label}</text>
          </g>
        ))}
        {hover && (
          <g pointerEvents="none">
            <line x1={hover.x} y1={0} x2={hover.x} y2={h} stroke="rgba(230,238,236,.5)" strokeWidth={1}
                  vectorEffect="non-scaling-stroke" />
            <line x1={0} y1={hover.y} x2={w} y2={hover.y} stroke="rgba(230,238,236,.5)" strokeWidth={1}
                  vectorEffect="non-scaling-stroke" />
          </g>
        )}
      </svg>
      {hover && (
        <span className="absolute right-2 top-2 rounded-md bg-black/70 px-2 py-0.5 t-micro tnum text-ink pointer-events-none">
          {hover.c.toFixed(1)} °C
        </span>
      )}
    </div>
  );
}
