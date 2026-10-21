// VitalsTrendChart.tsx — one vital over time, against its NEWS2 scoring bands.
//
// Data-viz rules applied: one series per chart (HR and RR never share an axis),
// a fixed clinical y-range so noise is not magnified, gaps stay gaps (a missing
// reading is never interpolated into a normal one), thin 2px line, recessive
// solid hairline grid, crosshair tooltip, and a table view as the accessible twin.
import { Table2, LineChart as LineIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { clock, fmtRate } from '../lib/format';
import type { History, VitalKey } from '../types';

interface Band { from: number; to: number; score: number }

const SPEC: Record<VitalKey, { title: string; unit: string; color: string; domain: [number, number];
  ticks: number[]; bands: Band[] }> = {
  hr: {
    title: 'Heart rate', unit: 'bpm', color: 'var(--hr-line)', domain: [30, 160],
    ticks: [40, 60, 80, 100, 120, 140, 160],
    bands: [
      { from: 30, to: 40.5, score: 3 }, { from: 40.5, to: 50.5, score: 1 },
      { from: 90.5, to: 110.5, score: 1 }, { from: 110.5, to: 130.5, score: 2 },
      { from: 130.5, to: 160, score: 3 },
    ],
  },
  rr: {
    title: 'Respiratory rate', unit: 'br/min', color: 'var(--rr-line)', domain: [0, 40],
    ticks: [0, 8, 12, 20, 25, 30, 40],
    bands: [
      { from: 0, to: 8.5, score: 3 }, { from: 8.5, to: 11.5, score: 1 },
      { from: 20.5, to: 24.5, score: 2 }, { from: 24.5, to: 40, score: 3 },
    ],
  },
};

const BAND_FILL: Record<number, string> = {
  1: 'rgba(255,155,61,0.08)', 2: 'rgba(255,155,61,0.16)', 3: 'rgba(255,77,94,0.15)',
};

function BandLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 t-micro text-mute">
      <span>Shading shows the early-warning score in that range:</span>
      {[1, 2, 3].map((s) => (
        <span key={s} className="inline-flex items-center gap-1.5">
          <i className="h-2.5 w-4 rounded-sm border border-hairline" style={{ background: BAND_FILL[s] }} />+{s}
        </span>
      ))}
    </div>
  );
}

interface Row { t: number; v: number | null }

function TipBox({ active, payload, unit }: {
  active?: boolean; payload?: { payload: Row }[]; unit: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-lg border border-hairline bg-raised px-3 py-2 shadow-lg">
      <div className="t-micro text-mute tnum">{clock(row.t)}</div>
      <div className="t-label text-ink tnum">
        {row.v === null ? 'No reading' : `${fmtRate(row.v)} ${unit}`}
      </div>
    </div>
  );
}

export default function VitalsTrendChart({ kind, history, minutes }: {
  kind: VitalKey;
  history: History | null;
  minutes: number;
}) {
  const spec = SPEC[kind];
  const [table, setTable] = useState(false);

  const rows: Row[] = useMemo(() => {
    if (!history) return [];
    const vals = history[kind];
    return history.t.map((t, i) => ({ t, v: vals[i] ?? null }));
  }, [history, kind]);

  const now = rows.length ? rows[rows.length - 1].t : Date.now() / 1000;
  const xDomain: [number, number] = [now - minutes * 60, now];
  const latest = [...rows].reverse().find((r) => r.v !== null)?.v ?? null;
  const coverage = rows.length ? rows.filter((r) => r.v !== null).length / rows.length : 0;

  return (
    <section className="panel p-5 flex flex-col gap-3 min-w-0" aria-labelledby={`trend-${kind}`}>
      <div className="flex items-center gap-3">
        <h3 id={`trend-${kind}`} className="t-head">{spec.title}</h3>
        <span className="t-label text-mute tnum">
          {latest === null ? 'no recent reading' : `latest ${fmtRate(latest)} ${spec.unit}`}
          {rows.length > 0 && `, measured ${Math.round(coverage * 100)}% of the time`}
        </span>
        <button type="button" className="btn btn-ghost h-8 px-2 ml-auto" onClick={() => setTable((x) => !x)}
                aria-pressed={table}>
          {table ? <LineIcon className="h-4 w-4" aria-hidden /> : <Table2 className="h-4 w-4" aria-hidden />}
          {table ? 'Chart' : 'Table'}
        </button>
      </div>

      {rows.length < 2 ? (
        <div className="h-[220px] grid place-items-center rounded-lg stale-hatch">
          <p className="t-label text-ink2">Collecting readings for the trend.</p>
        </div>
      ) : table ? (
        <div className="max-h-[220px] overflow-auto rounded-lg border border-hairline">
          <table className="w-full t-label tnum">
            <thead className="sticky top-0 bg-raised text-ink2">
              <tr><th className="text-left px-3 py-1.5 font-medium">Time</th>
                <th className="text-right px-3 py-1.5 font-medium">{spec.title} ({spec.unit})</th></tr>
            </thead>
            <tbody>
              {rows.filter((_, i) => i % Math.max(1, Math.floor(rows.length / 60)) === 0).reverse().map((r) => (
                <tr key={r.t} className="rule-t">
                  <td className="px-3 py-1 text-ink2">{clock(r.t)}</td>
                  <td className="px-3 py-1 text-right">{r.v === null ? <span className="text-mute">no reading</span> : fmtRate(r.v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-[220px]" role="img"
             aria-label={`${spec.title} over the last ${minutes} minutes, shaded by early-warning band`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
              {spec.bands.map((b) => (
                <ReferenceArea key={b.from} y1={b.from} y2={b.to} fill={BAND_FILL[b.score]} stroke="none"
                               ifOverflow="hidden" />
              ))}
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="t" type="number" domain={xDomain} scale="time" tickFormatter={clock}
                     stroke="var(--axis)" tick={{ fill: 'var(--mute)', fontSize: 12 }} tickLine={false}
                     minTickGap={48} allowDataOverflow />
              <YAxis domain={spec.domain} ticks={spec.ticks} stroke="var(--axis)" tickLine={false}
                     tick={{ fill: 'var(--mute)', fontSize: 12 }} width={44} allowDataOverflow />
              <Tooltip content={<TipBox unit={spec.unit} />} isAnimationActive={false}
                       cursor={{ stroke: 'var(--ink-2)', strokeWidth: 1 }} />
              <Line type="linear" dataKey="v" stroke={spec.color} strokeWidth={2} dot={false}
                    connectNulls={false} isAnimationActive={false}
                    activeDot={{ r: 4, stroke: 'var(--panel)', strokeWidth: 2, fill: spec.color }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {!table && rows.length >= 2 && <BandLegend />}
    </section>
  );
}
