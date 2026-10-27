// VitalsTrendChart.tsx — one vital over time against its NEWS2 bands, optionally with every
// estimator overlaid so they can be compared live.
//
// One vital per chart (HR and RR never share an axis); fixed clinical y-range so noise is not
// magnified; gaps stay gaps; learned sources are dashed and named in the legend so colour is
// never the only cue; a table view is the accessible twin.
import { LineChart as LineIcon, Table2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { clock, clockSec, fmtRate, SOURCES } from '../lib/format';
import type { History, VitalKey } from '../types';
import { Tip } from './Tooltip';

interface Band { from: number; to: number; score: number }

const SPEC: Record<VitalKey, { title: string; unit: string; domain: [number, number]; ticks: number[];
  bands: Band[]; sources: string[] }> = {
  hr: {
    title: 'Heart rate', unit: 'bpm', domain: [30, 160], ticks: [40, 60, 80, 100, 120, 140, 160],
    bands: [{ from: 30, to: 40.5, score: 3 }, { from: 40.5, to: 50.5, score: 1 }, { from: 90.5, to: 110.5, score: 1 },
            { from: 110.5, to: 130.5, score: 2 }, { from: 130.5, to: 160, score: 3 }],
    sources: ['hr.pos', 'hr.efficientphys'],
  },
  rr: {
    title: 'Respiratory rate', unit: 'br/min', domain: [0, 40], ticks: [0, 8, 12, 20, 25, 30, 40],
    bands: [{ from: 0, to: 8.5, score: 3 }, { from: 8.5, to: 11.5, score: 1 }, { from: 20.5, to: 24.5, score: 2 },
            { from: 24.5, to: 40, score: 3 }],
    sources: ['rr.depth', 'rr.thermal', 'rr.cnn_lbrd', 'rr.cnn_d435'],
  },
};

const BAND_FILL: Record<number, string> = {
  1: 'color-mix(in srgb, var(--amber) 7%, transparent)',
  2: 'color-mix(in srgb, var(--amber) 14%, transparent)',
  3: 'color-mix(in srgb, var(--red) 13%, transparent)',
};

type Row = Record<string, number | null> & { t: number };

function Swatch({ color, dash, width = 2.5 }: { color: string; dash?: string; width?: number }) {
  return <svg width="22" height="8" aria-hidden><line x1="1" y1="4" x2="21" y2="4" stroke={color} strokeWidth={width}
              strokeDasharray={dash} strokeLinecap="round" /></svg>;
}

export default function VitalsTrendChart({ kind, history, minutes, showSources }: {
  kind: VitalKey; history: History | null; minutes: number; showSources: boolean;
}) {
  const spec = SPEC[kind];
  const [table, setTable] = useState(false);
  const fusedKey = `${kind}.fused`;
  const lineColor = kind === 'hr' ? 'var(--hr-line)' : 'var(--rr-line)';

  const rows: Row[] = useMemo(() => {
    if (!history) return [];
    return history.t.map((t, i) => {
      const r: Row = { t, v: history[kind][i] ?? null };
      for (const s of spec.sources) r[s] = history.sources?.[s]?.[i] ?? null;
      return r;
    });
  }, [history, kind, spec.sources]);

  const present = spec.sources.filter((s) => rows.some((r) => r[s] !== null));
  const now = rows.length ? rows[rows.length - 1].t : Date.now() / 1000;
  const xDomain: [number, number] = [now - minutes * 60, now];
  const latest = [...rows].reverse().find((r) => r.v !== null)?.v ?? null;
  const coverage = rows.length ? rows.filter((r) => r.v !== null).length / rows.length : 0;

  return (
    <section className="card card-pad flex flex-col gap-3 min-w-0" aria-labelledby={`trend-${kind}`}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 id={`trend-${kind}`} className="t-head"><Tip term={kind}>{spec.title}</Tip></h3>
        <Tip term="coverageTrend" className="t-label text-mute tnum">
          {latest === null ? 'no recent reading' : `latest ${fmtRate(latest)} ${spec.unit}`}
          {rows.length > 0 && ` · measured ${Math.round(coverage * 100)}% of the time`}
        </Tip>
        <Tip title={table ? 'Chart view' : 'Table view'} asChild
             content={table ? 'Show the values as a chart again.' : 'Show the same values as a table, for exact numbers and screen readers.'}>
          <button type="button" className="btn btn-ghost btn-sm ml-auto" onClick={() => setTable((x) => !x)} aria-pressed={table}>
            {table ? <LineIcon className="h-4 w-4" aria-hidden /> : <Table2 className="h-4 w-4" aria-hidden />}
            {table ? 'Chart' : 'Table'}
          </button>
        </Tip>
      </div>

      {rows.length < 2 ? (
        <div className="h-[240px] grid place-items-center rounded-md hatch"><p className="t-label text-ink2">Collecting readings for the trend.</p></div>
      ) : table ? (
        <div className="max-h-[240px] overflow-auto rounded-md border border-line">
          <table className="data">
            <thead className="sticky top-0 bg-surface2">
              <tr><th>Time</th><th className="num">Shown ({spec.unit})</th>
                {showSources && present.map((s) => <th key={s} className="num">{SOURCES[s]?.name} ({spec.unit})</th>)}</tr>
            </thead>
            <tbody>
              {rows.filter((_, i) => i % Math.max(1, Math.floor(rows.length / 60)) === 0).reverse().map((r) => (
                <tr key={r.t}>
                  <td className="text-ink2 tnum">{clockSec(r.t)}</td>
                  <td className="num">{r.v === null ? <span className="text-mute">–</span> : fmtRate(r.v)}</td>
                  {showSources && present.map((s) => <td key={s} className="num">{r[s] === null ? <span className="text-mute">–</span> : (r[s] as number).toFixed(1)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-[240px]" role="img"
             aria-label={`${spec.title} over the last ${minutes} minutes${showSources ? ', with each estimator overlaid' : ''}`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
              {spec.bands.map((b) => (
                <ReferenceArea key={b.from} y1={b.from} y2={b.to} fill={BAND_FILL[b.score]} stroke="none" ifOverflow="hidden" />
              ))}
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="t" type="number" domain={xDomain} scale="time" tickFormatter={clock} stroke="var(--axis)"
                     tick={{ fill: 'var(--mute)', fontSize: 12 }} tickLine={false} minTickGap={48} allowDataOverflow />
              <YAxis domain={spec.domain} ticks={spec.ticks} stroke="var(--axis)" tickLine={false}
                     tick={{ fill: 'var(--mute)', fontSize: 12 }} width={44} allowDataOverflow />
              <Tooltip isAnimationActive={false} cursor={{ stroke: 'var(--ink-2)', strokeWidth: 1 }}
                       content={({ active, payload }) => {
                         if (!active || !payload?.length) return null;
                         const r = payload[0].payload as Row;
                         return (
                           <div className="rounded-md border border-line bg-surface2 px-3 py-2 shadow-lg flex flex-col gap-0.5">
                             <span className="t-micro text-ink2 tnum">{clockSec(r.t)}</span>
                             <span className="t-label tnum">Shown: {r.v === null ? 'no reading' : `${fmtRate(r.v)} ${spec.unit}`}</span>
                             {showSources && present.map((s) => (
                               <span key={s} className="t-micro text-ink2 tnum">{SOURCES[s]?.name}: {r[s] === null ? '–' : `${(r[s] as number).toFixed(1)} ${spec.unit}`}</span>
                             ))}
                           </div>
                         );
                       }} />
              {showSources && present.map((s) => (
                <Line key={s} type="linear" dataKey={s} stroke={SOURCES[s]?.color} strokeWidth={1.5}
                      strokeDasharray={SOURCES[s]?.dash} dot={false} connectNulls={false} isAnimationActive={false}
                      activeDot={false} />
              ))}
              <Line type="linear" dataKey="v" stroke={lineColor} strokeWidth={2.25} dot={false} connectNulls={false}
                    isAnimationActive={false} activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2, fill: lineColor }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 t-micro text-mute">
        <Tip term="shownLine" className="inline-flex items-center gap-1.5">
          <Swatch color={lineColor} />Shown ({SOURCES[fusedKey]?.name.toLowerCase()})
        </Tip>
        {showSources && present.map((s) => (
          <Tip key={s} title={SOURCES[s]?.name} className="inline-flex items-center gap-1.5"
               content={`${SOURCES[s]?.how}.${SOURCES[s]?.learned ? ' Experimental learned model, still being validated (dashed line).' : ''}`}>
            <Swatch color={SOURCES[s]?.color ?? 'var(--ink-2)'} dash={SOURCES[s]?.dash} width={1.75} />
            {SOURCES[s]?.name}{SOURCES[s]?.learned ? ' (exp.)' : ''}
          </Tip>
        ))}
        <Tip term="bands" className="inline-flex items-center gap-1.5 ml-auto">
          Shading = EWS band
          {[1, 2, 3].map((s) => <i key={s} className="h-2.5 w-4 rounded-sm border border-line" style={{ background: BAND_FILL[s] }} aria-hidden />)}
        </Tip>
      </div>
    </section>
  );
}
