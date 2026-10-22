// SourcesPanel.tsx — every estimator behind HR and RR, side by side.
// "Fused" sources make up the number on screen; "shown only" sources (the learned models,
// unless signal.yaml learned.fuse is on) are displayed for comparison and never alarm.
import { FlaskConical, Layers } from 'lucide-react';
import { SOURCES } from '../lib/format';
import type { PatientSnapshot, VitalKey } from '../types';
import { Quality } from './ui';

const ORDER: Record<VitalKey, string[]> = {
  rr: ['rr.depth', 'rr.thermal', 'rr.cnn_lbrd', 'rr.cnn_d435'],
  hr: ['hr.pos', 'hr.efficientphys'],
};

function Row({ src, snap, vital }: { src: string; snap: PatientSnapshot; vital: VitalKey }) {
  const v = snap.vitals?.[vital];
  const m = v?.sources?.[src];
  const info = SOURCES[src];
  const fused = m ? m.role === 'fused' : !info?.learned || !!v?.learned_fused;
  const reason = (m?.meta?.reason as string | null | undefined) ?? null;
  const unf = (m?.meta?.raw_output ?? m?.meta?.rate_unfiltered) as number | null | undefined;
  const name = src === 'rr.thermal' && m?.meta?.roi === 'warmest_patch' ? 'Thermal · warmest spot' : info?.name ?? src;
  return (
    <tr>
      <td>
        <div className="flex items-center gap-2">
          <svg width="22" height="8" aria-hidden><line x1="1" y1="4" x2="21" y2="4" stroke={info?.color ?? 'var(--ink-2)'}
               strokeWidth="2.5" strokeDasharray={info?.dash} strokeLinecap="round" /></svg>
          <span className="t-label">{name}</span>
          {info?.learned && <FlaskConical className="h-3.5 w-3.5 text-mute" aria-label="experimental" />}
        </div>
        <span className="t-micro text-mute block pl-[30px]">{info?.how}</span>
      </td>
      <td className="num">
        {m?.value != null ? <span className="font-cond font-semibold text-[20px]">{m.value.toFixed(1)}</span>
          : <span className="t-micro text-mute">{m ? (reason?.replace(/_/g, ' ') ?? 'none') : 'no reading'}
            {m && unf != null && <span className="block">net out {unf.toFixed(1)}</span>}</span>}
      </td>
      <td>{m ? <Quality sqi={m.sqi} compact /> : <span className="t-micro text-mute">—</span>}</td>
      <td>
        <span className={`chip ${fused ? '' : 'border-dashed'}`}>
          {fused ? <><Layers aria-hidden />Fused</> : 'Shown only'}
        </span>
      </td>
      <td className="num t-micro text-mute">{m ? `${m.latency_ms.toFixed(0)} ms` : ''}</td>
    </tr>
  );
}

export default function SourcesPanel({ snap }: { snap: PatientSnapshot }) {
  const learnedFused = !!snap.vitals?.rr?.learned_fused;
  return (
    <section className="card card-pad flex flex-col gap-3 min-w-0" aria-labelledby="src-h">
      <div className="flex flex-col gap-1">
        <span className="t-eyebrow text-mute">Where the numbers come from</span>
        <h2 id="src-h" className="t-head">Estimators, side by side</h2>
        <p className="t-label text-ink2 max-w-[70ch]">
          Fused paths are blended by inverse-variance weighting; if they disagree, no number is shown.
          {learnedFused
            ? ' Learned models are currently fused (signal.yaml learned.fuse: true).'
            : ' Learned models (dashed) run alongside for comparison and never drive alarms until you validate them on the Compare page.'}
        </p>
      </div>
      <div className="overflow-x-auto">
        {(['rr', 'hr'] as const).map((k) => (
          <table key={k} className="data min-w-[560px] mb-2">
            <thead>
              <tr>
                <th>{k === 'rr' ? 'Respiratory rate (br/min)' : 'Heart rate (bpm)'}</th>
                <th className="num">Estimate</th><th>Quality</th><th>Role</th><th className="num">Compute</th>
              </tr>
            </thead>
            <tbody>{ORDER[k].map((s) => <Row key={s} src={s} snap={snap} vital={k} />)}</tbody>
          </table>
        ))}
      </div>
    </section>
  );
}
