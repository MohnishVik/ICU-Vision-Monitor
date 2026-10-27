// SourcesPanel.tsx — every estimator behind HR and RR, side by side.
// "Fused" sources make up the number on screen; "shown only" sources (the learned models,
// unless signal.yaml learned.fuse is on) are displayed for comparison and never alarm.
import { FlaskConical, Layers } from 'lucide-react';
import { SOURCES } from '../lib/format';
import { sourceReasonKey } from '../lib/glossary';
import type { PatientSnapshot, VitalKey } from '../types';
import { InfoTip, Tip } from './Tooltip';
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
  const unit = vital === 'rr' ? 'br/min' : 'bpm';
  return (
    <tr>
      <td>
        <div className="flex items-center gap-2">
          <svg width="22" height="8" aria-hidden><line x1="1" y1="4" x2="21" y2="4" stroke={info?.color ?? 'var(--ink-2)'}
               strokeWidth="2.5" strokeDasharray={info?.dash} strokeLinecap="round" /></svg>
          <span className="t-label">{name}</span>
          {info?.learned && (
            <Tip term="learned" className="inline-flex text-mute">
              <FlaskConical className="h-3.5 w-3.5" aria-label="experimental" />
            </Tip>
          )}
        </div>
        <span className="t-micro text-mute block pl-[30px]">{info?.how}</span>
      </td>
      <td className="num">
        {m?.value != null ? (
          <span className="inline-flex items-baseline gap-1">
            <span className="font-cond font-semibold text-[20px]">{m.value.toFixed(1)}</span>
            <span className="t-micro text-mute">{unit}</span>
          </span>
        ) : (
          <span className="t-micro text-mute inline-flex flex-col items-end">
            <Tip term={m ? sourceReasonKey(reason) : 'src_none'}>{m ? (reason?.replace(/_/g, ' ') ?? 'none') : 'no reading'}</Tip>
            {m && unf != null && <Tip term="netOut">net out {unf.toFixed(1)} {unit}</Tip>}
          </span>
        )}
      </td>
      <td>{m ? <Quality sqi={m.sqi} compact /> : <span className="t-micro text-mute">—</span>}</td>
      <td>
        <Tip term={fused ? 'fused' : 'shownOnly'} className={`chip ${fused ? '' : 'border-dashed'}`}
             content={fused && info?.learned ? 'As a learned model it is a secondary path: dropped for any second in which it '
               + 'disagrees with the signal-processing reading.' : undefined}>
          {fused ? <><Layers aria-hidden />Fused</> : 'Shown only'}
        </Tip>
      </td>
      <td className="num t-micro text-mute">{m ? <Tip term="compute">{m.latency_ms.toFixed(0)} ms</Tip> : ''}</td>
    </tr>
  );
}

export default function SourcesPanel({ snap }: { snap: PatientSnapshot }) {
  const learnedFused = !!(snap.vitals?.rr?.learned_fused || snap.vitals?.hr?.learned_fused);
  return (
    <section className="card card-pad flex flex-col gap-3 min-w-0" aria-labelledby="src-h">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <h2 id="src-h" className="t-head">Estimators, side by side</h2>
          <InfoTip term="estimators" />
        </div>
        <p className="t-label text-ink2 max-w-[70ch]">
          Fused paths are blended by inverse-variance weighting; if the signal-processing paths
          disagree, no number is shown.
          {learnedFused
            ? ' Learned models marked Fused count only while they agree with the signal-processing reading; otherwise they are left out for that second.'
            : ' Learned models (dashed) run alongside for comparison and never drive alarms until you validate them on the Compare page.'}
        </p>
      </div>
      <div className="overflow-x-auto">
        {(['rr', 'hr'] as const).map((k) => (
          <table key={k} className="data min-w-[560px] mb-2">
            <thead>
              <tr>
                <th><Tip term={k} className="tip-term">{k === 'rr' ? 'Respiratory rate (br/min)' : 'Heart rate (bpm)'}</Tip></th>
                <th className="num"><Tip term="estimate" className="tip-term">Estimate</Tip></th>
                <th><Tip term="sqi" className="tip-term">Quality</Tip></th>
                <th><Tip term="role" className="tip-term">Role</Tip></th>
                <th className="num"><Tip term="compute" className="tip-term">Compute</Tip></th>
              </tr>
            </thead>
            <tbody>{ORDER[k].map((s) => <Row key={s} src={s} snap={snap} vital={k} />)}</tbody>
          </table>
        ))}
      </div>
    </section>
  );
}
