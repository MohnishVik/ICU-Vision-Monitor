// EWSIndicator.tsx — respiratory-cardiac early-warning sub-score with its reasons.
// Routine = quiet text; orange/red get a filled band, an icon and a label. The explanation
// sentence is the point: it tells the nurse why ("RR 16→26 over 40 min").
import { Info, OctagonAlert, TriangleAlert } from 'lucide-react';
import { glossKey } from '../lib/glossary';
import type { EWS } from '../types';
import { Tip } from './Tooltip';

const LEVEL_TEXT = { green: 'Routine', amber: 'Watch closely', red: 'Urgent review', unknown: 'No score' };

export default function EWSIndicator({ ews }: { ews: EWS | null }) {
  if (!ews) return <p className="t-label text-mute">The early-warning score appears once vitals arrive.</p>;
  const level = ews.level;
  const loud = level === 'amber' || level === 'red';
  const Icon = level === 'red' ? OctagonAlert : TriangleAlert;
  return (
    <div className="flex flex-col gap-3">
      <div className={`flex items-center gap-4 rounded-lg border px-4 py-3 ${
        level === 'red' ? 'band-red' : level === 'amber' ? 'band-amber' : 'border-line'}`}>
        <Tip term="ewsTotal" className="flex items-baseline gap-1"
             content={ews.missing.length ? `Missing now: ${ews.missing.map((m) => m.toUpperCase()).join(', ')}.` : undefined}>
          <span className="t-num-s tnum">{ews.total ?? '–'}</span>
          <span className="t-label opacity-70">/ 6</span>
        </Tip>
        <div className="flex flex-col min-w-0">
          <span className="flex">
            <Tip term={glossKey('ews_', level, 'ews_unknown')} className="t-head flex items-center gap-2">
              {loud && <Icon className="h-5 w-5 shrink-0" aria-hidden />}{LEVEL_TEXT[level]}
            </Tip>
          </span>
          <span className="t-label opacity-85">{ews.action_text || 'Early-warning score'}</span>
        </div>
      </div>
      <p className="t-body max-w-[62ch]">{ews.explanation || 'Vitals stable.'}</p>
      <dl className="grid grid-cols-2 gap-x-6">
        {(['rr', 'hr'] as const).map((k) => (
          <div key={k} className="flex items-center justify-between rule-b py-1.5">
            <dt className="t-label text-ink2">
              <Tip term={k === 'rr' ? 'ewsRR' : 'ewsHR'} className="tip-term">{k === 'rr' ? 'Respiratory rate' : 'Heart rate'}</Tip>
            </dt>
            <dd className="t-label tnum">
              {ews.subscores[k] == null ? <Tip term="notScored" className="text-mute">not scored</Tip> : `+${ews.subscores[k]}`}
            </dd>
          </div>
        ))}
      </dl>
      <p className="t-micro text-mute flex gap-1.5 items-start max-w-[62ch]">
        <Info className="h-3.5 w-3.5 mt-[2px] shrink-0" aria-hidden />
        Camera-derived RR and HR only. Not a full NEWS2: SpO₂, temperature, blood pressure and
        consciousness are not measured.{ews.missing.length > 0 && ` Missing now: ${ews.missing.map((m) => m.toUpperCase()).join(', ')}.`}
      </p>
    </div>
  );
}
