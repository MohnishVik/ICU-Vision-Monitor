// EWSIndicator.tsx — respiratory-cardiac early-warning sub-score with its reasons.
// Dark cockpit: a routine score is quiet text; amber/red get a filled band, an
// icon and a label. The explanation sentence is the point: it tells the nurse
// *why*, e.g. "RR 16→26 over 40 min. Suggest urgent clinical assessment."
import { Info, OctagonAlert, TriangleAlert } from 'lucide-react';
import type { EWS } from '../types';

const LEVEL_TEXT = { green: 'Routine', amber: 'Watch closely', red: 'Urgent review', unknown: 'No score' };

export default function EWSIndicator({ ews, compact = false }: { ews: EWS | null; compact?: boolean }) {
  if (!ews) {
    return <p className="t-label text-mute">Early-warning score appears once vitals arrive.</p>;
  }
  const level = ews.level;
  const loud = level === 'amber' || level === 'red';
  const Icon = level === 'red' ? OctagonAlert : TriangleAlert;
  const band =
    level === 'red' ? 'bg-[#3d141a] border-[#7a2630] text-[#ffd0d4]'
      : level === 'amber' ? 'bg-[#3a2611] border-[#6b4419] text-[#ffd9b0]'
        : 'bg-transparent border-hairline text-ink';

  return (
    <div className="flex flex-col gap-3">
      <div className={`flex items-center gap-4 rounded-xl border px-4 py-3 ${band}`}>
        <div className="flex items-baseline gap-1.5">
          <span className="t-num-s tnum">{ews.total ?? '--'}</span>
          <span className="t-label opacity-70">/ 6</span>
        </div>
        <div className="flex flex-col min-w-0">
          <span className="t-head flex items-center gap-2">
            {loud && <Icon className="h-5 w-5 shrink-0" aria-hidden />}
            {LEVEL_TEXT[level]}
          </span>
          {!compact && <span className="t-label opacity-80">{ews.action_text || 'Early-warning score'}</span>}
        </div>
      </div>

      {!compact && (
        <>
          <p className="t-body text-ink max-w-[62ch]">{ews.explanation || 'Vitals stable.'}</p>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1">
            {(['rr', 'hr'] as const).map((k) => (
              <div key={k} className="flex items-center justify-between rule-b py-1.5">
                <dt className="t-label text-ink2">{k === 'rr' ? 'Respiratory rate' : 'Heart rate'}</dt>
                <dd className="t-label tnum">
                  {ews.subscores[k] === null || ews.subscores[k] === undefined ? (
                    <span className="text-mute">not scored</span>
                  ) : (
                    `+${ews.subscores[k]}`
                  )}
                </dd>
              </div>
            ))}
          </dl>
          <p className="t-micro text-mute flex gap-1.5 items-start max-w-[62ch]">
            <Info className="h-3.5 w-3.5 mt-[1px] shrink-0" aria-hidden />
            Camera-derived RR and HR only. Not a full NEWS2: SpO₂, temperature, blood pressure and
            consciousness are not measured.
            {ews.missing.length > 0 && ` Missing now: ${ews.missing.map((m) => m.toUpperCase()).join(', ')}.`}
          </p>
        </>
      )}
    </div>
  );
}
