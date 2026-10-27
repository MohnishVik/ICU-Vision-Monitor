// VitalReadout.tsx — the big HR / RR numeral.
//
// The HR dot beats at the measured rate; the RR bar breathes at the measured rate. When the
// value is null or stale both stop and the numeral becomes a hatched "—" with the reason,
// so liveness itself shows trust.
//
// Every part explains itself on hover / focus: the numeral (what it is, normal range), the
// sub-score chip (its NEWS2 band), the hatched dash (why there is no number), the quality bars.
import { OctagonAlert } from 'lucide-react';
import { fmtRate, vitalUnavailableText } from '../lib/format';
import { reasonKey } from '../lib/glossary';
import type { VitalKey } from '../types';
import { Tip } from './Tooltip';
import { Quality } from './ui';

export const VITAL_META = {
  hr: { name: 'Heart rate', short: 'HR', unit: 'bpm', color: 'var(--hr)' },
  rr: { name: 'Respiratory rate', short: 'RR', unit: 'br/min', color: 'var(--rr)' },
} as const;

function Pulse({ kind, live, value }: { kind: VitalKey; live: boolean; value: number | null }) {
  const color = live ? VITAL_META[kind].color : 'var(--line-strong)';
  const period = live && value ? 60 / Math.max(4, value) : undefined;
  if (kind === 'hr') {
    return <span aria-hidden className={`h-2.5 w-2.5 rounded-full ${live ? 'beat' : ''}`}
                 style={{ background: color, ['--period' as string]: `${period ?? 1}s` }} />;
  }
  return (
    <span aria-hidden className="h-2.5 w-6 rounded-full bg-surface3 overflow-hidden">
      <span className={`block h-full w-full rounded-full ${live ? 'breathe' : ''}`}
            style={{ background: color, transform: live ? undefined : 'scaleX(.3)',
                     ['--period' as string]: `${period ?? 4}s` }} />
    </span>
  );
}

export function SubscoreChip({ s, kind }: { s: number | null; kind: VitalKey }) {
  if (s === null || s === 0) return null;
  return (
    <Tip term={kind === 'rr' ? 'ewsRR' : 'ewsHR'} className={`chip tnum ${s >= 3 ? 'band-red' : 'band-amber'}`}
         content={`This reading adds ${s} ${s === 1 ? 'point' : 'points'} to the early-warning score.`}>
      EWS +{s}
    </Tip>
  );
}

export default function VitalReadout({ kind, value, stale, sqi, subscore, reason, apnea, ageS, size = 'l' }: {
  kind: VitalKey; value: number | null; stale: boolean; sqi: number; subscore: number | null;
  reason: string | null; apnea?: boolean; ageS?: number | null; size?: 'm' | 'l';
}) {
  const m = VITAL_META[kind];
  const live = value !== null && !stale;
  const num = size === 'l' ? 't-num-l' : 't-num-m';

  return (
    <div className="flex flex-col gap-2 min-w-0">
      <div className="flex items-center gap-2">
        <Pulse kind={kind} live={live} value={value} />
        <span className="t-eyebrow text-ink2">{size === 'l' ? m.name : m.short}</span>
        <span className="ml-auto"><SubscoreChip s={subscore} kind={kind} /></span>
      </div>
      {live ? (
        <div className="flex" aria-live="polite">
          <Tip term={kind} className="flex items-baseline gap-2">
            <span className={`${num} tnum`} style={{ color: m.color }}>{fmtRate(value)}</span>
            <span className="t-label text-mute">{m.unit}</span>
          </Tip>
        </div>
      ) : (
        <Tip term={reasonKey(reason, stale, apnea)} title={apnea ? undefined : `${m.name}: ${vitalUnavailableText({ reason, stale, value }, apnea)}`}
             content="The dash on a hatched background means there is no trustworthy number right now; it is never shown as a normal reading."
             className={`flex items-baseline flex-wrap gap-x-3 rounded-md hatch ${size === 'l' ? 'px-2' : 'px-1.5'}`}>
          <span className={`${num} text-mute`} aria-label={`${m.name} unavailable`}>—</span>
          <span className={`t-label ${apnea ? 'text-red t-strong' : 'text-ink2'} inline-flex items-center gap-1.5`}>
            {apnea && <OctagonAlert className="h-4 w-4" aria-hidden />}
            {vitalUnavailableText({ reason, stale, value }, apnea)}
          </span>
        </Tip>
      )}
      <div className="flex items-center gap-3">
        <Quality sqi={live ? sqi : 0} compact={size === 'm'} />
        {size === 'l' && stale && ageS != null && (
          <Tip term="lastReading" className="t-micro text-mute tnum">last reading {Math.round(ageS)} s ago</Tip>
        )}
      </div>
    </div>
  );
}
