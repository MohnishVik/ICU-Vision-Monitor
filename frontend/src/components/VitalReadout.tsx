// VitalReadout.tsx — the large HR / RR numeral.
//
// The heart marker beats at the measured HR; the breath bar breathes at the
// measured RR. When the value is null or stale both stop and the numeral turns
// to a hatched "--" with the reason, so liveness itself shows trust.
import { OctagonAlert } from 'lucide-react';
import { fmtRate, vitalUnavailableText } from '../lib/format';
import type { VitalKey } from '../types';
import SignalQualityBar from './SignalQualityBar';

interface Props {
  kind: VitalKey;
  value: number | null;
  stale: boolean;
  sqi: number;
  subscore: number | null;
  reason: string | null;
  apnea?: boolean;
  ageS?: number | null;
  size?: 'm' | 'l';
}

const META = {
  hr: { name: 'Heart rate', short: 'HR', unit: 'bpm', color: 'var(--hr)' },
  rr: { name: 'Respiratory rate', short: 'RR', unit: 'br/min', color: 'var(--rr)' },
} as const;

function SubscoreChip({ s }: { s: number | null }) {
  if (s === null || s === 0) return null;
  const red = s >= 3;
  return (
    <span
      className={`t-micro tnum inline-flex items-center h-6 px-2 rounded-md ${
        red ? 'bg-[#3d141a] text-[#ffb3ba]' : 'bg-[#3a2611] text-[#ffc488]'
      }`}
      title={`Early-warning sub-score ${s}`}
    >
      EWS +{s}
    </span>
  );
}

export default function VitalReadout({ kind, value, stale, sqi, subscore, reason, apnea, ageS,
                                       size = 'l' }: Props) {
  const m = META[kind];
  const live = value !== null && !stale;
  const period = live && value ? 60 / Math.max(4, value) : undefined;
  const numCls = size === 'l' ? 't-num-l' : 't-num-m';

  return (
    <div className="flex flex-col gap-2 min-w-0">
      <div className="flex items-center gap-2">
        {kind === 'hr' ? (
          <span
            aria-hidden
            className={`h-2.5 w-2.5 rounded-full ${live ? 'beat' : ''}`}
            style={{ background: live ? m.color : 'var(--hairline-strong)',
                     ['--period' as string]: `${period ?? 1}s` }}
          />
        ) : (
          <span aria-hidden className="h-2.5 w-5 rounded-full bg-hairline overflow-hidden">
            <span
              className={`block h-full w-full rounded-full ${live ? 'breathe' : ''}`}
              style={{ background: live ? m.color : 'var(--hairline-strong)',
                       transform: live ? undefined : 'scaleX(.3)',
                       ['--period' as string]: `${period ?? 4}s` }}
            />
          </span>
        )}
        <span className="t-label text-ink2">{size === 'l' ? m.name : m.short}</span>
        <span className="ml-auto"><SubscoreChip s={subscore} /></span>
      </div>

      {live ? (
        <div className="flex items-baseline gap-2" aria-live="polite">
          <span className={`${numCls}`} style={{ color: m.color }}>{fmtRate(value)}</span>
          <span className="t-label text-mute">{m.unit}</span>
        </div>
      ) : (
        <div className={`flex items-baseline gap-3 rounded-lg stale-hatch ${size === 'l' ? 'px-2' : 'px-1.5'}`}>
          <span className={`${numCls} text-mute`} aria-label={`${m.name} unavailable`}>{'—'}</span>
          <span className={`t-label ${apnea ? 'text-red' : 'text-ink2'} flex items-center gap-1.5`}>
            {apnea && <OctagonAlert className="h-4 w-4" aria-hidden />}
            {vitalUnavailableText({ reason, stale, value }, apnea)}
          </span>
        </div>
      )}

      <div className="flex items-center gap-3">
        <SignalQualityBar sqi={live ? sqi : 0} compact={size === 'm'} />
        {size === 'l' && stale && ageS !== null && ageS !== undefined && (
          <span className="t-micro text-mute tnum">last reading {Math.round(ageS)} s ago</span>
        )}
      </div>
    </div>
  );
}
