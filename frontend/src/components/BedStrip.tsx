// BedStrip.tsx — one bed on the ward view, laid out like a monitor channel strip:
// who/where | HR + 30 min | RR + 30 min | early warning | pain (experimental).
// Quiet when stable, loud only on deviation. Collapses to a card on narrow screens.
// The bed name is a stretched link covering the card, so the values inside stay hoverable and
// focusable for their explanations while a click anywhere else still opens the bed.
import { CameraOff, ChevronRight, Crosshair, OctagonAlert, TriangleAlert, Users } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { PATIENT_STATE_TEXT, STATUS_TEXT } from '../lib/format';
import { glossKey } from '../lib/glossary';
import type { History, PatientSummary } from '../types';
import Sparkline from './Sparkline';
import { Tip } from './Tooltip';
import VitalReadout from './VitalReadout';

function useBedHistory(id: string, live: boolean) {
  const [h, setH] = useState<History | null>(null);
  useEffect(() => {
    if (!live) return;
    let alive = true;
    const run = () => api.history(id, 30, 180).then((d) => alive && setH(d)).catch(() => undefined);
    void run();
    const t = window.setInterval(run, 10000);
    return () => { alive = false; window.clearInterval(t); };
  }, [id, live]);
  return h;
}

const LEVEL_WORD = { green: 'Routine', amber: 'Watch', red: 'Urgent', unknown: '—' } as const;

export default function BedStrip({ p }: { p: PatientSummary }) {
  const alarm = p.top_alarm && p.top_alarm.state === 'active' ? p.top_alarm : null;
  const acked = p.top_alarm && p.top_alarm.state === 'acknowledged' ? p.top_alarm : null;
  const red = alarm?.severity === 'red';
  const running = p.status === 'running';
  const hist = useBedHistory(p.patient_id, running);
  const level = p.ews.level;
  const edge = alarm ? (red ? 'var(--red)' : 'var(--amber)')
    : level === 'red' ? 'var(--red-line)' : level === 'amber' ? 'var(--amber-line)' : undefined;
  const href = `/patients/${encodeURIComponent(p.patient_id)}`;

  return (
    <div className={`group stretch-card card relative overflow-hidden transition-[background-color,border-color] duration-200 hover:bg-surface2 hover:border-linestrong ${
           alarm && !alarm.suppressed ? (red ? 'alarm-flash-high' : 'alarm-flash-medium') : ''}`}
         style={{ borderColor: edge,
                  ['--pulse-color' as string]: red ? 'color-mix(in srgb, var(--red) 40%, transparent)'
                    : 'color-mix(in srgb, var(--amber) 35%, transparent)' }}>
      {alarm && (
        <div className={`flex items-center px-4 py-2 border-b ${red ? 'band-red' : 'band-amber'}`}>
          <Tip term={red ? 'alarm_red' : 'alarm_amber'} content={alarm.message} className="flex items-center gap-2 min-w-0">
            {red ? <OctagonAlert className="h-4 w-4 shrink-0" aria-hidden /> : <TriangleAlert className="h-4 w-4 shrink-0" aria-hidden />}
            <span className="t-label t-strong truncate">{alarm.title}</span>
          </Tip>
        </div>
      )}

      <div className="grid grid-cols-2 gap-x-6 gap-y-4 p-4 xl:grid-cols-[minmax(150px,200px)_minmax(0,1fr)_minmax(0,1fr)_minmax(200px,1.1fr)_auto] xl:items-center">
        {/* who / where */}
        <div className="flex flex-col gap-1 min-w-0 col-span-2 xl:col-span-1">
          <div className="flex items-center gap-2">
            <Link to={href} className="stretch-link t-title rounded-md"
                  aria-label={`${p.label}. ${alarm ? `Alarm: ${alarm.title}.` : ''} Open bed.`}>{p.label}</Link>
            {p.recording && (
              <Tip term="recording" className="chip">
                <span className="h-2 w-2 rounded-full bg-red rec-dot" aria-hidden />Rec
              </Tip>
            )}
          </div>
          <span className="flex">
            <Tip className="t-label text-ink2"
                 term={running ? glossKey('state_', p.fall.patient_state, 'state_UNKNOWN') : glossKey('status_', p.status, 'status_error')}>
              {running ? PATIENT_STATE_TEXT[p.fall.patient_state ?? 'UNKNOWN'] : STATUS_TEXT[p.status]}
            </Tip>
          </span>
          {running && (p.scene.staff_count ?? 0) > 0 && (
            <span className="flex">
              <Tip term="staff" className="t-micro text-mute inline-flex items-center gap-1">
                <Users className="h-3.5 w-3.5" aria-hidden />{p.scene.staff_count} with patient
              </Tip>
            </span>
          )}
        </div>

        {running ? (
          <>
            {(['hr', 'rr'] as const).map((k) => (
              <div key={k} className="flex flex-col gap-2 min-w-0">
                <VitalReadout kind={k} size="m" value={p[k].value} stale={p[k].stale} sqi={p[k].sqi}
                              subscore={p[k].subscore} reason={p[k].reason} apnea={k === 'rr' ? p.apnea : undefined} />
                <Tip term="sparkline" className="block w-full max-w-[220px]">
                  <Sparkline kind={k} t={hist?.t ?? []} v={hist?.[k] ?? []}
                             label={`${k === 'hr' ? 'Heart rate' : 'Respiratory rate'}, ${p.label}`} />
                </Tip>
              </div>
            ))}

            <div className="flex flex-col gap-1.5 min-w-0 col-span-2 xl:col-span-1 xl:rule-l xl:pl-6">
              <div className="flex items-baseline gap-2">
                <Tip term="ews" className="t-eyebrow text-ink2 tip-term">Early warning</Tip>
                <Tip term={glossKey('ews_', level, 'ews_unknown')} className="flex items-baseline gap-2"
                     content={p.ews.total == null ? undefined : `Now ${p.ews.total} of 6 points.`}>
                  <span className={`t-num-s tnum ${level === 'red' ? 'text-red' : level === 'amber' ? 'text-amber' : ''}`}>
                    {p.ews.total ?? '–'}
                  </span>
                  <span className={`t-label ${level === 'red' ? 'text-red' : level === 'amber' ? 'text-amber' : 'text-mute'}`}>
                    {LEVEL_WORD[level]}
                  </span>
                </Tip>
              </div>
              <p className="t-micro text-ink2 line-clamp-3">{p.ews.explanation || 'Vitals stable.'}</p>
              <div className="flex flex-wrap gap-2">
                {p.taa_flag && <Tip term="taaFlag" className="chip band-amber">Paradoxical breathing</Tip>}
                {acked && <Tip term="acked" className="chip">Alarm acknowledged</Tip>}
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-start gap-2.5 text-ink2 col-span-2 xl:col-span-3">
            {p.status === 'awaiting_calibration'
              ? <Crosshair className="h-5 w-5 shrink-0" aria-hidden />
              : <CameraOff className="h-5 w-5 shrink-0 text-mute" aria-hidden />}
            <p className="t-label line-clamp-3">
              {p.status === 'awaiting_calibration' ? 'Open the bed and mark the chair to start monitoring.'
                : p.status_detail || 'Waiting for the camera.'}
            </p>
          </div>
        )}

        <div className="flex xl:flex-col items-center xl:items-end gap-3 col-span-2 xl:col-span-1">
          {running && p.pain?.available && p.pain.probability != null && (
            <Tip term="pain" className="flex flex-col xl:items-end">
              <span className="t-eyebrow text-mute">Pain · experimental</span>
              <span className="font-cond font-semibold text-[22px] tnum leading-tight">
                {Math.round(p.pain.probability * 100)}%
              </span>
            </Tip>
          )}
          <ChevronRight className="h-5 w-5 text-mute group-hover:text-ink ml-auto xl:ml-0" aria-hidden />
        </div>
      </div>
    </div>
  );
}
