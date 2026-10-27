// AlarmBar.tsx — ward-wide alarm bar, under the header on every page.
//
// Shows every active, unacknowledged alarm across beds, red first. Escalation follows the
// server's alarm FSM (log -> dashboard -> chime -> phone): the band flashes from
// "dashboard_highlight" (red at 2 Hz, orange at 0.5 Hz), a tone plays from "chime" when sound
// is on. Alarms logged while staff are at the bedside are shown without sound. Red alarms are
// always shown in full; past two rows the remaining orange ones fold behind a count so the
// page stays usable.
import { BellOff, ChevronDown, OctagonAlert, TriangleAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { clockSec, duration, STAGE_TEXT } from '../lib/format';
import { playAlarm } from '../lib/sound';
import type { Alarm, PatientSummary } from '../types';
import { Tip } from './Tooltip';

const STAGE_RANK: Record<string, number> = { log_only: 0, dashboard_highlight: 1, chime: 2, phone_push: 3 };

export default function AlarmBar({ ward, soundOn }: { ward: PatientSummary[] | null; soundOn: boolean }) {
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const sounded = useRef<Map<number, number>>(new Map());

  const rows = (ward ?? [])
    .filter((p) => p.top_alarm && p.top_alarm.state === 'active')
    .map((p) => ({ patient: p, alarm: p.top_alarm as Alarm }))
    .sort((a, b) => (a.alarm.severity === b.alarm.severity ? a.alarm.first_seen - b.alarm.first_seen
      : a.alarm.severity === 'red' ? -1 : 1));

  useEffect(() => {
    if (!soundOn) return;
    for (const { alarm } of rows) {
      const rank = STAGE_RANK[alarm.action] ?? 0;
      if (rank < 2 || alarm.suppressed) continue;
      if (rank > (sounded.current.get(alarm.id) ?? -1)) {
        sounded.current.set(alarm.id, rank);
        playAlarm(alarm.severity);
      }
    }
  });

  if (rows.length === 0) return null;
  const now = Date.now() / 1000;
  const reds = rows.filter((r) => r.alarm.severity === 'red').length;
  const keep = Math.max(reds, 2);
  const folded = expanded ? 0 : Math.max(rows.length - keep, 0);
  const shown = folded ? rows.slice(0, keep) : rows;

  const ack = async (pid: string, id: number) => {
    setBusy(id);
    setError(null);
    try { await api.ack(pid, id); } catch (e) { setError((e as Error).message); } finally { setBusy(null); }
  };

  return (
    <section aria-label="Active alarms" role="alert" className="px-4 sm:px-6 pt-3 flex flex-col gap-2">
      {shown.map(({ patient, alarm }) => {
        const red = alarm.severity === 'red';
        const Icon = red ? OctagonAlert : TriangleAlert;
        const pulsing = (STAGE_RANK[alarm.action] ?? 0) >= 1 && !alarm.suppressed;
        return (
          <div key={`${patient.patient_id}-${alarm.id}`}
               className={`flex flex-col md:flex-row md:items-center gap-x-5 gap-y-3 rounded-lg border px-4 py-3 ${
                 red ? 'band-red' : 'band-amber'} ${pulsing ? (red ? 'alarm-flash-high' : 'alarm-flash-medium') : ''}`}
               style={{ ['--pulse-color' as string]: red ? 'color-mix(in srgb, var(--red) 40%, transparent)'
                 : 'color-mix(in srgb, var(--amber) 35%, transparent)' }}>
            <div className="flex items-start gap-3 min-w-0 flex-1">
              <Icon className={`h-6 w-6 shrink-0 mt-0.5 ${red ? 'text-red' : 'text-amber'}`} aria-hidden />
              <div className="flex flex-col min-w-0">
                <span className="t-head">
                  <Tip term={red ? 'alarm_red' : 'alarm_amber'} className="t-eyebrow mr-2 align-middle tip-term">
                    {red ? 'High priority' : 'Medium'}
                  </Tip>
                  {patient.label}: {alarm.title}
                  {patient.simulated && <span className="t-micro opacity-80 ml-2">(simulated)</span>}
                </span>
                <span className="t-label opacity-90 line-clamp-2">{alarm.message}</span>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pl-9 md:pl-0">
              {alarm.suppressed ? (
                <Tip term="staffSuppress" className="t-micro tnum opacity-90 mr-auto md:mr-0 inline-flex items-center gap-1">
                  <BellOff className="h-3.5 w-3.5" aria-hidden />Staff at bedside, logged only
                </Tip>
              ) : (
                <Tip term="escalation" className="t-micro tnum opacity-90 mr-auto md:mr-0"
                     content={`Now: ${STAGE_TEXT[alarm.action] ?? alarm.action}. Active since ${clockSec(alarm.active_since ?? alarm.first_seen)}.`}>
                  since {clockSec(alarm.active_since ?? alarm.first_seen)} · {duration(now - (alarm.active_since ?? alarm.first_seen))}
                </Tip>
              )}
              <Link to={`/patients/${encodeURIComponent(patient.patient_id)}`} className="btn btn-sm">Open bed</Link>
              <Tip term="ack" asChild>
                <button type="button" className={`btn btn-sm ${red ? 'btn-red' : 'btn-amber'}`}
                        disabled={busy === alarm.id} onClick={() => void ack(patient.patient_id, alarm.id)}>
                  {busy === alarm.id ? 'Acknowledging' : 'Acknowledge'}
                </button>
              </Tip>
            </div>
          </div>
        );
      })}
      {(folded > 0 || (expanded && rows.length > keep)) && (
        <button type="button" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}
                className="hit self-start inline-flex items-center gap-2 rounded-lg border band-amber px-3 min-h-[36px] t-label">
          <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} aria-hidden />
          {expanded ? 'Show fewer' : `${folded} more orange`}
        </button>
      )}
      {error && <p className="t-label text-red">Could not acknowledge: {error}</p>}
    </section>
  );
}
