// AlarmNotification.tsx — ward-wide alarm bar, always at the top of the screen.
//
// Shows every active, unacknowledged alarm across beds. Escalation follows the
// server's alarm FSM (log -> dashboard -> chime -> phone): the band pulses from
// "dashboard_highlight", and a tone plays from "chime" when sound is enabled.
// Alarms logged while staff are at the bedside are shown without sound.
import { BellOff, OctagonAlert, TriangleAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../lib/api';
import { duration } from '../lib/format';
import { playAlarm } from '../lib/sound';
import type { Alarm, PatientSummary } from '../types';

const STAGE_RANK: Record<string, number> = { log_only: 0, dashboard_highlight: 1, chime: 2, phone_push: 3 };

interface Row { patient: PatientSummary; alarm: Alarm }

export default function AlarmNotification({ ward, soundOn }: { ward: PatientSummary[] | null; soundOn: boolean }) {
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sounded = useRef<Map<number, number>>(new Map());

  const rows: Row[] = (ward ?? [])
    .filter((p) => p.top_alarm && p.top_alarm.state === 'active')
    .map((p) => ({ patient: p, alarm: p.top_alarm as Alarm }))
    .sort((a, b) => (a.alarm.severity === b.alarm.severity ? a.alarm.first_seen - b.alarm.first_seen
      : a.alarm.severity === 'red' ? -1 : 1));

  useEffect(() => {
    if (!soundOn) return;
    for (const { alarm } of rows) {
      const rank = STAGE_RANK[alarm.action] ?? 0;
      if (rank < 2 || alarm.suppressed) continue;
      const last = sounded.current.get(alarm.id) ?? -1;
      if (rank > last) {
        sounded.current.set(alarm.id, rank);
        playAlarm(alarm.severity);
      }
    }
  });

  if (rows.length === 0) return null;
  const now = Date.now() / 1000;

  const ack = async (pid: string, id: number) => {
    setBusy(id);
    setError(null);
    try {
      await api.ack(pid, id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section aria-label="Active alarms" role="alert" className="px-4 sm:px-6 pt-3 flex flex-col gap-2">
      {rows.map(({ patient, alarm }) => {
        const red = alarm.severity === 'red';
        const Icon = red ? OctagonAlert : TriangleAlert;
        const pulsing = (STAGE_RANK[alarm.action] ?? 0) >= 1 && !alarm.suppressed;
        return (
          <div
            key={`${patient.patient_id}-${alarm.id}`}
            className={`flex flex-col md:flex-row md:items-center gap-x-4 gap-y-3 rounded-xl border px-4 py-3 ${
              red ? 'bg-[#3d141a] border-[#7a2630]' : 'bg-[#3a2611] border-[#6b4419]'
            } ${pulsing ? 'alarm-pulse' : ''}`}
            style={{ ['--pulse-color' as string]: red ? 'rgba(255,77,94,.35)' : 'rgba(255,155,61,.3)' }}
          >
            <div className="flex items-start gap-3 min-w-0 flex-1">
              <Icon className={`h-6 w-6 shrink-0 mt-0.5 ${red ? 'text-red' : 'text-amber'}`} aria-hidden />
              <div className="flex flex-col min-w-0">
                <span className="t-head text-ink">
                  {patient.label}: {alarm.title}
                  {patient.simulated && <span className="t-micro text-[#d8ccff] ml-2">simulated</span>}
                </span>
                <span className="t-label text-ink2 line-clamp-2">{alarm.message}</span>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pl-9 md:pl-0">
            <span className="t-micro text-ink2 tnum mr-auto md:mr-0">
              {alarm.suppressed ? (
                <span className="inline-flex items-center gap-1"><BellOff className="h-3.5 w-3.5" aria-hidden />
                  Staff at bedside, logged only</span>
              ) : (
                `for ${duration(now - (alarm.active_since ?? alarm.first_seen))}`
              )}
            </span>
            <div className="flex gap-2">
              <Link to={`/patients/${encodeURIComponent(patient.patient_id)}`} className="btn btn-ghost">
                Open bed
              </Link>
              <button
                type="button"
                className={red ? 'btn btn-danger' : 'btn btn-primary'}
                disabled={busy === alarm.id}
                onClick={() => void ack(patient.patient_id, alarm.id)}
              >
                {busy === alarm.id ? 'Acknowledging' : 'Acknowledge'}
              </button>
            </div>
            </div>
          </div>
        );
      })}
      {error && <p className="t-label text-red">Could not acknowledge: {error}</p>}
    </section>
  );
}
