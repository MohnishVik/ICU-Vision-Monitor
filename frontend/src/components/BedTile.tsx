// BedTile.tsx — one bed on the ward view. Quiet when stable, loud only on deviation.
import { CameraOff, Crosshair, OctagonAlert, TriangleAlert, Users } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { urls } from '../lib/api';
import { PATIENT_STATE_TEXT, STATUS_TEXT } from '../lib/format';
import type { PatientSummary } from '../types';
import VitalReadout from './VitalReadout';

function Thumb({ id, live }: { id: string; live: boolean }) {
  const [n, setN] = useState(0);
  const [ok, setOk] = useState(true);
  useEffect(() => {
    if (!live) return;
    const h = window.setInterval(() => setN((x) => x + 1), 2000);
    return () => window.clearInterval(h);
  }, [live]);
  if (!live || !ok) {
    return <div className="h-[60px] w-[80px] rounded-md bg-raised grid place-items-center" aria-hidden>
      <CameraOff className="h-4 w-4 text-mute" /></div>;
  }
  return (
    <img src={`${urls.frame(id)}?n=${n}`} alt="" aria-hidden onError={() => setOk(false)}
         className="h-[60px] w-[80px] rounded-md object-cover bg-black" />
  );
}

export default function BedTile({ p }: { p: PatientSummary }) {
  const alarm = p.top_alarm && p.top_alarm.state === 'active' ? p.top_alarm : null;
  const acked = p.top_alarm && p.top_alarm.state === 'acknowledged' ? p.top_alarm : null;
  const red = alarm?.severity === 'red';
  const running = p.status === 'running';
  const level = p.ews.level;
  const border = alarm ? (red ? 'border-red' : 'border-amber')
    : level === 'red' ? 'border-[#7a2630]' : level === 'amber' ? 'border-[#6b4419]' : 'border-hairline';

  return (
    <Link
      to={`/patients/${encodeURIComponent(p.patient_id)}`}
      className={`group flex flex-col rounded-2xl border bg-panel overflow-hidden transition-colors hover:bg-[#182a30] ${border} ${
        alarm && !alarm.suppressed ? 'alarm-pulse' : ''}`}
      style={{ ['--pulse-color' as string]: red ? 'rgba(255,77,94,.35)' : 'rgba(255,155,61,.3)' }}
      aria-label={`${p.label}. ${alarm ? `Alarm: ${alarm.title}.` : ''} Open bed.`}
    >
      {alarm && (
        <div className={`flex items-center gap-2 px-4 py-2 ${red ? 'bg-[#3d141a] text-[#ffd0d4]' : 'bg-[#3a2611] text-[#ffd9b0]'}`}>
          {red ? <OctagonAlert className="h-4 w-4 shrink-0" aria-hidden /> : <TriangleAlert className="h-4 w-4 shrink-0" aria-hidden />}
          <span className="t-label truncate">{alarm.title}</span>
        </div>
      )}

      <div className="flex items-start gap-3 px-4 pt-4">
        <div className="flex flex-col min-w-0 flex-1">
          <span className="t-title text-ink">{p.label}</span>
          <span className="t-label text-ink2 truncate">
            {running ? PATIENT_STATE_TEXT[p.fall.patient_state ?? 'UNKNOWN'] : STATUS_TEXT[p.status]}
            {running && (p.scene.staff_count ?? 0) > 0 && (
              <span className="inline-flex items-center gap-1 ml-2 text-mute">
                <Users className="h-3.5 w-3.5" aria-hidden />{p.scene.staff_count} with patient
              </span>
            )}
          </span>
        </div>
        <Thumb id={p.patient_id} live={p.status !== 'camera_offline' && p.status !== 'starting'} />
      </div>

      {running ? (
        <div className="grid grid-cols-2 gap-4 px-4 py-4">
          <VitalReadout kind="hr" size="m" value={p.hr.value} stale={p.hr.stale} sqi={p.hr.sqi}
                        subscore={p.hr.subscore} reason={p.hr.reason} />
          <VitalReadout kind="rr" size="m" value={p.rr.value} stale={p.rr.stale} sqi={p.rr.sqi}
                        subscore={p.rr.subscore} reason={p.rr.reason} apnea={p.apnea} />
        </div>
      ) : (
        <div className="px-4 py-5 flex items-start gap-2.5 text-ink2">
          {p.status === 'awaiting_calibration'
            ? <Crosshair className="h-5 w-5 shrink-0 text-scrub" aria-hidden />
            : <CameraOff className="h-5 w-5 shrink-0 text-mute" aria-hidden />}
          <p className="t-label line-clamp-3">
            {p.status === 'awaiting_calibration' ? 'Open the bed and mark the chair to start monitoring.'
              : p.status_detail || 'Waiting for the camera.'}
          </p>
        </div>
      )}

      {running && (
        <div className="mt-auto rule-t px-4 py-3 flex flex-col gap-1">
          <div className="flex items-baseline gap-2">
            <span className="t-label text-ink2">Early warning</span>
            <span className={`t-label tnum font-semibold ${level === 'red' ? 'text-[#ffb3ba]'
              : level === 'amber' ? 'text-[#ffc488]' : 'text-ink'}`}>
              {p.ews.total ?? '--'}
            </span>
            {p.taa_flag && <span className="t-micro text-[#ffc488] ml-auto">Paradoxical breathing</span>}
            {acked && <span className="t-micro text-mute ml-auto">Alarm acknowledged</span>}
          </div>
          <p className="t-micro text-mute line-clamp-2 min-h-[2.7em]">{p.ews.explanation || 'Vitals stable.'}</p>
        </div>
      )}
    </Link>
  );
}
