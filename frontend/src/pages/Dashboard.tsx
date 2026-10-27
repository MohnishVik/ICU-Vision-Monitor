// Dashboard.tsx — the ward: one monitor strip per bed, in a fixed order (spatial memory),
// under a headline sentence built from live data that says where to look.
import BedStrip from '../components/BedStrip';
import { useWard } from '../context/WardContext';
import type { PatientSummary } from '../types';

function headline(ward: PatientSummary[]): { text: string; tone: 'calm' | 'amber' | 'red' } {
  const active = ward.filter((p) => p.top_alarm?.state === 'active');
  const red = active.filter((p) => p.top_alarm?.severity === 'red');
  const amber = active.filter((p) => p.top_alarm?.severity === 'amber');
  const setup = ward.filter((p) => p.status === 'awaiting_calibration');
  const offline = ward.filter((p) => p.status === 'camera_offline' || p.status === 'error');
  const watch = ward.filter((p) => p.status === 'running' && !p.top_alarm
    && (p.ews.level === 'amber' || p.ews.level === 'red'));

  if (red.length === 1) return { text: `${red[0].label} needs attention now.`, tone: 'red' };
  if (red.length > 1) return { text: `${red.length} beds need attention now.`, tone: 'red' };
  if (amber.length === 1) return { text: `Watch ${amber[0].label} closely.`, tone: 'amber' };
  if (amber.length > 1) return { text: `${amber.length} beds to watch closely.`, tone: 'amber' };
  if (offline.length === ward.length) return { text: ward.length === 1 ? 'The camera is not connected.' : 'No cameras connected.', tone: 'calm' };
  if (setup.length) return { text: setup.length === 1 ? `${setup[0].label} is waiting for chair setup.` : `${setup.length} beds are waiting for chair setup.`, tone: 'calm' };
  if (watch.length) return { text: `${watch.map((p) => p.label).join(' and ')} trending up. No alarms.`, tone: 'calm' };
  return { text: ward.length === 1 ? `${ward[0].label} is stable.` : `All ${ward.length} beds stable.`, tone: 'calm' };
}

export default function Dashboard() {
  const { ward, error } = useWard();

  if (!ward) {
    return (
      <div className="py-16 flex flex-col gap-3">
        <p className="t-display text-ink2">Connecting to the ward</p>
        {error && <p className="t-body text-ink2 max-w-[60ch]">{error}</p>}
      </div>
    );
  }

  const h = headline(ward);
  const monitoring = ward.filter((p) => p.status === 'running').length;
  const acked = ward.filter((p) => p.top_alarm?.state === 'acknowledged').length;
  const recording = ward.filter((p) => p.recording).length;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <h1 className={`t-display max-w-[26ch] ${h.tone === 'red' ? 'text-red' : h.tone === 'amber' ? 'text-amber' : ''}`}>
          {h.text}
        </h1>
        <p className="t-body text-ink2">
          {monitoring} of {ward.length} {ward.length === 1 ? 'bed' : 'beds'} monitoring
          {acked > 0 && ` · ${acked} acknowledged ${acked === 1 ? 'alarm' : 'alarms'} still open`}
          {recording > 0 && ` · ${recording} recording a comparison session`}
        </p>
        <p className="t-label text-mute max-w-[80ch]">
          One strip per bed, always in the same order: heart and breathing rate with their last 30 minutes,
          and the early-warning score; select a bed to open it.
        </p>
      </header>
      <div className="stagger flex flex-col gap-3">
        {ward.map((p) => <BedStrip key={p.patient_id} p={p} />)}
      </div>
    </div>
  );
}
