// Dashboard.tsx — ICU grid: one patient card per bed zone.
// The headline is a sentence built from live data: what needs a nurse right now.
// Beds keep a fixed order (spatial memory), they never re-sort by severity.
import BedTile from '../components/BedTile';
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

  // The alarm bar above carries the details; the headline only says where to look.
  if (red.length === 1) return { text: `${red[0].label} needs attention now.`, tone: 'red' };
  if (red.length > 1) return { text: `${red.length} beds need attention now.`, tone: 'red' };
  if (amber.length === 1) return { text: `Watch ${amber[0].label} closely.`, tone: 'amber' };
  if (amber.length > 1) return { text: `${amber.length} beds to watch closely.`, tone: 'amber' };
  if (offline.length === ward.length) return { text: ward.length === 1 ? 'The camera is not connected.' : 'No cameras connected.', tone: 'calm' };
  if (setup.length) return { text: setup.length === 1 ? `${setup[0].label} is waiting for chair setup.` : `${setup.length} beds are waiting for chair setup.`, tone: 'calm' };
  if (watch.length) return { text: `${watch.map((p) => p.label).join(' and ')} trending up. No alarms.`, tone: 'calm' };
  return { text: ward.length === 1 ? `${ward[0].label} stable.` : `All ${ward.length} beds stable.`, tone: 'calm' };
}

export default function Dashboard() {
  const { ward, error } = useWard();

  if (!ward) {
    return (
      <div className="py-16">
        <p className="font-cond font-semibold text-[48px] leading-none text-ink2">Connecting to the ward</p>
        {error && <p className="t-body text-ink2 mt-4 max-w-[60ch]">{error}</p>}
      </div>
    );
  }

  const h = headline(ward);
  const monitoring = ward.filter((p) => p.status === 'running').length;
  const acked = ward.filter((p) => p.top_alarm?.state === 'acknowledged').length;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2 pt-2">
        <h1 className={`font-cond font-semibold text-[40px] sm:text-[56px] leading-[0.95] tracking-[-0.02em] max-w-[24ch] ${
          h.tone === 'red' ? 'text-[#ffb3ba]' : h.tone === 'amber' ? 'text-[#ffc488]' : 'text-ink'}`}>
          {h.text}
        </h1>
        <p className="t-lead text-ink2">
          {monitoring} of {ward.length} {ward.length === 1 ? 'bed' : 'beds'} monitoring
          {acked > 0 && `, ${acked} acknowledged ${acked === 1 ? 'alarm' : 'alarms'} still open`}
        </p>
      </header>

      <div className={`grid gap-4 ${ward.length === 1 ? 'max-w-[460px]' : 'grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4'}`}>
        {ward.map((p) => <BedTile key={p.patient_id} p={p} />)}
      </div>
    </div>
  );
}
