// PatientDetail.tsx — expanded view: all panels + full trend history
import { ArrowLeft, Bell, Crosshair, History as HistoryIcon, Thermometer, ThermometerSnowflake } from 'lucide-react';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import EWSIndicator from '../components/EWSIndicator';
import FallDetectorPanel from '../components/FallDetectorPanel';
import ModuleHealth from '../components/ModuleHealth';
import PainPanel from '../components/PainPanel';
import StatusBadge from '../components/StatusBadge';
import StreamPanel from '../components/StreamPanel';
import TAAPanel from '../components/TAAPanel';
import ThermalPanel from '../components/ThermalPanel';
import VitalReadout from '../components/VitalReadout';
import VitalsTrendChart from '../components/VitalsTrendChart';
import { usePatientHistory } from '../hooks/usePatientData';
import { usePatientStream } from '../hooks/useWebSocket';
import { api } from '../lib/api';
import { ago, clock, duration, STATUS_TEXT } from '../lib/format';
import type { Alarm, Measurement, PatientSnapshot } from '../types';

const WINDOWS = [
  { m: 5, label: '5 min' }, { m: 15, label: '15 min' }, { m: 60, label: '1 hour' }, { m: 240, label: '4 hours' },
];

function AlarmsPanel({ snap }: { snap: PatientSnapshot }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const open = snap.alarms ?? [];
  const now = Date.now() / 1000;
  const ack = async (id: number | null) => {
    setBusy(true);
    setErr(null);
    try { await api.ack(snap.patient_id, id); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const row = (a: Alarm, past = false) => (
    <li key={a.id} className="flex flex-wrap items-start gap-x-3 gap-y-1 py-3 rule-b last:border-b-0">
      <span className={`mt-1.5 h-2.5 w-2.5 rounded-full shrink-0 ${a.severity === 'red' ? 'bg-red' : 'bg-amber'}`} aria-hidden />
      <div className="flex flex-col min-w-0 flex-1">
        <span className="t-label text-ink">
          {a.title}
          <span className="text-mute ml-2">{a.severity === 'red' ? 'High priority' : 'Medium priority'}</span>
        </span>
        <span className="t-micro text-ink2 max-w-[70ch]">{a.message}</span>
        <span className="t-micro text-mute tnum">
          {past ? `Cleared ${ago(a.cleared_at)}, lasted ${duration((a.cleared_at ?? now) - a.first_seen)}`
            : a.state === 'acknowledged' ? `Acknowledged ${ago(a.acked_at)}`
              : `Started ${clock(a.first_seen)}${a.suppressed ? ', staff at bedside' : `, escalation: ${a.action.replace('_', ' ')}`}`}
        </span>
      </div>
      {!past && a.state === 'active' && (
        <button type="button" className={a.severity === 'red' ? 'btn btn-danger' : 'btn btn-primary'}
                disabled={busy} onClick={() => void ack(a.id)}>Acknowledge</button>
      )}
    </li>
  );
  return (
    <section className="panel p-5" aria-labelledby="alarms-h">
      <div className="flex items-center gap-2">
        <Bell className="h-4 w-4 text-ink2" aria-hidden />
        <h2 id="alarms-h" className="t-head">Alarms</h2>
        <span className="t-micro text-mute ml-auto tnum">
          {snap.metrics?.actionable_alerts_shift ?? 0} sounded this shift (target 3 or fewer)
        </span>
      </div>
      {open.length === 0 ? (
        <p className="t-label text-ink2 mt-3">No open alarms.</p>
      ) : <ul className="mt-1">{open.map((a) => row(a))}</ul>}
      {snap.alarm_history?.length > 0 && (
        <details className="mt-3 group">
          <summary className="t-label text-ink2 cursor-pointer inline-flex items-center gap-2 select-none">
            <HistoryIcon className="h-4 w-4" aria-hidden /> Earlier alarms ({snap.alarm_history.length})
          </summary>
          <ul className="mt-1">{snap.alarm_history.map((a) => row(a, true))}</ul>
        </details>
      )}
      {err && <p className="t-label text-red mt-2">Could not acknowledge: {err}</p>}
    </section>
  );
}

function SourcesPanel({ snap }: { snap: PatientSnapshot }) {
  const rows: { name: string; m: Measurement | null; weight?: number }[] = [];
  for (const k of ['rr', 'hr'] as const) {
    const v = snap.vitals?.[k];
    if (!v) continue;
    for (const [src, m] of Object.entries(v.sources ?? {})) rows.push({ name: src, m });
  }
  const label: Record<string, string> = {
    'rr.depth': 'RR from chest depth (D435)', 'rr.thermal': 'RR from nostril thermal (ESP32)',
    'hr.pos': 'HR from skin colour (POS rPPG)',
  };
  return (
    <section className="panel p-5" aria-labelledby="src-h">
      <h2 id="src-h" className="t-head">How each reading was measured</h2>
      <p className="t-label text-ink2 mt-1 max-w-[64ch]">
        Each sensor path reports its own estimate and signal quality. Paths are combined by
        inverse-variance weighting; if they disagree, no number is shown.
      </p>
      <table className="w-full mt-3 t-label tnum">
        <thead className="text-mute">
          <tr><th className="text-left font-medium py-1.5">Path</th><th className="text-right font-medium">Estimate</th>
            <th className="text-right font-medium">Quality</th><th className="text-right font-medium">Compute</th></tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr className="rule-t"><td colSpan={4} className="py-3 text-ink2">
              No sensor readings yet. They appear once the camera is running and the chair is set up.
            </td></tr>
          )}
          {rows.map(({ name, m }) => (
            <tr key={name} className="rule-t">
              <td className="py-2 text-ink">
                {name === 'rr.thermal' && m?.meta?.roi === 'warmest_patch'
                  ? 'RR from thermal, warmest spot (not aligned to the nose)'
                  : label[name] ?? name}
              </td>
              <td className="text-right">{m?.value == null ? <span className="text-mute">none</span> : Math.round(m.value)}</td>
              <td className="text-right">{m ? `${Math.round(m.sqi * 100)}%` : <span className="text-mute">offline</span>}</td>
              <td className="text-right text-ink2">{m ? `${m.latency_ms.toFixed(1)} ms` : '--'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default function PatientDetail() {
  const { id } = useParams();
  const { data: snap, link } = usePatientStream(id);
  const [minutes, setMinutes] = useState(15);
  const { data: history } = usePatientHistory(id, minutes);

  if (!snap) {
    return (
      <div className="py-16 flex flex-col gap-4">
        <Link to="/" className="btn btn-ghost w-fit"><ArrowLeft className="h-4 w-4" aria-hidden />Ward</Link>
        <p className="font-cond font-semibold text-[48px] leading-none text-ink2">
          {link === 'reconnecting' ? `Cannot reach ${id}` : 'Opening bed'}
        </p>
      </div>
    );
  }

  const v = snap.vitals;
  const running = snap.status === 'running';
  // When the engine isn't running, say why, instead of a generic "collecting signal".
  const why = running ? null : snap.status === 'error' ? 'camera_offline' : snap.status;
  const reasonFor = (r: string | null | undefined) => why ?? r ?? 'warming_up';

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Link to="/" className="btn btn-ghost px-2" aria-label="Back to ward"><ArrowLeft className="h-4 w-4" aria-hidden /></Link>
        <h1 className="font-cond font-semibold text-[40px] leading-none tracking-[-0.015em]">{snap.label}</h1>
        <div className="flex flex-wrap gap-2">
          <StatusBadge tone={running ? 'neutral' : snap.status === 'awaiting_calibration' ? 'info' : 'amber'}>
            {STATUS_TEXT[snap.status]}
          </StatusBadge>
          {snap.mode === 'camera' && running && (
            <StatusBadge icon={snap.scene?.thermal_online ? <Thermometer /> : <ThermometerSnowflake />}>
              {snap.scene?.thermal_online ? 'Thermal camera on' : 'Thermal camera off'}
            </StatusBadge>
          )}
          {snap.simulated && <StatusBadge tone="demo">Simulated patient</StatusBadge>}
          {link !== 'live' && <StatusBadge tone="amber">Not live, reconnecting</StatusBadge>}
        </div>
        <Link to={`/patients/${encodeURIComponent(snap.patient_id)}/calibrate`} className="btn ml-auto">
          <Crosshair className="h-4 w-4" aria-hidden />{snap.scene?.calibrated ? 'Adjust chair' : 'Set up chair'}
        </Link>
      </header>

      <div className="grid gap-5 lg:grid-cols-12">
        <div className="lg:col-span-7 flex flex-col gap-5 min-w-0">
          <StreamPanel snap={snap} />
          <ThermalPanel snap={snap} />
        </div>
        <div className="lg:col-span-5 flex flex-col gap-5 min-w-0">
          <section className="panel p-5 grid gap-6 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2" aria-label="Vital signs">
            <VitalReadout kind="hr" value={v?.hr.value ?? null} stale={v?.hr.stale ?? true} sqi={v?.hr.sqi ?? 0}
                          subscore={v?.hr.subscore ?? null} reason={reasonFor(v?.hr.reason)} ageS={v?.hr.age_s} />
            <VitalReadout kind="rr" value={v?.rr.value ?? null} stale={v?.rr.stale ?? true} sqi={v?.rr.sqi ?? 0}
                          subscore={v?.rr.subscore ?? null} reason={reasonFor(v?.rr.reason)}
                          apnea={v?.rr.apnea} ageS={v?.rr.age_s} />
          </section>
          <section className="panel p-5" aria-labelledby="ews-h">
            <h2 id="ews-h" className="t-head mb-3">Early warning</h2>
            <EWSIndicator ews={snap.ews} />
          </section>
          <AlarmsPanel snap={snap} />
        </div>
      </div>

      <section className="flex flex-col gap-4" aria-labelledby="trends-h">
        <div className="flex flex-wrap items-center gap-3">
          <h2 id="trends-h" className="t-title">Trends</h2>
          <div role="radiogroup" aria-label="Trend window" className="inline-flex rounded-lg border border-hairline p-0.5 ml-auto">
            {WINDOWS.map((w) => (
              <button key={w.m} type="button" role="radio" aria-checked={minutes === w.m}
                      onClick={() => setMinutes(w.m)}
                      className={`h-8 px-3 rounded-md t-label transition-colors ${
                        minutes === w.m ? 'bg-raised text-ink' : 'text-ink2 hover:text-ink'}`}>
                {w.label}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-5 xl:grid-cols-2">
          <VitalsTrendChart kind="hr" history={history} minutes={minutes} />
          <VitalsTrendChart kind="rr" history={history} minutes={minutes} />
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-3">
        <FallDetectorPanel fall={snap.fall} scene={snap.scene} history={history} />
        <TAAPanel taa={snap.taa} />
        <PainPanel pain={snap.pain} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <SourcesPanel snap={snap} />
        <section className="panel p-5" aria-labelledby="mods-h">
          <h2 id="mods-h" className="t-head mb-1">Pipeline status</h2>
          <ModuleHealth modules={snap.modules} />
        </section>
      </div>
    </div>
  );
}
