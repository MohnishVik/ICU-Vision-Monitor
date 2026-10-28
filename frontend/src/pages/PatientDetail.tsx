// PatientDetail.tsx — one bed: live view, vitals, every estimator, trends, recording, status.
import { ArrowLeft, Bell, Camera, Crosshair, History as HistoryIcon, Thermometer, ThermometerSnowflake } from 'lucide-react';
import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import EWSIndicator from '../components/EWSIndicator';
import FallDetectorPanel from '../components/FallDetectorPanel';
import ModuleHealth from '../components/ModuleHealth';
import PainPanel from '../components/PainPanel';
import SessionControl from '../components/SessionControl';
import SourcesPanel from '../components/SourcesPanel';
import StreamPanel from '../components/StreamPanel';
import TAAPanel from '../components/TAAPanel';
import ThermalPanel from '../components/ThermalPanel';
import { InfoTip, Tip } from '../components/Tooltip';
import { Chip, SectionHead } from '../components/ui';
import VitalReadout from '../components/VitalReadout';
import VitalsTrendChart from '../components/VitalsTrendChart';
import { usePatientHistory } from '../hooks/usePatientData';
import { usePatientStream } from '../hooks/useWebSocket';
import { api } from '../lib/api';
import { ago, clockSec, duration, STAGE_TEXT, STATUS_TEXT } from '../lib/format';
import { glossKey } from '../lib/glossary';
import type { Alarm, PatientSnapshot } from '../types';

const WINDOWS = [{ m: 5, label: '5 min' }, { m: 15, label: '15 min' }, { m: 60, label: '1 h' }, { m: 240, label: '4 h' }];

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
        <span className="t-label t-strong">{a.title}
          <Tip term={a.severity === 'red' ? 'alarm_red' : 'alarm_amber'} className="t-micro text-mute ml-2 font-normal tip-term">
            {a.severity === 'red' ? 'High priority' : 'Medium priority'}
          </Tip>
        </span>
        <span className="t-micro text-ink2 max-w-[70ch]">{a.message}</span>
        <span className="flex">
          {past ? (
            <Tip term="alarmTimes" className="t-micro text-mute tnum">
              Started {clockSec(a.first_seen)}, cleared {clockSec(a.cleared_at)} ({ago(a.cleared_at)}), lasted {duration((a.cleared_at ?? now) - a.first_seen)}
            </Tip>
          ) : a.state === 'acknowledged' ? (
            <Tip term="acked" className="t-micro text-mute tnum">
              Started {clockSec(a.first_seen)}, acknowledged {clockSec(a.acked_at)} ({ago(a.acked_at)})
            </Tip>
          ) : a.suppressed ? (
            <Tip term="staffSuppress" className="t-micro text-mute tnum">Started {clockSec(a.first_seen)}, staff at bedside</Tip>
          ) : (
            <Tip term="escalation" className="t-micro text-mute tnum"
                 content={`Started ${clockSec(a.first_seen)} (24-hour clock). Now: ${STAGE_TEXT[a.action] ?? a.action}.`}>
              Started {clockSec(a.first_seen)}, escalation: {STAGE_TEXT[a.action] ?? a.action}
            </Tip>
          )}
        </span>
      </div>
      {!past && a.state === 'active' && (
        <Tip term="ack" asChild>
          <button type="button" className={`btn btn-sm ${a.severity === 'red' ? 'btn-red' : 'btn-amber'}`} disabled={busy}
                  onClick={() => void ack(a.id)}>Acknowledge</button>
        </Tip>
      )}
    </li>
  );
  return (
    <section className="card card-pad flex flex-col gap-2" aria-labelledby="alarms-h">
      <SectionHead id="alarms-h" title="Alarms" info="alarms">
        <Tip term="shiftAlerts" className="t-micro text-mute tnum">
          {snap.metrics?.actionable_alerts_shift ?? 0} sounded this shift (target ≤ 3)
        </Tip>
      </SectionHead>
      {open.length === 0 ? <p className="t-label text-ink2 inline-flex items-center gap-2"><Bell className="h-4 w-4" aria-hidden />No open alarms.</p>
        : <ul>{open.map((a) => row(a))}</ul>}
      {snap.alarm_history?.length > 0 && (
        <details>
          <summary className="t-label text-ink2 cursor-pointer inline-flex items-center gap-2 select-none min-h-[40px]">
            <HistoryIcon className="h-4 w-4" aria-hidden /> Earlier alarms ({snap.alarm_history.length})
          </summary>
          <ul>{snap.alarm_history.map((a) => row(a, true))}</ul>
        </details>
      )}
      {err && <p className="t-label text-red">Could not acknowledge: {err}</p>}
    </section>
  );
}

export default function PatientDetail() {
  const { id } = useParams();
  const { data: snap, link } = usePatientStream(id);
  const [minutes, setMinutes] = useState(15);
  const [view, setView] = useState<'camera' | 'thermal'>('camera');
  const [showSources, setShowSources] = useState(false);
  const { data: history } = usePatientHistory(id, minutes);

  if (!snap) {
    if (link === 'reconnecting') {
      return (
        <div className="py-16 flex flex-col gap-4">
          <Link to="/ward" className="btn btn-ghost w-fit"><ArrowLeft className="h-4 w-4" aria-hidden />Ward</Link>
          <p className="t-display text-ink2">Cannot reach {id}</p>
          <p className="t-body text-ink2">Check that the monitor server is running. This page reconnects by itself.</p>
        </div>
      );
    }
    return (
      <div className="flex flex-col gap-6" aria-busy="true" aria-label="Opening bed">
        <div className="skeleton h-11 w-56" />
        <div className="grid gap-5 xl:grid-cols-12">
          <div className="skeleton xl:col-span-7 aspect-[4/3]" />
          <div className="xl:col-span-5 flex flex-col gap-5">
            <div className="skeleton h-40" /><div className="skeleton h-56" />
          </div>
        </div>
      </div>
    );
  }

  const v = snap.vitals;
  const running = snap.status === 'running';
  const why = running ? null : snap.status === 'error' ? 'camera_offline' : snap.status;
  const reasonFor = (r: string | null | undefined) => why ?? r ?? 'warming_up';
  const pid = encodeURIComponent(snap.patient_id);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <Tip title="Back to the ward" content="All beds at a glance." asChild side="bottom">
          <Link to="/ward" className="btn btn-ghost px-2.5" aria-label="Back to ward"><ArrowLeft className="h-5 w-5" aria-hidden /></Link>
        </Tip>
        <h1 className="t-display">{snap.label}</h1>
        <div className="flex flex-wrap gap-2 pb-1">
          <Chip tone={running ? 'neutral' : 'amber'} term={glossKey('status_', snap.status, 'status_error')}
                tip={snap.status_detail || undefined}>{STATUS_TEXT[snap.status]}</Chip>
          {snap.mode === 'camera' && running && (
            <Chip icon={snap.scene?.thermal_online ? <Thermometer /> : <ThermometerSnowflake />}
                  term={snap.scene?.thermal_online ? 'thermalOn' : 'thermalOff'}>
              {snap.scene?.thermal_online ? 'Thermal on' : 'Thermal off'}
            </Chip>
          )}
          {snap.simulated ? <Chip tone="sim" term="demo">Simulated patient</Chip> : <Chip icon={<Camera />} term="d435">D435 camera</Chip>}
          {snap.session?.active && <Chip tone="red" term="recording" icon={<span className="block h-2 w-2 rounded-full bg-red rec-dot" />}>Recording</Chip>}
          {link !== 'live' && <Chip tone="amber" term="live">Not live, reconnecting</Chip>}
        </div>
        <div className="ml-auto flex flex-wrap gap-2 pb-1">
          <Tip term="thermalAlign" asChild side="bottom">
            <Link to={`/patients/${pid}/thermal`} className="btn"><Thermometer className="h-4 w-4" aria-hidden />Align thermal</Link>
          </Tip>
          <Tip term="chairCal" asChild side="bottom">
            <Link to={`/patients/${pid}/calibrate`} className="btn">
              <Crosshair className="h-4 w-4" aria-hidden />{snap.scene?.calibrated ? 'Adjust chair' : 'Set up chair'}
            </Link>
          </Tip>
        </div>
        <p className="basis-full t-label text-mute">
          <span className="block max-w-[90ch]">
            Everything the cameras measure for this bed: the live view, heart and breathing rate, the early-warning
            score and alarms, trends, and each method behind the numbers.
          </span>
        </p>
      </header>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-7 flex flex-col gap-3 min-w-0">
          <div role="tablist" aria-label="Live view" className="seg w-fit">
            <button type="button" role="tab" aria-selected={view === 'camera'} onClick={() => setView('camera')}>Colour camera</button>
            <button type="button" role="tab" aria-selected={view === 'thermal'} onClick={() => setView('thermal')}>Thermal</button>
          </div>
          <div role="tabpanel">{view === 'camera' ? <StreamPanel snap={snap} /> : <ThermalPanel snap={snap} />}</div>
        </div>
        <div className="xl:col-span-5 flex flex-col gap-5 min-w-0">
          <section className="card card-pad grid gap-6 sm:grid-cols-2" aria-label="Vital signs">
            <VitalReadout kind="hr" value={v?.hr.value ?? null} stale={v?.hr.stale ?? true} sqi={v?.hr.sqi ?? 0}
                          subscore={v?.hr.subscore ?? null} reason={reasonFor(v?.hr.reason)} ageS={v?.hr.age_s} />
            <VitalReadout kind="rr" value={v?.rr.value ?? null} stale={v?.rr.stale ?? true} sqi={v?.rr.sqi ?? 0}
                          subscore={v?.rr.subscore ?? null} reason={reasonFor(v?.rr.reason)} apnea={v?.rr.apnea} ageS={v?.rr.age_s} />
          </section>
          <section className="card card-pad flex flex-col gap-3" aria-labelledby="ews-h">
            <SectionHead id="ews-h" title="Early warning" info="ews" />
            <EWSIndicator ews={snap.ews} />
          </section>
          <AlarmsPanel snap={snap} />
        </div>
      </div>

      <section className="flex flex-col gap-4" aria-labelledby="trends-h">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex items-center gap-2">
            <h2 id="trends-h" className="t-title">Trends</h2>
            <InfoTip term="trends" />
          </div>
          <Tip term="overlay" asChild>
            <label className="ml-auto inline-flex items-center gap-2 t-label min-h-[40px] cursor-pointer">
              <input type="checkbox" className="h-4 w-4 accent-current" checked={showSources} onChange={(e) => setShowSources(e.target.checked)} />
              Overlay every estimator
            </label>
          </Tip>
          <Tip term="trendWindow" asChild>
            <div role="radiogroup" aria-label="Trend window" className="seg">
              {WINDOWS.map((w) => (
                <button key={w.m} type="button" role="radio" aria-checked={minutes === w.m} onClick={() => setMinutes(w.m)}>{w.label}</button>
              ))}
            </div>
          </Tip>
        </div>
        <div className="grid gap-5 xl:grid-cols-2">
          <VitalsTrendChart kind="hr" history={history} minutes={minutes} showSources={showSources} />
          <VitalsTrendChart kind="rr" history={history} minutes={minutes} showSources={showSources} />
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-7 min-w-0"><SourcesPanel snap={snap} /></div>
        <section className="xl:col-span-5 card card-pad flex flex-col gap-4" aria-labelledby="rec-h">
          <SectionHead id="rec-h" title="Record a comparison session" info="recording">
            <Link to="/compare" className="btn btn-ghost btn-sm">Results</Link>
          </SectionHead>
          <SessionControl patientId={snap.patient_id} compact />
        </section>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <FallDetectorPanel fall={snap.fall} scene={snap.scene} history={history} />
        <TAAPanel taa={snap.taa} />
        <PainPanel pain={snap.pain} patientId={snap.patient_id} />
      </div>

      <section className="card card-pad flex flex-col gap-2" aria-labelledby="mods-h">
        <SectionHead id="mods-h" title="Pipeline status" info="modules" />
        <ModuleHealth modules={snap.modules} />
      </section>
    </div>
  );
}
