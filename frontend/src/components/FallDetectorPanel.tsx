// FallDetectorPanel.tsx — where the patient is, and the fall detector's state.
// Posture over time is shown as a strip: normal is quiet, falling amber, lying red,
// each also named in the legend so colour never works alone.
import { Armchair, PersonStanding, ShieldAlert, Users } from 'lucide-react';
import { ago, pct, PATIENT_STATE_TEXT } from '../lib/format';
import type { FallInfo, History, Scene } from '../types';

const STATE_TEXT = { NORMAL: 'Upright or seated', FALLING: 'Falling motion', LYING_DOWN: 'Lying down',
                     UNKNOWN: 'No reading' } as const;

function PostureStrip({ history }: { history: History | null }) {
  const vals = history?.fall ?? [];
  if (vals.length < 2) {
    return <p className="t-micro text-mute">Posture history builds up as the bed is monitored.</p>;
  }
  const n = vals.length;
  return (
    <div className="flex flex-col gap-1.5">
      <svg viewBox={`0 0 ${n} 10`} preserveAspectRatio="none" className="w-full h-3 rounded-sm overflow-hidden"
           role="img" aria-label="Posture over the selected window">
        <rect x={0} y={0} width={n} height={10} fill="var(--surface-2)" />
        {vals.map((v, i) => (v === 1 || v === 2) && (
          <rect key={i} x={i} y={0} width={1.02} height={10} fill={v === 2 ? 'var(--red)' : 'var(--amber)'} />
        ))}
      </svg>
      <div className="flex gap-4 t-micro text-mute">
        <span className="inline-flex items-center gap-1.5"><i className="h-2 w-3 rounded-sm bg-surface2" />Normal</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-2 w-3 rounded-sm bg-amber" />Falling</span>
        <span className="inline-flex items-center gap-1.5"><i className="h-2 w-3 rounded-sm bg-red" />Lying</span>
      </div>
    </div>
  );
}

export default function FallDetectorPanel({ fall, scene, history }: {
  fall: FallInfo | null;
  scene: Scene | null;
  history: History | null;
}) {
  const ps = fall?.patient_state ?? 'UNKNOWN';
  const danger = ps === 'ON_FLOOR' || fall?.state_name === 'FALLING';
  const Icon = ps === 'IN_CHAIR' || ps === 'SITTING_EDGE' ? Armchair : danger ? ShieldAlert : PersonStanding;
  const frac = scene?.in_chair_frac ?? 0;

  return (
    <section className="card card-pad flex flex-col gap-4" aria-labelledby="fall-h">
      <h2 id="fall-h" className="t-head">Position and falls</h2>
      <div className="flex items-center gap-3">
        <Icon className={`h-7 w-7 ${danger ? 'text-red' : 'text-ink2'}`} aria-hidden />
        <div className="flex flex-col">
          <span className={`t-title ${danger ? 'text-redink' : 'text-ink'}`}>{PATIENT_STATE_TEXT[ps]}</span>
          <span className="t-label text-ink2">
            {fall?.stale ? 'Fall detector has no recent reading' : STATE_TEXT[fall?.state_name ?? 'UNKNOWN']}
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between t-label">
          <span className="text-ink2">In the chair</span>
          <span className="tnum">{scene?.has_patient ? pct(frac) : '--'}</span>
        </div>
        <div className="h-1.5 rounded-full bg-surface2 overflow-hidden">
          <div className="h-full bg-ink2 rounded-full" style={{ width: `${Math.round(frac * 100)}%` }} />
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 t-label">
        <dt className="text-ink2">Last confirmed fall</dt>
        <dd className="text-right tnum">{fall?.last_confirmed ? ago(fall.last_confirmed) : 'none this session'}</dd>
        <dt className="text-ink2">Model confidence</dt>
        <dd className="text-right tnum">{fall?.ml_probability === null || fall?.ml_probability === undefined
          ? '--' : pct(fall.ml_probability)}</dd>
        <dt className="text-ink2 inline-flex items-center gap-1.5"><Users className="h-3.5 w-3.5" aria-hidden />Others in view</dt>
        <dd className="text-right tnum">{scene?.staff_count ?? 0}</dd>
      </dl>

      <PostureStrip history={history} />
    </section>
  );
}
