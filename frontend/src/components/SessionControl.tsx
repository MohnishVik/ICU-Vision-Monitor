// SessionControl.tsx — start / annotate / stop a numbers-only comparison recording on one bed.
//
// While a session runs, every RR/HR estimator's value is logged once a second next to the
// reference you trust (metronome rate, counted breaths, pulse oximeter). Nothing visual is
// stored. Score sessions on the Compare page or with scripts/compare_sessions.py.
import { Circle, Plus, Square } from 'lucide-react';
import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { api } from '../lib/api';
import { duration } from '../lib/format';
import type { SessionStatus } from '../types';

export const CONDITIONS = [
  'Rest, normal breathing', 'Paced breathing (metronome)', 'Dim light', 'Talking or moving',
  'After exercise', 'Breath-hold', 'Other',
];

const num = (s: string): number | null => {
  const v = parseFloat(s);
  return s.trim() === '' || !Number.isFinite(v) ? null : v;
};

export function useSession(patientId: string | undefined, everyMs = 2000) {
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const refresh = useCallback(() => {
    if (!patientId) return;
    api.session(patientId).then(setStatus).catch(() => undefined);
  }, [patientId]);
  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, everyMs);
    return () => window.clearInterval(t);
  }, [refresh, everyMs]);
  return { status, setStatus, refresh };
}

export default function SessionControl({ patientId, onStopped, compact = false }: {
  patientId: string; onStopped?: (id: string) => void; compact?: boolean;
}) {
  const { status, setStatus } = useSession(patientId);
  const [label, setLabel] = useState('');
  const [condition, setCondition] = useState(CONDITIONS[0]);
  const [rr, setRr] = useState('');
  const [hr, setHr] = useState('');
  const [notes, setNotes] = useState('');
  const [refRr, setRefRr] = useState('');
  const [refHr, setRefHr] = useState('');
  const [refNote, setRefNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async (fn: () => Promise<SessionStatus>) => {
    setBusy(true);
    setErr(null);
    try {
      const s = await fn();
      setStatus(s);
      return s;
    } catch (e) {
      setErr((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  };

  const start = (e: FormEvent) => {
    e.preventDefault();
    void run(() => api.startSession(patientId, {
      label: label.trim() || condition, condition, notes, reference: { rr: num(rr), hr: num(hr) } }));
  };
  const addRef = (e: FormEvent) => {
    e.preventDefault();
    if (num(refRr) === null && num(refHr) === null) { setErr('Enter a breathing rate, a heart rate, or both.'); return; }
    void run(() => api.addReference(patientId, { rr: num(refRr), hr: num(refHr), note: refNote })).then((s) => {
      if (s) { setRefRr(''); setRefHr(''); setRefNote(''); }
    });
  };
  const stop = () => void run(() => api.stopSession(patientId)).then((s) => { if (s?.id) onStopped?.(s.id); });

  if (status?.active) {
    const ref = status.reference ?? { rr: null, hr: null };
    return (
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="chip band-red"><span className="h-2 w-2 rounded-full bg-red rec-dot" aria-hidden />Recording</span>
          <span className="t-head truncate">{status.label}</span>
          <span className="t-label text-mute tnum">{duration(status.elapsed_s)} · {status.rows ?? 0} s logged</span>
          <button type="button" className="btn btn-sm ml-auto" disabled={busy} onClick={stop}>
            <Square className="h-4 w-4" aria-hidden />Stop
          </button>
        </div>
        <dl className="grid grid-cols-2 gap-3">
          <div className="well px-3 py-2"><dt className="t-micro text-mute">Reference RR</dt>
            <dd className="font-cond font-semibold text-[24px] tnum">{ref.rr ?? '–'} <span className="t-micro text-mute">br/min</span></dd></div>
          <div className="well px-3 py-2"><dt className="t-micro text-mute">Reference HR</dt>
            <dd className="font-cond font-semibold text-[24px] tnum">{ref.hr ?? '–'} <span className="t-micro text-mute">bpm</span></dd></div>
        </dl>
        <form onSubmit={addRef} className="flex flex-col gap-3" aria-label="Update the reference">
          <p className="t-label text-ink2">Changed pace, or took a new oximeter reading? Log it as it happens:</p>
          <div className="grid grid-cols-2 gap-3">
            <label className="field"><span>Breathing rate (br/min)</span>
              <input className="input tnum" inputMode="decimal" value={refRr} onChange={(e) => setRefRr(e.target.value)} /></label>
            <label className="field"><span>Heart rate (bpm)</span>
              <input className="input tnum" inputMode="decimal" value={refHr} onChange={(e) => setRefHr(e.target.value)} /></label>
          </div>
          {!compact && <label className="field"><span>Note (optional)</span>
            <input className="input" value={refNote} maxLength={200} onChange={(e) => setRefNote(e.target.value)}
                   placeholder="e.g. metronome to 20" /></label>}
          <button type="submit" className="btn w-fit" disabled={busy}><Plus className="h-4 w-4" aria-hidden />Log reference</button>
        </form>
        {status.error && <p className="t-label text-red" role="alert">Recording stopped: {status.error}</p>}
        {err && <p className="t-label text-red" role="alert">{err}</p>}
      </div>
    );
  }

  return (
    <form onSubmit={start} className="flex flex-col gap-3" aria-label="Start a comparison recording">
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="field"><span>Condition</span>
          <select className="input" value={condition} onChange={(e) => setCondition(e.target.value)}>
            {CONDITIONS.map((c) => <option key={c}>{c}</option>)}
          </select></label>
        <label className="field"><span>Label</span>
          <input className="input" value={label} maxLength={80} onChange={(e) => setLabel(e.target.value)}
                 placeholder="e.g. Sub01 paced 12" /></label>
        <label className="field"><span>Reference breathing rate (br/min)</span>
          <input className="input tnum" inputMode="decimal" value={rr} onChange={(e) => setRr(e.target.value)}
                 placeholder="metronome or counted" /></label>
        <label className="field"><span>Reference heart rate (bpm)</span>
          <input className="input tnum" inputMode="decimal" value={hr} onChange={(e) => setHr(e.target.value)}
                 placeholder="pulse oximeter" /></label>
      </div>
      {!compact && <label className="field"><span>Notes (optional)</span>
        <textarea className="input" value={notes} maxLength={500} onChange={(e) => setNotes(e.target.value)}
                  placeholder="lighting, distance, clothing… no names" /></label>}
      <div className="flex flex-wrap items-center gap-3">
        <button type="submit" className="btn btn-primary" disabled={busy}>
          <Circle className="h-4 w-4 fill-current text-red" aria-hidden />Start recording
        </button>
        <span className="t-micro text-mute">Numbers only: no video or images are saved.</span>
      </div>
      {err && <p className="t-label text-red" role="alert">{err}</p>}
    </form>
  );
}
