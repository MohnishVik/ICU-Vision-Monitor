// SessionControl.tsx — start / annotate / stop a numbers-only comparison recording on one bed.
//
// While a session runs, every RR/HR estimator's value is logged once a second next to the
// reference you trust (metronome rate, counted breaths, pulse oximeter). Nothing visual is
// stored. Score sessions on the Compare page or with scripts/compare_sessions.py.
// Optional training capture adds one numbers-only row per camera frame, and events marked during
// the session (a fall, talking, a CPOT score) give the learned models their labels.
import { Circle, Plus, Square } from 'lucide-react';
import { cloneElement, useCallback, useEffect, useId, useState, type FormEvent, type ReactElement } from 'react';
import { api } from '../lib/api';
import { clockSec, duration } from '../lib/format';
import type { GlossKey } from '../lib/glossary';
import type { SessionStatus } from '../types';
import { InfoTip, Tip } from './Tooltip';

/** A labelled field whose label explains itself on hover; the control carries the same text
 *  as its description, so keyboard and screen-reader users get it on focus. */
function Field({ term, label, children }: { term: GlossKey; label: string; children: ReactElement }) {
  const id = `field-${useId().replace(/:/g, '')}`;
  return (
    <Tip term={term} id={id} asChild>
      <label className="field"><span>{label}</span>{cloneElement(children, { 'aria-describedby': id })}</label>
    </Tip>
  );
}

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

/** Event buttons, in the order a nurse is likely to reach for them. Kinds match
 *  session_log.EVENT_KINDS; pain_score has its own control below. */
const EVENTS: { kind: string; label: string }[] = [
  { kind: 'fall', label: 'Fall' },
  { kind: 'chair_exit', label: 'Chair exit' },
  { kind: 'sit_to_stand', label: 'Sit-to-stand' },
  { kind: 'lying_down', label: 'Lying down' },
  { kind: 'talking', label: 'Talking' },
  { kind: 'movement', label: 'Movement' },
  { kind: 'breath_hold_start', label: 'Breath-hold start' },
  { kind: 'breath_hold_end', label: 'Breath-hold end' },
  { kind: 'paradoxical_start', label: 'Paradoxical breathing start' },
  { kind: 'paradoxical_end', label: 'Paradoxical breathing end' },
  { kind: 'procedure_start', label: 'Procedure start' },
  { kind: 'procedure_end', label: 'Procedure end' },
  { kind: 'other', label: 'Other' },
];

const PAIN_SCALES = { CPOT: { min: 0, max: 8 }, BPS: { min: 3, max: 12 } } as const;
type PainScale = keyof typeof PAIN_SCALES;

/** "Mark an event": each press posts at once and says so inline. Neutral buttons on purpose:
 *  these are annotations in the record, never alarms, so they stay out of the red/orange family. */
function EventMarker({ patientId, onStatus }: { patientId: string; onStatus: (s: SessionStatus) => void }) {
  const [kinds, setKinds] = useState<string[] | null>(null);
  const [note, setNote] = useState('');
  const [scale, setScale] = useState<PainScale>('CPOT');
  const [score, setScore] = useState('');
  const [pending, setPending] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  // hide any button the server does not know (older backend); show all until the list arrives
  useEffect(() => {
    api.sessionEventKinds().then((r) => setKinds(r.kinds)).catch(() => undefined);
  }, []);
  const known = (k: string) => kinds === null || kinds.includes(k);

  const mark = async (kind: string, label: string, value: number | null = null, prefix = '') => {
    setPending(true);
    try {
      const n = note.trim();
      const s = await api.sessionEvent(patientId, kind, value, [prefix, n].filter(Boolean).join('; '));
      onStatus(s);
      setMsg({ ok: true, text: `${label} marked at ${clockSec(Date.now() / 1000)}` });
      setNote('');
      if (kind === 'pain_score') setScore('');
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setPending(false);
    }
  };

  const markPain = (e: FormEvent) => {
    e.preventDefault();
    const { min, max } = PAIN_SCALES[scale];
    const v = num(score);
    if (v === null || !Number.isInteger(v) || v < min || v > max) {
      setMsg({ ok: false, text: `Enter a whole-number ${scale} score from ${min} to ${max}.` });
      return;
    }
    void mark('pain_score', `Pain score ${v} (${scale})`, v, scale);
  };

  const { min, max } = PAIN_SCALES[scale];
  return (
    <section className="rule-t pt-4 flex flex-col gap-3" aria-labelledby={`evt-h-${patientId}`}>
      <div className="flex items-center gap-2">
        <h3 id={`evt-h-${patientId}`} className="t-strong text-ink">Mark an event</h3>
        <InfoTip term="sessionEvents" />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {EVENTS.filter((e) => known(e.kind)).map((e) => (
          <button key={e.kind} type="button" className="btn whitespace-normal text-center leading-tight py-1"
                  disabled={pending} onClick={() => void mark(e.kind, e.label)}>
            {e.label}
          </button>
        ))}
      </div>
      {known('pain_score') && (
        <form onSubmit={markPain} className="flex flex-wrap items-end gap-3" aria-label="Mark a pain score">
          <Tip term="painScoreCPOT" asChild>
            <div role="radiogroup" aria-label="Pain scale" className="seg self-end">
              {(Object.keys(PAIN_SCALES) as PainScale[]).map((k) => (
                <button key={k} type="button" role="radio" aria-checked={scale === k} onClick={() => setScale(k)}>{k}</button>
              ))}
            </div>
          </Tip>
          <div className="w-28">
            <Field term="painScoreCPOT" label={`Score (${min}–${max})`}>
              <input className="input tnum" inputMode="numeric" value={score} maxLength={2}
                     onChange={(e) => setScore(e.target.value)} />
            </Field>
          </div>
          <button type="submit" className="btn" disabled={pending}>Mark pain score</button>
        </form>
      )}
      <Field term="sessionEvents" label="Note for the next event (optional)">
        <input className="input" value={note} maxLength={180} onChange={(e) => setNote(e.target.value)}
               placeholder="what happened… no names" />
      </Field>
      {/* one reserved line, so a confirmation or error never pushes the layout */}
      <p className={`t-label min-h-[1.5em] tnum ${msg && !msg.ok ? 'text-red' : 'text-ink2'}`}
         role={msg && !msg.ok ? 'alert' : 'status'} aria-live="polite">
        {msg?.text ?? ''}
      </p>
    </section>
  );
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
  const [capture, setCapture] = useState(false);
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
      label: label.trim() || condition, condition, notes, reference: { rr: num(rr), hr: num(hr) }, capture }));
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
          <Tip term="recording" className="chip band-red"><span className="h-2 w-2 rounded-full bg-red rec-dot" aria-hidden />Recording</Tip>
          <span className="t-head truncate">{status.label}</span>
          <Tip title="Elapsed and logged" className="t-label text-mute tnum"
               content="Time since the recording started, and how many one-second rows have been written so far.">
            {duration(status.elapsed_s)} · {status.rows ?? 0} s logged
          </Tip>
          {status.capture && (
            <Tip term="trainingCapture" className="t-label text-mute tnum">
              Training capture on · {(status.frames ?? 0).toLocaleString()} frames
            </Tip>
          )}
          <Tip term="sessionEvents" className="t-label text-mute tnum">
            {status.events ?? 0} {status.events === 1 ? 'event' : 'events'}
          </Tip>
          <Tip title="Stop recording" content="Ends the session and saves it; it then appears on the Record & compare page for scoring." asChild>
            <button type="button" className="btn btn-sm ml-auto" disabled={busy} onClick={stop}>
              <Square className="h-4 w-4" aria-hidden />Stop
            </button>
          </Tip>
        </div>
        <dl className="grid grid-cols-2 gap-3">
          <div className="well px-3 py-2"><dt className="t-micro text-mute"><Tip term="refRR" className="tip-term">Reference RR</Tip></dt>
            <dd className="font-cond font-semibold text-[24px] tnum">{ref.rr ?? '–'} <span className="t-micro text-mute">br/min</span></dd></div>
          <div className="well px-3 py-2"><dt className="t-micro text-mute"><Tip term="refHR" className="tip-term">Reference HR</Tip></dt>
            <dd className="font-cond font-semibold text-[24px] tnum">{ref.hr ?? '–'} <span className="t-micro text-mute">bpm</span></dd></div>
        </dl>
        <form onSubmit={addRef} className="flex flex-col gap-3" aria-label="Update the reference">
          <p className="t-label text-ink2">Changed pace, or took a new oximeter reading? Log it as it happens:</p>
          <div className="grid grid-cols-2 gap-3">
            <Field term="refRR" label="Breathing rate (br/min)">
              <input className="input tnum" inputMode="decimal" value={refRr} onChange={(e) => setRefRr(e.target.value)} /></Field>
            <Field term="refHR" label="Heart rate (bpm)">
              <input className="input tnum" inputMode="decimal" value={refHr} onChange={(e) => setRefHr(e.target.value)} /></Field>
          </div>
          {!compact && <Field term="sessNotes" label="Note (optional)">
            <input className="input" value={refNote} maxLength={200} onChange={(e) => setRefNote(e.target.value)}
                   placeholder="e.g. metronome to 20" /></Field>}
          <Tip term="logReference" asChild>
            <button type="submit" className="btn w-fit" disabled={busy}><Plus className="h-4 w-4" aria-hidden />Log reference</button>
          </Tip>
        </form>
        <EventMarker patientId={patientId} onStatus={setStatus} />
        {status.error && <p className="t-label text-red" role="alert">Recording stopped: {status.error}</p>}
        {err && <p className="t-label text-red" role="alert">{err}</p>}
      </div>
    );
  }

  return (
    <form onSubmit={start} className="flex flex-col gap-3" aria-label="Start a comparison recording">
      <div className="grid gap-3 sm:grid-cols-2">
        <Field term="condition" label="Condition">
          <select className="input" value={condition} onChange={(e) => setCondition(e.target.value)}>
            {CONDITIONS.map((c) => <option key={c}>{c}</option>)}
          </select></Field>
        <Field term="sessLabel" label="Label">
          <input className="input" value={label} maxLength={80} onChange={(e) => setLabel(e.target.value)}
                 placeholder="e.g. Sub01 paced 12" /></Field>
        <Field term="refRR" label="Reference breathing rate (br/min)">
          <input className="input tnum" inputMode="decimal" value={rr} onChange={(e) => setRr(e.target.value)}
                 placeholder="metronome or counted" /></Field>
        <Field term="refHR" label="Reference heart rate (bpm)">
          <input className="input tnum" inputMode="decimal" value={hr} onChange={(e) => setHr(e.target.value)}
                 placeholder="pulse oximeter" /></Field>
      </div>
      {!compact && <Field term="sessNotes" label="Notes (optional)">
        <textarea className="input" value={notes} maxLength={500} onChange={(e) => setNotes(e.target.value)}
                  placeholder="lighting, distance, clothing… no names" /></Field>}
      <Tip term="trainingCapture" asChild>
        <label className="inline-flex items-center gap-2 t-label min-h-[40px] cursor-pointer w-fit">
          <input type="checkbox" className="h-4 w-4 accent-current" checked={capture}
                 onChange={(e) => setCapture(e.target.checked)} />
          Capture training data
        </label>
      </Tip>
      <div className="flex flex-wrap items-center gap-3">
        <Tip term="recording" asChild>
          <button type="submit" className="btn btn-primary" disabled={busy}>
            <Circle className="h-4 w-4 fill-current text-red" aria-hidden />Start recording
          </button>
        </Tip>
        <span className="t-micro text-mute">Numbers only: no video or images are saved
          {capture ? '; training capture adds about 30 rows a second.' : '.'}</span>
      </div>
      {err && <p className="t-label text-red" role="alert">{err}</p>}
    </form>
  );
}
