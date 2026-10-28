// PainPanel.tsx — experimental pain-expression score from the face crop, read against the
// patient's own calm baseline (the model's absolute output is biased high on real faces).
// It is never scored and never raises an alarm.
import { FlaskConical, RotateCcw, ScanFace } from 'lucide-react';
import { useState } from 'react';
import { api } from '../lib/api';
import type { PainInfo } from '../types';
import { InfoTip, Tip } from './Tooltip';
import { Experimental, Quality } from './ui';

export default function PainPanel({ pain, patientId }: { pain: PainInfo | null | undefined; patientId?: string }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const p = pain?.probability ?? null;
  const thr = pain?.threshold ?? null;
  const has = !!pain?.available && p !== null;
  const learning = pain?.baseline_state === 'learning';
  const base = pain?.baseline ?? null;

  const relearn = async () => {
    if (!patientId) return;
    setBusy(true);
    setErr(null);
    try { await api.painBaseline(patientId); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <section className="card card-pad flex flex-col gap-4" aria-labelledby="pain-h">
      <div className="flex items-start gap-3">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <h2 id="pain-h" className="t-head">Pain expression</h2>
            <InfoTip term="pain" />
          </div>
          <span className="t-micro text-mute">Face model, compared with this patient’s calm face</span>
        </div>
        <span className="ml-auto"><Experimental /></span>
      </div>

      {!pain?.available ? (
        <div className="flex gap-3 items-start rounded-md hatch p-3">
          <ScanFace className="h-5 w-5 text-mute shrink-0 mt-0.5" aria-hidden />
          <p className="t-label text-ink2">{pain?.reason || 'Pain model not loaded.'}</p>
        </div>
      ) : !has ? (
        <div className="flex gap-3 items-start rounded-md hatch p-3">
          <ScanFace className="h-5 w-5 text-mute shrink-0 mt-0.5" aria-hidden />
          <p className="t-label text-ink2">
            {pain?.reason === 'face_too_small' ? 'Face too small in the image to score.' : 'No face in view.'}
          </p>
        </div>
      ) : learning || thr === null ? (
        <div className="flex flex-col gap-2">
          <Tip term="painBaseline" className="t-label text-ink2">
            Learning this patient’s calm face · {Math.round((pain?.baseline_progress ?? 0) * 100)}%
          </Tip>
          <span className="block h-1.5 rounded-full bg-surface3 overflow-hidden" aria-hidden>
            <span className="block h-full rounded-full bg-ink2 transition-[width] duration-500"
                  style={{ width: `${Math.round((pain?.baseline_progress ?? 0) * 100)}%` }} />
          </span>
          <p className="t-micro text-mute">Needs about a minute of the face in view while the patient is settled.</p>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline gap-3">
            <Tip term="pain" className="t-num-m tnum">{Math.round(p! * 100)}<span className="t-head text-mute">%</span></Tip>
            <Tip term="painThreshold" className={`t-label ${pain?.pain_detected ? 'text-ink' : 'text-ink2'}`}>
              {pain?.pain_detected ? 'clearly above this patient’s calm level' : 'within this patient’s calm range'}
            </Tip>
          </div>
          <div className="relative h-2 rounded-full bg-surface3" role="meter" aria-label="Pain-expression score"
               aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p! * 100)}>
            <div className="absolute inset-y-0 left-0 rounded-full bg-ink2" style={{ width: `${p! * 100}%` }} />
            {base !== null && (
              <div className="absolute -top-0.5 -bottom-0.5 w-[2px] bg-mute" style={{ left: `${base * 100}%` }} />
            )}
            <div className="absolute -top-1 -bottom-1 w-[2px] bg-ink" style={{ left: `${thr * 100}%` }} />
          </div>
          <div className="flex justify-between t-micro text-mute tnum">
            <Tip term="painBaseline">calm {base !== null ? Math.round(base * 100) : '–'}%</Tip>
            <Tip term="painThreshold">threshold {Math.round(thr * 100)}%</Tip>
          </div>
          <Quality sqi={pain?.sqi} label="Face" term="sqiFace" />
        </div>
      )}

      {patientId && pain?.available && (
        <div className="flex flex-wrap items-center gap-2">
          <Tip term="painBaseline">
            <button type="button" className="btn btn-sm" onClick={() => void relearn()} disabled={busy}>
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />{busy ? 'Resetting' : 'Set calm baseline'}
            </button>
          </Tip>
          {err && <span className="t-micro text-red" role="alert">{err}</span>}
        </div>
      )}

      <p className="t-micro text-mute flex gap-1.5 items-start">
        <FlaskConical className="h-3.5 w-3.5 mt-[2px] shrink-0" aria-hidden />
        Trained on synthetic faces only; on real faces its raw score runs high, so only a change from the
        patient’s own calm face is shown. For research; not a pain score and never alarms.
      </p>
    </section>
  );
}
