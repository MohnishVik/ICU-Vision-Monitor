// PainPanel.tsx — experimental pain-expression probability from the face crop.
// Shown as a probability against the model's trained threshold, with its limits stated
// plainly. It is never scored and never raises an alarm.
import { FlaskConical, ScanFace } from 'lucide-react';
import type { PainInfo } from '../types';
import { Experimental, Quality } from './ui';

export default function PainPanel({ pain }: { pain: PainInfo | null | undefined }) {
  const p = pain?.probability ?? null;
  const thr = pain?.threshold ?? 0.2962;
  const has = !!pain?.available && p !== null;
  return (
    <section className="card card-pad flex flex-col gap-4" aria-labelledby="pain-h">
      <div className="flex items-start gap-3">
        <div className="flex flex-col gap-1">
          <span className="t-eyebrow text-mute">Face · MobileNetV3</span>
          <h2 id="pain-h" className="t-head">Pain expression</h2>
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
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex items-baseline gap-3">
            <span className="t-num-m tnum">{Math.round(p! * 100)}<span className="t-head text-mute">%</span></span>
            <span className="t-label text-ink2">
              {pain?.pain_detected ? 'above the model’s threshold' : 'below the model’s threshold'}
            </span>
          </div>
          <div className="relative h-2 rounded-full bg-surface3" role="meter" aria-label="Pain-expression probability"
               aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p! * 100)}>
            <div className="absolute inset-y-0 left-0 rounded-full bg-ink2" style={{ width: `${p! * 100}%` }} />
            <div className="absolute -top-1 -bottom-1 w-[2px] bg-ink" style={{ left: `${thr * 100}%` }} />
          </div>
          <div className="flex justify-between t-micro text-mute tnum">
            <span>0%</span><span>threshold {Math.round(thr * 100)}%</span><span>100%</span>
          </div>
          <Quality sqi={pain?.sqi} label="Face" />
        </div>
      )}

      <p className="t-micro text-mute flex gap-1.5 items-start">
        <FlaskConical className="h-3.5 w-3.5 mt-[2px] shrink-0" aria-hidden />
        Trained on part of SynPAIN only (2,667 pain vs 50 no-pain images), so it leans towards “pain”: even a
        blank image scores about 44%. For research; not a pain score and never alarms.
      </p>
    </section>
  );
}
