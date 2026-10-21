// TAAPanel.tsx — thoraco-abdominal asynchrony ("paradoxical breathing").
// Chest and abdomen depth signals should rise together (phase ~0). A sustained
// phase difference above 1.2 rad (~68°) for 10 s raises the flag.
import { TriangleAlert } from 'lucide-react';
import type { TAAInfo } from '../types';

const THRESH = 1.2;

export default function TAAPanel({ taa }: { taa: TAAInfo | null }) {
  const phase = taa?.phase_rad ?? null;
  const flagged = !!taa?.flag;
  const frac = phase === null ? 0 : Math.min(1, Math.max(0, phase / Math.PI));
  const thr = THRESH / Math.PI;

  return (
    <section className="panel p-5 flex flex-col gap-4" aria-labelledby="taa-h">
      <div className="flex items-center justify-between gap-3">
        <h2 id="taa-h" className="t-head">Breathing synchrony</h2>
        {flagged && (
          <span className="t-micro inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full bg-[#3a2611] text-[#ffc488] border border-[#6b4419]">
            <TriangleAlert className="h-3.5 w-3.5" aria-hidden /> Paradoxical
          </span>
        )}
      </div>
      <p className="t-label text-ink2 max-w-[58ch]">
        Chest and abdomen should move together. A lasting phase gap suggests respiratory muscle fatigue
        or airway obstruction.
      </p>

      {taa?.stale || phase === null ? (
        <p className="t-label text-mute">
          {taa?.reason === 'amplitude_below_min' ? 'Breathing movement too small to compare.' : 'Waiting for 20 s of chest and abdomen signal.'}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <div className="relative h-2 rounded-full bg-raised" role="meter" aria-label="Chest-abdomen phase gap"
               aria-valuemin={0} aria-valuemax={180} aria-valuenow={Math.round((phase * 180) / Math.PI)}>
            <div className={`absolute inset-y-0 left-0 rounded-full ${flagged ? 'bg-amber' : 'bg-ink2'}`}
                 style={{ width: `${frac * 100}%` }} />
            <div className="absolute -top-1 -bottom-1 w-[2px] bg-ink" style={{ left: `${thr * 100}%` }}
                 title="Alert threshold 68°" />
          </div>
          <div className="flex justify-between t-micro text-mute tnum">
            <span>In phase</span>
            <span>{Math.round((phase * 180) / Math.PI)}° gap{taa?.sustain_s ? `, ${Math.round(taa.sustain_s)} s` : ''}</span>
            <span>Opposite</span>
          </div>
        </div>
      )}
    </section>
  );
}
