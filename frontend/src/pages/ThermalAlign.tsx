// ThermalAlign.tsx — one-time alignment of the ESP32 thermal image with the colour camera.
//
// The two cameras sit a few centimetres apart with different lenses, so the
// same thermal and colour pixel do not look at the same spot. Marking the same
// physical point in both images (4+ times) lets the server solve the mapping,
// after which respiratory rate is read from the temperature at the nostrils.
// Faces are blurred in the colour image, so a warm object (a mug of hot water)
// is the marker.
import { ArrowLeft, Camera, Check, Trash2, Undo2, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { Link, useParams } from 'react-router-dom';
import ThermalCanvas, { robustRange, ThermalLegend, type ThermalMarker } from '../components/ThermalCanvas';
import { api, urls } from '../lib/api';
import type { PointPair, ThermalRegistrationStatus, ThermalSnapshot } from '../types';

type Pt = [number, number];

const QUALITY_TEXT = {
  good: 'Good alignment',
  fair: 'Fair alignment',
  poor: 'Poor alignment: realign',
  unchecked: 'Aligned (4 points fit exactly, so accuracy is unchecked)',
} as const;

export default function ThermalAlign() {
  const { id = '' } = useParams();
  const [status, setStatus] = useState<ThermalRegistrationStatus | null>(null);
  const [snap, setSnap] = useState<ThermalSnapshot | null>(null);
  const [pairs, setPairs] = useState<PointPair[]>([]);
  const [pending, setPending] = useState<{ rgb?: Pt; thermal?: Pt }>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkNonce, setCheckNonce] = useState(0);
  const [justSaved, setJustSaved] = useState(false);
  const rgbSvg = useRef<SVGSVGElement>(null);

  useEffect(() => {
    api.thermalRegistration(id).then(setStatus).catch((e: Error) => setError(e.message));
  }, [id]);

  const range = useMemo(() => (snap ? robustRange(snap.thermal.data) : ([20, 37] as [number, number])), [snap]);

  const take = async () => {
    setBusy(true);
    setError(null);
    setPending({});
    try {
      setSnap(await api.thermalSnapshot(id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const complete = (next: { rgb?: Pt; thermal?: Pt }) => {
    if (next.rgb && next.thermal) {
      setPairs((p) => [...p, { rgb: next.rgb!, thermal: next.thermal! }]);
      setPending({});
    } else {
      setPending(next);
    }
  };

  const onRgb = (e: PointerEvent<SVGSVGElement>) => {
    const el = rgbSvg.current;
    const m = el?.getScreenCTM();
    if (!el || !m || !snap) return;
    const p = el.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(m.inverse());
    const [h, w] = snap.rgb_size;
    if (q.x < 0 || q.y < 0 || q.x > w || q.y > h) return;
    complete({ ...pending, rgb: [Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10] });
  };
  const onThermal = (x: number, y: number) =>
    complete({ ...pending, thermal: [Math.round(x * 100) / 100, Math.round(y * 100) / 100] });

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.setThermalRegistration(id, pairs));
      setJustSaved(true);
      setCheckNonce((n) => n + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      setStatus(await api.clearThermalRegistration(id));
      setJustSaved(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const markers = (key: 'rgb' | 'thermal'): ThermalMarker[] => [
    ...pairs.map((p, i) => ({ x: p[key][0], y: p[key][1], label: String(i + 1) })),
    ...(pending[key] ? [{ x: pending[key]![0], y: pending[key]![1], label: String(pairs.length + 1), pending: true }] : []),
  ];

  const prompt = !snap ? 'Take a snapshot to start.'
    : pending.rgb && !pending.thermal ? `Now click the same spot in the thermal image (point ${pairs.length + 1}).`
      : pending.thermal && !pending.rgb ? `Now click the same spot in the colour image (point ${pairs.length + 1}).`
        : 'Click the mug in the colour image, then in the thermal image.';
  const [rh, rw] = snap?.rgb_size ?? [480, 640];
  const r = rw / 64;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-center gap-3">
        <Link to={`/patients/${encodeURIComponent(id)}`} className="btn btn-ghost px-2" aria-label="Back to bed">
          <ArrowLeft className="h-4 w-4" aria-hidden />
        </Link>
        <h1 className="font-cond font-semibold text-[40px] leading-none tracking-[-0.015em]">Align the thermal camera</h1>
      </header>

      <div className="grid gap-5 lg:grid-cols-12">
        <div className="lg:col-span-8 flex flex-col gap-4 min-w-0">
          <p className="t-body text-ink" aria-live="polite">{prompt}</p>
          {snap ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <figure className="m-0 flex flex-col gap-2">
                <div className="relative w-full overflow-hidden rounded-lg bg-black" style={{ aspectRatio: `${rw} / ${rh}` }}>
                  <img src={urls.snapshotRgb(id, snap.snapshot_id)} alt="Frozen colour frame, faces blurred"
                       className="absolute inset-0 h-full w-full object-contain select-none" draggable={false} />
                  <svg ref={rgbSvg} viewBox={`0 0 ${rw} ${rh}`} className="absolute inset-0 h-full w-full cursor-crosshair"
                       onPointerDown={onRgb} role="img" aria-label="Colour image: click the mug">
                    {markers('rgb').map((m, i) => (
                      <g key={i} opacity={m.pending ? 0.75 : 1}>
                        <circle cx={m.x} cy={m.y} r={r} fill="var(--ink)" stroke="var(--bg)" strokeWidth={1.5}
                                vectorEffect="non-scaling-stroke" />
                        <text x={m.x} y={m.y} dy="0.36em" textAnchor="middle" fontSize={r * 1.2}
                              fontFamily="IBM Plex Sans Condensed" fontWeight={600} fill="var(--bg)">{m.label}</text>
                      </g>
                    ))}
                  </svg>
                </div>
                <figcaption className="t-micro text-mute">Colour camera</figcaption>
              </figure>
              <figure className="m-0 flex flex-col gap-2">
                <ThermalCanvas grid={snap.thermal} range={range} markers={markers('thermal')} onPick={onThermal}
                               pixelated label="Thermal image: click the warm mug" />
                <ThermalLegend lo={range[0]} hi={range[1]} />
                <figcaption className="t-micro text-mute">Thermal camera (hover to read °C; the mug is the hot spot)</figcaption>
              </figure>
            </div>
          ) : (
            <div className="card p-8 grid place-items-center text-center">
              <div className="flex flex-col items-center gap-3 max-w-[46ch]">
                <Camera className="h-8 w-8 text-mute" aria-hidden />
                <p className="t-body text-ink2">Hold the mug in view, then take a snapshot of both cameras at the same moment.</p>
                <button type="button" className="btn btn-primary" onClick={() => void take()} disabled={busy}>
                  <Camera className="h-4 w-4" aria-hidden />Take snapshot
                </button>
              </div>
            </div>
          )}

          {justSaved && status?.registered && (
            <section className="card card-pad flex flex-col gap-3" aria-labelledby="check-h">
              <h2 id="check-h" className="t-head">Check the alignment</h2>
              <p className="t-label text-ink2 max-w-[64ch]">
                The thermal image is now drawn on top of the colour image. Warm areas (face, hands, the mug)
                should sit on the matching objects. If they are shifted, add more points and save again.
              </p>
              <img src={`${urls.thermalOverlay(id)}?n=${checkNonce}`} alt="Colour image with the aligned thermal image overlaid"
                   className="w-full rounded-lg bg-black" />
              <button type="button" className="btn w-fit" onClick={() => setCheckNonce((n) => n + 1)}>Refresh check</button>
            </section>
          )}
        </div>

        <aside className="lg:col-span-4 flex flex-col gap-5">
          <section className="card card-pad flex flex-col gap-3" aria-labelledby="how-h">
            <h2 id="how-h" className="t-head">How to align</h2>
            <ol className="flex flex-col gap-2 t-label text-ink2 list-decimal pl-5 max-w-[46ch]">
              <li>Fill a mug with hot water and hold it at about the patient's distance from the camera.</li>
              <li>Take a snapshot. Click the mug in the colour image, then in the thermal image.</li>
              <li>Move the mug to another part of the view, take a new snapshot, and repeat.</li>
              <li>Mark at least 4 spots spread across the view. With 6 or more, the system can measure its own accuracy.</li>
            </ol>
            <p className="t-micro text-mute">
              Faces are blurred in the colour image, so a warm object works better than facial features.
              This only needs redoing if either camera is moved.
            </p>
          </section>

          <section className="card card-pad flex flex-col gap-3" aria-labelledby="pairs-h">
            <div className="flex items-baseline gap-2">
              <h2 id="pairs-h" className="t-head">Marked points</h2>
              <span className="t-micro text-mute tnum ml-auto">
                {pairs.length < 4 ? `${pairs.length} of 4 needed` : `${pairs.length} marked`}
              </span>
            </div>
            {pairs.length === 0 ? (
              <p className="t-label text-mute">None yet.</p>
            ) : (
              <ul className="flex flex-col">
                {pairs.map((p, i) => (
                  <li key={i} className="flex items-center gap-3 py-1.5 rule-b last:border-b-0 t-label tnum">
                    <span className="h-6 w-6 rounded-full bg-ink text-[var(--bg)] grid place-items-center t-micro font-semibold">{i + 1}</span>
                    <span className="text-ink2 flex-1">
                      colour {Math.round(p.rgb[0])}, {Math.round(p.rgb[1])} &nbsp;thermal {p.thermal[0].toFixed(1)}, {p.thermal[1].toFixed(1)}
                    </span>
                    <button type="button" className="btn btn-ghost h-7 px-1.5" aria-label={`Remove point ${i + 1}`}
                            onClick={() => setPairs((s) => s.filter((_, j) => j !== i))}>
                      <X className="h-3.5 w-3.5" aria-hidden />
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex flex-wrap gap-2">
              <button type="button" className="btn" onClick={() => void take()} disabled={busy}>
                <Camera className="h-4 w-4" aria-hidden />{snap ? 'New snapshot' : 'Take snapshot'}
              </button>
              <button type="button" className="btn" disabled={!pairs.length && !pending.rgb && !pending.thermal}
                      onClick={() => (pending.rgb || pending.thermal ? setPending({}) : setPairs((s) => s.slice(0, -1)))}>
                <Undo2 className="h-4 w-4" aria-hidden />Undo
              </button>
            </div>
            <button type="button" className="btn btn-primary h-11 justify-center" disabled={pairs.length < 4 || busy}
                    onClick={() => void save()}>
              <Check className="h-4 w-4" aria-hidden />{busy ? 'Saving' : 'Save alignment'}
            </button>
            {error && <p className="t-label text-red" role="alert">{error}</p>}
          </section>

          <section className="card card-pad flex flex-col gap-2" aria-labelledby="cur-h">
            <h2 id="cur-h" className="t-head">Current alignment</h2>
            {status?.registered ? (
              <>
                <p className="t-label text-ink">{QUALITY_TEXT[status.quality ?? 'unchecked']}</p>
                <p className="t-micro text-mute tnum">
                  {status.n_pairs > 0 ? `${status.n_pairs} points` : 'Fixed matrix'}
                  {status.error_px != null && status.n_pairs > 4 ? `, average error ${status.error_px.toFixed(1)} px` : ''}
                  {status.source === 'camera.yaml' ? ', from camera.yaml' : status.source === 'demo' ? ', simulated' : ''}
                </p>
                <button type="button" className="btn btn-ghost w-fit px-2" onClick={() => void remove()} disabled={busy}>
                  <Trash2 className="h-4 w-4" aria-hidden />Remove alignment
                </button>
              </>
            ) : (
              <p className="t-label text-ink2">Not aligned. Breathing is read from the warmest spot.</p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
