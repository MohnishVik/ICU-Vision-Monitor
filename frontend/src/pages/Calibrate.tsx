// Calibrate.tsx — one-time chair/bed setup at the start of a session.
// Click the four corners of the seat on a still frame (faces blurred). Corners
// can be dragged, or moved with arrow keys after focusing them. The server
// orders the corners and samples the seat depth from the aligned depth frame.
import { ArrowLeft, Check, RefreshCw, RotateCcw, Undo2 } from 'lucide-react';
import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { InfoTip, Tip } from '../components/Tooltip';
import { useCalibration } from '../hooks/usePatientData';
import { api, urls } from '../lib/api';

type Pt = [number, number];

export default function Calibrate() {
  const { id = '' } = useParams();
  const nav = useNavigate();
  const { data: status } = useCalibration(id);
  const [pts, setPts] = useState<Pt[]>([]);
  const [label, setLabel] = useState<'chair' | 'bed'>('chair');
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [nonce, setNonce] = useState(0);
  const [imgError, setImgError] = useState(false);
  const [drag, setDrag] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const toImage = (e: PointerEvent): Pt | null => {
    const el = svg.current;
    const m = el?.getScreenCTM();
    if (!el || !m || !size) return null;
    const p = el.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(m.inverse());
    return [Math.max(0, Math.min(size.w, q.x)), Math.max(0, Math.min(size.h, q.y))];
  };

  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    if (drag !== null || pts.length >= 4) return;
    const p = toImage(e);
    if (p) setPts((s) => [...s, p]);
  };
  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    if (drag === null) return;
    const p = toImage(e);
    if (p) setPts((s) => s.map((q, i) => (i === drag ? p : q)));
  };
  const nudge = (i: number, e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2;
    const d: Record<string, Pt> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (d[e.key] && size) {
      e.preventDefault();
      setPts((s) => s.map((q, j) => (j === i
        ? [Math.max(0, Math.min(size.w, q[0] + d[e.key][0])), Math.max(0, Math.min(size.h, q[1] + d[e.key][1]))] as Pt
        : q)));
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      setPts((s) => s.filter((_, j) => j !== i));
    }
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await api.setCalibration(id, pts.map(([x, y]) => [Math.round(x), Math.round(y)] as Pt), label);
      nav(`/patients/${encodeURIComponent(id)}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  const reuse = async () => {
    setSaving(true);
    setError(null);
    try {
      await api.reuseCalibration(id);
      nav(`/patients/${encodeURIComponent(id)}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const remaining = 4 - pts.length;
  const r = size ? Math.max(8, size.w / 64) : 10;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-wrap items-center gap-3">
        <Tip title="Back to the bed" content="Leave without changing the current setup." asChild side="bottom">
          <Link to={`/patients/${encodeURIComponent(id)}`} className="btn btn-ghost px-2" aria-label="Back to bed">
            <ArrowLeft className="h-4 w-4" aria-hidden />
          </Link>
        </Tip>
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="font-cond font-semibold text-[40px] leading-none tracking-[-0.015em]">Mark the {label}</h1>
          <InfoTip term="chairCal" />
        </div>
        <p className="basis-full t-label text-mute">
          <span className="block max-w-[80ch]">
            Trace the seat once per session so the monitor knows where the patient should be; position and fall
            detection are judged against it.
          </span>
        </p>
      </header>

      <div className="grid gap-5 lg:grid-cols-12">
        <div className="lg:col-span-8 card overflow-hidden">
          <div className="relative bg-black" style={{ aspectRatio: size ? `${size.w} / ${size.h}` : '4 / 3' }}>
            {!imgError ? (
              <img key={nonce} src={`${urls.calibrationFrame(id)}?n=${nonce}`} alt="Still camera frame for marking the seat, faces blurred"
                   className="absolute inset-0 h-full w-full object-contain select-none" draggable={false}
                   onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                   onError={() => setImgError(true)} />
            ) : (
              <div className="absolute inset-0 grid place-items-center p-6 text-center">
                <div className="flex flex-col items-center gap-3 max-w-[44ch]">
                  <p className="t-head">No camera frame yet</p>
                  <p className="t-label text-ink2">The camera has not delivered a frame. Check the D435 connection, then refresh.</p>
                  <button type="button" className="btn" onClick={() => { setImgError(false); setNonce((n) => n + 1); }}>
                    <RefreshCw className="h-4 w-4" aria-hidden />Refresh frame
                  </button>
                </div>
              </div>
            )}

            {size && !imgError && (
              <svg ref={svg} viewBox={`0 0 ${size.w} ${size.h}`} className={`absolute inset-0 h-full w-full touch-none ${
                pts.length < 4 ? 'cursor-crosshair' : ''}`}
                   onPointerDown={onDown} onPointerMove={onMove} onPointerUp={() => setDrag(null)}
                   onPointerLeave={() => setDrag(null)} role="application"
                   aria-label={`Seat marking area. ${remaining > 0 ? `${remaining} corners left to place.` : 'All corners placed.'}`}>
                {status?.corners && pts.length === 0 && (
                  <polygon points={status.corners.map((p) => p.join(',')).join(' ')} fill="none"
                           stroke="var(--mute)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                )}
                {pts.length >= 2 && (
                  <polygon points={pts.map((p) => p.join(',')).join(' ')}
                           fill={pts.length === 4 ? 'color-mix(in srgb, var(--ink) 12%, transparent)' : 'none'}
                           stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                )}
                {pts.map((p, i) => (
                  <g key={i} tabIndex={0} role="button" aria-label={`Corner ${i + 1}. Drag or use arrow keys to move, Delete to remove.`}
                     onKeyDown={(e) => nudge(i, e)}
                     onPointerDown={(e) => { e.stopPropagation(); (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(i); }}
                     className="cursor-grab focus:outline-none">
                    <circle cx={p[0]} cy={p[1]} r={r * 1.8} fill="transparent" />
                    <circle cx={p[0]} cy={p[1]} r={r} fill="var(--ink)" stroke="var(--bg)" strokeWidth={2}
                            vectorEffect="non-scaling-stroke" />
                    <text x={p[0]} y={p[1]} dy="0.35em" textAnchor="middle" fontSize={r * 1.15}
                          fontFamily="IBM Plex Sans Condensed" fontWeight={600} fill="var(--bg)">{i + 1}</text>
                  </g>
                ))}
              </svg>
            )}
          </div>
        </div>

        <aside className="lg:col-span-4 flex flex-col gap-5">
          <section className="card card-pad flex flex-col gap-4">
            <p className="t-body text-ink max-w-[46ch]">
              {remaining > 0
                ? `Click the ${remaining === 4 ? 'four corners' : `${remaining} remaining ${remaining === 1 ? 'corner' : 'corners'}`} of the seat, in any order.`
                : 'All four corners placed. Drag a corner to adjust, then start monitoring.'}
            </p>
            <p className="t-label text-ink2 max-w-[46ch]">
              Trace the seat as the camera sees it, not a rectangle. The system tracks the patient inside
              this area and uses its depth as the floor reference for falls.
            </p>

            <Tip term="furniture" asChild>
              <div role="radiogroup" aria-label="Furniture type" className="inline-flex w-fit rounded-lg border border-line p-0.5">
                {(['chair', 'bed'] as const).map((l) => (
                  <button key={l} type="button" role="radio" aria-checked={label === l} onClick={() => setLabel(l)}
                          className={`hit h-8 px-4 rounded-md t-label capitalize ${label === l ? 'bg-surface2 text-ink' : 'text-ink2 hover:text-ink'}`}>
                    {l}
                  </button>
                ))}
              </div>
            </Tip>

            <div className="flex flex-wrap gap-2">
              <Tip title="Undo" content="Remove the last corner you placed." asChild>
                <button type="button" className="btn" disabled={!pts.length} onClick={() => setPts((s) => s.slice(0, -1))}>
                  <Undo2 className="h-4 w-4" aria-hidden />Undo
                </button>
              </Tip>
              <Tip title="Clear" content="Remove all corners and start again." asChild>
                <button type="button" className="btn" disabled={!pts.length} onClick={() => setPts([])}>
                  <RotateCcw className="h-4 w-4" aria-hidden />Clear
                </button>
              </Tip>
              <Tip title="New frame" content="Take a fresh still from the camera, for example after moving the chair. Faces are blurred." asChild>
                <button type="button" className="btn" onClick={() => { setImgError(false); setNonce((n) => n + 1); }}>
                  <RefreshCw className="h-4 w-4" aria-hidden />New frame
                </button>
              </Tip>
            </div>

            <Tip title="Start monitoring" asChild
                 content="Saves the four corners; the server measures the seat’s distance and monitoring starts. Needs all four corners.">
              <button type="button" className="btn btn-primary h-11 justify-center" disabled={pts.length !== 4 || saving}
                      onClick={() => void save()}>
                <Check className="h-4 w-4" aria-hidden />{saving ? 'Saving' : 'Start monitoring'}
              </button>
            </Tip>

            {status?.has_saved && (
              <Tip term="reuseCalibration" asChild>
                <button type="button" className="btn btn-ghost justify-center" disabled={saving} onClick={() => void reuse()}>
                  Use last calibration instead
                </button>
              </Tip>
            )}
            {error && <p className="t-label text-red" role="alert">{error}</p>}
          </section>

          {status?.calibrated && status.seat_depth_m != null && (
            <p className="t-label text-ink2 px-1">
              <Tip term="seatDepth">Current seat is {status.seat_depth_m.toFixed(2)} m from the camera.</Tip>
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
