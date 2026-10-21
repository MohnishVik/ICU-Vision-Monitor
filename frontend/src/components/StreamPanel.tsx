// StreamPanel.tsx — live, face-blurred video with a vector overlay.
//
// The server streams MJPEG with every detected face blurred (src/scene/privacy.py).
// The overlay (chair polygon, patient skeleton, staff boxes) is drawn here in SVG
// from the keypoints in the snapshot, so it stays crisp and can be toggled.
import { CameraOff, Crosshair, EyeOff, Layers, ShieldCheck, ShieldAlert } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { urls } from '../lib/api';
import type { PatientSnapshot } from '../types';

const BONES: [number, number][] = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

export default function StreamPanel({ snap }: { snap: PatientSnapshot }) {
  const [overlay, setOverlay] = useState(true);
  const [nonce, setNonce] = useState(0);
  const [broken, setBroken] = useState(false);
  const scene = snap.scene;
  const [h, w] = scene?.image_size ?? [480, 640];
  const offline = snap.status === 'camera_offline' || snap.status === 'error' || snap.status === 'starting';
  const needsSetup = snap.status === 'awaiting_calibration';

  useEffect(() => {
    if (!broken) return;
    const t = window.setTimeout(() => {
      setBroken(false);
      setNonce((n) => n + 1);
    }, 2000);
    return () => window.clearTimeout(t);
  }, [broken]);

  const kp = scene?.keypoints;
  const conf = scene?.keypoints_conf;
  const ok = (i: number) =>
    !!kp && Array.isArray(kp[i]) && kp[i].length === 2 && (!conf || (conf[i] ?? 0) >= 0.3);

  return (
    <figure className="panel overflow-hidden m-0">
      <div className="relative bg-black" style={{ aspectRatio: `${w} / ${h}` }}>
        {!offline && !broken && (
          <img
            key={nonce}
            src={`${urls.stream(snap.patient_id)}?n=${nonce}`}
            alt={`Live camera view of ${snap.label}, faces blurred`}
            className="absolute inset-0 h-full w-full object-contain"
            onError={() => setBroken(true)}
          />
        )}

        {overlay && !offline && (
          <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 h-full w-full" aria-hidden>
            {scene?.chair && (
              <polygon
                points={scene.chair.map((p) => p.join(',')).join(' ')}
                fill="rgba(92,198,208,.08)"
                stroke="var(--scrub)"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {scene?.others?.map((b, i) => (
              <g key={i}>
                <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} rx={6}
                      fill="none" stroke="var(--mute)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                <text x={b[0] + 6} y={b[1] + 18} fill="var(--ink-2)" fontSize={14}
                      fontFamily="Barlow Semi Condensed">Staff</text>
              </g>
            ))}
            {scene?.thermal_registered && scene?.nostril_box && (
              <g>
                <rect x={scene.nostril_box[0]} y={scene.nostril_box[1]}
                      width={scene.nostril_box[2] - scene.nostril_box[0]}
                      height={scene.nostril_box[3] - scene.nostril_box[1]} rx={3}
                      fill="rgba(194,174,255,.12)" stroke="var(--rr)" strokeWidth={2}
                      vectorEffect="non-scaling-stroke" />
                <text x={scene.nostril_box[2] + 6} y={scene.nostril_box[3]} fill="var(--rr)" fontSize={13}
                      fontFamily="Barlow Semi Condensed">Breathing</text>
              </g>
            )}
            {kp && (
              <g stroke="var(--ink)" strokeWidth={2.5} strokeLinecap="round" vectorEffect="non-scaling-stroke">
                {BONES.filter(([a, b]) => ok(a) && ok(b)).map(([a, b]) => (
                  <line key={`${a}-${b}`} x1={kp[a][0]} y1={kp[a][1]} x2={kp[b][0]} y2={kp[b][1]}
                        vectorEffect="non-scaling-stroke" opacity={0.85} />
                ))}
                {kp.map((p, i) => ok(i) && (
                  <circle key={i} cx={p[0]} cy={p[1]} r={3.5} fill="var(--ink)" stroke="none" />
                ))}
              </g>
            )}
          </svg>
        )}

        {offline && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center">
            <div className="flex flex-col items-center gap-3 max-w-[46ch]">
              <CameraOff className="h-8 w-8 text-mute" aria-hidden />
              <p className="t-head text-ink">
                {snap.status === 'starting' ? 'Starting the camera' : 'Camera not connected'}
              </p>
              <p className="t-label text-ink2">{snap.status_detail || 'Waiting for the D435.'}</p>
            </div>
          </div>
        )}

        {needsSetup && (
          <div className="absolute inset-x-0 bottom-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="t-lead text-ink">Mark the chair before monitoring starts.</p>
              <Link to={`/patients/${encodeURIComponent(snap.patient_id)}/calibrate`} className="btn btn-primary">
                <Crosshair className="h-4 w-4" aria-hidden /> Set up chair
              </Link>
            </div>
          </div>
        )}
      </div>

      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5 rule-t">
        <button type="button" className="btn btn-ghost h-8 px-2" onClick={() => setOverlay((o) => !o)}
                aria-pressed={overlay}>
          {overlay ? <Layers className="h-4 w-4" aria-hidden /> : <EyeOff className="h-4 w-4" aria-hidden />}
          {overlay ? 'Hide overlay' : 'Show overlay'}
        </button>
        {!offline && <span className="t-micro text-ink2 inline-flex items-center gap-1.5">
          {scene?.redaction === 'faces' ? (
            <><ShieldCheck className="h-3.5 w-3.5 text-ink2" aria-hidden /> Faces blurred, nothing recorded</>
          ) : scene?.redaction === 'disabled' ? (
            <><EyeOff className="h-3.5 w-3.5" aria-hidden /> Video streaming turned off</>
          ) : (
            <span className="text-amber inline-flex items-center gap-1.5">
              <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
              Face blur unavailable: pose model not loaded
            </span>
          )}
        </span>}
        {snap.simulated && <span className="t-micro text-[#d8ccff]">Simulated feed, no camera</span>}
        <span className="t-micro text-mute ml-auto tnum">
          {snap.metrics?.fps ? `${snap.metrics.fps} fps` : ''}
          {snap.metrics?.latency_ms ? `  ${Math.round(snap.metrics.latency_ms)} ms per frame` : ''}
        </span>
      </figcaption>
    </figure>
  );
}
