// System.tsx — server health: what is running, what is missing, and how to fix it.
import { Link } from 'react-router-dom';
import ModuleHealth from '../components/ModuleHealth';
import StatusBadge from '../components/StatusBadge';
import { useHealth } from '../hooks/usePatientData';
import { duration, STATUS_TEXT } from '../lib/format';

export default function System() {
  const { data: h, error } = useHealth(3000);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2 pt-2">
        <h1 className="font-cond font-semibold text-[40px] sm:text-[56px] leading-[0.95] tracking-[-0.02em]">
          {!h ? 'Checking the system' : h.status === 'ok' ? 'All systems running.' : 'Running with gaps.'}
        </h1>
        {h && (
          <p className="t-lead text-ink2">
            Version {h.version}, {h.mode === 'demo' ? 'demo mode with simulated beds' : 'camera mode'}, up for {duration(h.uptime_s)}
          </p>
        )}
        {error && <p className="t-body text-red">{error}</p>}
      </header>

      <div className="grid gap-5 lg:grid-cols-2">
        {h?.patients.map((p) => (
          <section key={p.patient_id} className="panel p-5 flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="t-head">
                <Link className="hover:underline underline-offset-4" to={`/patients/${encodeURIComponent(p.patient_id)}`}>{p.patient_id}</Link>
              </h2>
              <StatusBadge tone={p.status === 'running' ? 'neutral' : 'amber'}>{STATUS_TEXT[p.status]}</StatusBadge>
              <span className="t-micro text-mute tnum ml-auto">
                {p.metrics?.fps ? `${p.metrics.fps} fps, ${Math.round(p.metrics.latency_ms ?? 0)} ms per frame` : ''}
              </span>
            </div>
            {p.status_detail && <p className="t-label text-ink2">{p.status_detail}</p>}
            <ModuleHealth modules={p.modules} />
          </section>
        ))}
      </div>

      <section className="panel p-5 flex flex-col gap-3 max-w-[80ch]">
        <h2 className="t-head">Running VisionICU</h2>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[auto,1fr] t-label">
          <dt className="text-ink2">With the camera</dt><dd><code className="text-ink">python main.py</code></dd>
          <dt className="text-ink2">Without hardware</dt><dd><code className="text-ink">python main.py --demo</code></dd>
          <dt className="text-ink2">From another PC</dt><dd><code className="text-ink">python main.py --host 0.0.0.0</code></dd>
          <dt className="text-ink2">Skip chair setup</dt><dd><code className="text-ink">python main.py --reuse-calibration</code></dd>
        </dl>
        <p className="t-micro text-mute">
          Pose needs models/public/yolov8n-pose.pt. Fall needs models/public/fall_gru_pseudo.pt.
          Missing weights switch that module off; everything else keeps running.
        </p>
      </section>
    </div>
  );
}
