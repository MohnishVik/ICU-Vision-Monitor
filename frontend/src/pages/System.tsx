// System.tsx — server health, model files and per-bed pipeline status, with how to fix gaps.
import { CircleCheck, CircleX, FileQuestion } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ModuleHealth from '../components/ModuleHealth';
import { Chip, SectionHead } from '../components/ui';
import { useHealth } from '../hooks/usePatientData';
import { api } from '../lib/api';
import { duration, STATUS_TEXT } from '../lib/format';
import type { ModelsReport } from '../types';

function ModelFiles() {
  const [m, setM] = useState<ModelsReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.models().then(setM).catch((e: Error) => setErr(e.message)); }, []);
  return (
    <section className="card card-pad flex flex-col gap-3" aria-labelledby="models-h">
      <SectionHead id="models-h" eyebrow="models/" title="Weight files">
        {m && (m.missing_required.length
          ? <Chip tone="amber">{m.missing_required.length} required missing</Chip>
          : <Chip tone="ok" icon={<CircleCheck />}>Required files present</Chip>)}
      </SectionHead>
      {err && <p className="t-label text-red">{err}</p>}
      {m && (
        <div className="overflow-x-auto">
          <table className="data min-w-[620px]">
            <thead><tr><th>File</th><th>Used by</th><th className="num">Size</th><th>Status</th></tr></thead>
            <tbody>
              {m.files.map((f) => (
                <tr key={f.file}>
                  <td><code>{f.file}</code></td>
                  <td className="t-label text-ink2">{f.used_by}</td>
                  <td className="num t-label">{f.size_bytes ? `${Math.round(f.size_bytes / 1024).toLocaleString()} KB` : '–'}</td>
                  <td>{f.present ? <span className="inline-flex items-center gap-1.5 t-label text-ok"><CircleCheck className="h-4 w-4" aria-hidden />Present</span>
                    : f.required ? <span className="inline-flex items-center gap-1.5 t-label text-amber"><CircleX className="h-4 w-4" aria-hidden />Missing</span>
                      : <span className="inline-flex items-center gap-1.5 t-label text-mute"><FileQuestion className="h-4 w-4" aria-hidden />Optional, absent</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="t-micro text-mute">A missing file switches off only the part that needs it; everything else keeps running.
        In demo mode these files are not used.</p>
    </section>
  );
}

export default function System() {
  const { data: h, error } = useHealth(3000);
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <span className="t-eyebrow text-mute">System</span>
        <h1 className="t-display">{!h ? 'Checking the system' : h.status === 'ok' ? 'All systems running.' : 'Running with gaps.'}</h1>
        {h && <p className="t-body text-ink2">Version {h.version} · {h.mode === 'demo' ? 'demo mode with simulated beds' : 'camera mode'} · up {duration(h.uptime_s)}</p>}
        {error && <p className="t-body text-red">{error}</p>}
      </header>

      <ModelFiles />

      <div className="grid gap-5 lg:grid-cols-2">
        {h?.patients.map((p) => (
          <section key={p.patient_id} className="card card-pad flex flex-col gap-3">
            <SectionHead eyebrow="Pipeline" title={p.patient_id}>
              <Chip tone={p.status === 'running' ? 'neutral' : 'amber'}>{STATUS_TEXT[p.status]}</Chip>
              <Link className="btn btn-ghost btn-sm" to={`/patients/${encodeURIComponent(p.patient_id)}`}>Open</Link>
            </SectionHead>
            <span className="t-micro text-mute tnum">{p.metrics?.fps ? `${p.metrics.fps} fps · ${Math.round(p.metrics.latency_ms ?? 0)} ms per frame` : ''}</span>
            {p.status_detail && <p className="t-label text-ink2">{p.status_detail}</p>}
            <ModuleHealth modules={p.modules} />
          </section>
        ))}
      </div>

      <section className="card card-pad flex flex-col gap-3 max-w-[90ch]">
        <SectionHead eyebrow="Reference" title="Running VisionICU" />
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[auto,1fr] t-label">
          <dt className="text-ink2">With the camera</dt><dd><code>python main.py</code></dd>
          <dt className="text-ink2">Without hardware</dt><dd><code>python main.py --demo</code></dd>
          <dt className="text-ink2">From another PC</dt><dd><code>python main.py --host 0.0.0.0</code></dd>
          <dt className="text-ink2">Skip chair setup</dt><dd><code>python main.py --reuse-calibration</code></dd>
          <dt className="text-ink2">Score recordings</dt><dd><code>python scripts/compare_sessions.py</code></dd>
          <dt className="text-ink2">Fuse learned models</dt><dd><code>config/signal.yaml → learned: fuse: true</code> (after validating)</dd>
        </dl>
      </section>
    </div>
  );
}
