// Compare.tsx — record sessions against trusted references, then score every estimator.
//
// MAE is never shown without coverage: a source that answers 20 % of the time with a small
// error is not better than one that answers 95 % of the time with a slightly larger one.
import { Award, ClipboardList, FlaskConical, RefreshCw, Terminal } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import SessionControl from '../components/SessionControl';
import { Bar, Chip, SectionHead } from '../components/ui';
import { useWard } from '../context/WardContext';
import { api } from '../lib/api';
import { duration, fmt1, SOURCES } from '../lib/format';
import type { SessionMeta, SessionReport, SourceScore, VitalKey } from '../types';

const pctTxt = (x: number | null) => (x === null ? '–' : `${Math.round(x * 100)}%`);

function ScoreTable({ vital, scores }: { vital: VitalKey; scores: Record<string, SourceScore> }) {
  const names = Object.keys(scores).sort((a, b) => (a.endsWith('.fused') ? -1 : b.endsWith('.fused') ? 1 : a.localeCompare(b)));
  const scored = names.filter((n) => scores[n].mae !== null);
  const maxMae = Math.max(1, ...scored.map((n) => scores[n].mae as number));
  const eligible = scored.filter((n) => !n.endsWith('.fused') && (scores[n].coverage ?? 0) >= 0.5);
  const best = eligible.sort((a, b) => (scores[a].mae as number) - (scores[b].mae as number))[0];
  const unit = vital === 'rr' ? 'br/min' : 'bpm';
  const n0 = scores[names[0]]?.n_scored ?? 0;
  if (!n0) {
    return <p className="t-label text-ink2 hatch rounded-md p-3">No {vital === 'rr' ? 'breathing' : 'heart'}-rate reference in the
      scored part of this session. Add one while recording (or lower the warm-up).</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="data min-w-[860px]">
        <caption className="sr-only">{vital.toUpperCase()} estimators against the reference</caption>
        <thead>
          <tr>
            <th>{vital === 'rr' ? 'Respiratory rate' : 'Heart rate'} · source</th>
            <th className="w-[150px]">Coverage</th>
            <th className="w-[170px]">MAE ({unit})</th>
            <th className="num">Bias</th><th className="num">RMSE</th>
            <th className="num">Within ±{scores[names[0]].tolerance}</th><th className="num">r</th>
            <th className="num" title="Network output before range / quality gates">MAE unfiltered</th>
          </tr>
        </thead>
        <tbody>
          {names.map((n) => {
            const s = scores[n];
            const info = SOURCES[n];
            return (
              <tr key={n} className={n.endsWith('.fused') ? 'bg-surface2/60' : ''}>
                <td>
                  <span className="inline-flex items-center gap-2">
                    <svg width="22" height="8" aria-hidden><line x1="1" y1="4" x2="21" y2="4" stroke={info?.color ?? 'var(--ink-2)'}
                         strokeWidth="2.5" strokeDasharray={info?.dash} strokeLinecap="round" /></svg>
                    <span className="t-label t-strong">{info?.name ?? n}</span>
                    {n.endsWith('.fused') && <span className="t-micro text-mute">(what the screen showed)</span>}
                    {info?.learned && <FlaskConical className="h-3.5 w-3.5 text-mute" aria-label="experimental" />}
                    {n === best && <Chip tone="ok" icon={<Award />}>Lowest MAE</Chip>}
                  </span>
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <span className="t-label tnum w-10 text-right">{pctTxt(s.coverage)}</span>
                    <Bar value={s.coverage} label={`coverage ${pctTxt(s.coverage)}`} tone={(s.coverage ?? 0) < 0.5 ? 'amber' : 'ink'} />
                  </div>
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <span className="t-label tnum w-10 text-right t-strong">{fmt1(s.mae)}</span>
                    <Bar value={s.mae} max={maxMae} label={`mean absolute error ${fmt1(s.mae)}`} tone={vital} />
                  </div>
                </td>
                <td className="num">{s.bias === null ? '–' : `${s.bias > 0 ? '+' : ''}${s.bias.toFixed(1)}`}</td>
                <td className="num">{fmt1(s.rmse)}</td>
                <td className="num">{pctTxt(s.within_tol)}</td>
                <td className="num">{fmt1(s.pearson_r, 2)}</td>
                <td className="num text-ink2">{s.mae_unfiltered === null ? '–' : `${fmt1(s.mae_unfiltered)} · ${pctTxt(s.coverage_unfiltered)}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="t-micro text-mute mt-2 tnum">{n0} seconds scored. Bias = average of (estimate − reference); r needs ≥ 10 s and a changing reference.</p>
    </div>
  );
}

export default function Compare() {
  const { ward } = useWard();
  const [bed, setBed] = useState<string>('');
  const [sessions, setSessions] = useState<SessionMeta[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [warmup, setWarmup] = useState(30);
  const [settle, setSettle] = useState(20);
  const [report, setReport] = useState<SessionReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (!bed && ward?.length) setBed(ward[0].patient_id); }, [ward, bed]);

  const loadSessions = useCallback(() => {
    api.sessions().then((s) => { setSessions(s); setErr(null); }).catch((e: Error) => setErr(e.message));
  }, []);
  useEffect(() => { loadSessions(); const t = window.setInterval(loadSessions, 10000); return () => window.clearInterval(t); }, [loadSessions]);
  useEffect(() => {
    if (!sel && sessions?.length) setSel((sessions.find((s) => s.ended) ?? sessions[0]).id);
  }, [sessions, sel]);
  useEffect(() => {
    if (!sel) return;
    let alive = true;
    setReport(null);
    api.report(sel, warmup, settle).then((r) => alive && setReport(r)).catch((e: Error) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [sel, warmup, settle]);

  const selMeta = useMemo(() => sessions?.find((s) => s.id === sel) ?? null, [sessions, sel]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <span className="t-eyebrow text-mute">Validation</span>
        <h1 className="t-display max-w-[28ch]">Record against a reference, then see which estimator holds up.</h1>
        <p className="t-body text-ink2 max-w-[80ch]">
          Every breathing and heart-rate source, including the experimental learned models, is logged once a second
          beside the value you trust. Learned models only join the main reading after they beat the signal-processing paths here.
        </p>
      </header>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-5 flex flex-col gap-5 min-w-0">
          <section className="card card-pad flex flex-col gap-4" aria-labelledby="rec-h">
            <SectionHead id="rec-h" eyebrow="Step 1" title="Record">
              <label className="inline-flex items-center"><span className="sr-only">Bed</span>
                <select className="input w-auto" value={bed} onChange={(e) => setBed(e.target.value)} aria-label="Bed to record">
                  {(ward ?? []).map((p) => <option key={p.patient_id} value={p.patient_id}>{p.label}{p.simulated ? ' (simulated)' : ''}</option>)}
                </select>
              </label>
            </SectionHead>
            {bed && <SessionControl key={bed} patientId={bed} onStopped={(sid) => { loadSessions(); setSel(sid); }} />}
          </section>

        </div>

        <div className="xl:col-span-7 flex flex-col gap-5 min-w-0">
          <section className="card card-pad flex flex-col gap-3" aria-labelledby="sess-h">
            <SectionHead id="sess-h" eyebrow="Step 2" title="Sessions">
              <button type="button" className="btn btn-ghost btn-sm" onClick={loadSessions}><RefreshCw className="h-4 w-4" aria-hidden />Refresh</button>
            </SectionHead>
            {!sessions ? <p className="t-label text-mute">Loading sessions…</p>
              : sessions.length === 0 ? (
                <div className="flex gap-3 items-start hatch rounded-md p-4">
                  <ClipboardList className="h-5 w-5 text-mute shrink-0 mt-0.5" aria-hidden />
                  <p className="t-label text-ink2">No sessions yet. Start one on the left; it appears here when you stop it.</p>
                </div>
              ) : (
                <div className="max-h-[300px] overflow-auto rounded-md border border-line">
                  <table className="data">
                    <thead className="sticky top-0 bg-surface2"><tr><th>Started</th><th>Bed</th><th>Session</th><th className="num">Length</th></tr></thead>
                    <tbody>
                      {sessions.map((s) => (
                        <tr key={s.id} className={s.id === sel ? 'bg-surface2' : ''}>
                          <td className="tnum t-label whitespace-nowrap">
                            <button type="button" className="text-left underline-offset-4 hover:underline min-h-[32px]" onClick={() => setSel(s.id)}
                                    aria-pressed={s.id === sel}>
                              {new Date(s.started * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                            </button>
                          </td>
                          <td className="t-label">{s.patient_id}</td>
                          <td>
                            <span className="t-label t-strong block">{s.label}</span>
                            <span className="t-micro text-mute">{s.condition}{s.simulated ? ' · simulated' : ''}{s.ended ? '' : ' · recording'}</span>
                          </td>
                          <td className="num t-label">{s.ended ? duration(s.ended - s.started) : '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            {err && <p className="t-label text-red" role="alert">{err}</p>}
          </section>


        </div>
      </div>

      <section className="card card-pad flex flex-col gap-4" aria-labelledby="rep-h" aria-busy={!!sel && !report}>
            <SectionHead id="rep-h" eyebrow="Step 3" title={selMeta ? `Results · ${selMeta.label}` : 'Results'}>
              <label className="inline-flex items-center gap-2 t-label text-ink2 whitespace-nowrap" title="Seconds skipped at the start (windows filling)">Skip first
                <input className="input w-[72px] tnum" type="number" min={0} max={600} value={warmup}
                       onChange={(e) => setWarmup(Math.max(0, Number(e.target.value) || 0))} />s</label>
              <label className="inline-flex items-center gap-2 t-label text-ink2 whitespace-nowrap" title="Seconds skipped after each reference change">Settle
                <input className="input w-[72px] tnum" type="number" min={0} max={600} value={settle}
                       onChange={(e) => setSettle(Math.max(0, Number(e.target.value) || 0))} />s</label>
            </SectionHead>
            {selMeta?.simulated && <p className="t-micro text-mute">Simulated session: the numbers show how the page works, not how the models perform.</p>}
            {selMeta && !selMeta.fuse_learned && <p className="t-micro text-mute">Learned models were shown only (not fused) during this session.</p>}
            {!sel ? <p className="t-label text-mute">Pick a session.</p>
              : !report ? <p className="t-label text-mute">Scoring…</p>
                : (
                  <div className="flex flex-col gap-6">
                    <ScoreTable vital="rr" scores={report.rr} />
                    <ScoreTable vital="hr" scores={report.hr} />
                  </div>
                )}
          </section>

      <div className="grid gap-5 xl:grid-cols-12">
        <div className="xl:col-span-7">
          <section className="card card-pad flex flex-col gap-3" aria-labelledby="proto-h">
            <SectionHead id="proto-h" eyebrow="How to" title="A good validation run" />
            <ol className="flex flex-col gap-2.5 t-label text-ink2 list-decimal pl-5 marker:text-mute">
              <li><span className="text-ink t-strong">Breathing:</span> pace to a metronome app (6, 10, 15, 20, 30 br/min), 2 min each.
                Start with the first rate as the reference, then log each change as you switch.</li>
              <li><span className="text-ink t-strong">Heart rate:</span> wear a fingertip pulse oximeter; log its reading every 30–60 s
                (or when it moves by more than 3 bpm).</li>
              <li><span className="text-ink t-strong">Free breathing:</span> count breaths for a full minute and log the count.</li>
              <li>Repeat under each condition you care about: bright and dim light, sitting still and talking, thermal on and off.</li>
              <li>Keep the defaults below: the first 30 s and 20 s after each reference change are not scored, because the windows are still filling.</li>
            </ol>
          </section>
        </div>
        <div className="xl:col-span-5">
          <section className="card card-pad flex gap-3 items-start">
            <Terminal className="h-5 w-5 text-mute shrink-0 mt-0.5" aria-hidden />
            <div className="flex flex-col gap-1 min-w-0">
              <p className="t-label">Pool many sessions, grouped by condition, from the project folder:</p>
              <code className="block well px-3 py-2 overflow-x-auto whitespace-pre-wrap break-all">python scripts/compare_sessions.py --by condition --out reports/compare.md</code>
              <p className="t-micro text-mute">Sessions are stored as numbers only in data/sessions/ (git-ignored).</p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
