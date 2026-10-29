// ThermalPanel.tsx — live ESP32 thermal view with the breathing region and its signal.
//
// The breath-temperature trace is the raw signal the thermal RR path analyses:
// air warms the nostrils on each exhale and cools them on each inhale.
import { Crosshair, ThermometerSnowflake, TriangleAlert } from 'lucide-react';
import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { useThermal } from '../hooks/usePatientData';
import type { PatientSnapshot } from '../types';
import ThermalCanvas, { robustRange, ThermalLegend } from './ThermalCanvas';
import { InfoTip, Tip } from './Tooltip';

const QUALITY_TEXT = { good: 'good', fair: 'fair', poor: 'poor, realign', unchecked: 'unchecked' } as const;

export default function ThermalPanel({ snap }: { snap: PatientSnapshot }) {
  const { data: th } = useThermal(snap.patient_id);
  const grid = th && th.online && th.data && th.w && th.h
    ? { w: th.w, h: th.h, data: th.data, min: th.min ?? 0, max: th.max ?? 0 } : null;
  // keyed on the polled payload, not the derived object, so the sort runs once per frame
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const range = useMemo(() => (grid ? robustRange(grid.data) : ([20, 37] as [number, number])), [th]);
  const trace = useMemo(() => (th?.trace?.t ?? []).map((t, i) => ({ t, c: th!.trace.c[i] })), [th]);
  const reg = th?.registration;
  const atNostrils = th?.roi_source === 'nostril';
  const alignHref = `/patients/${encodeURIComponent(snap.patient_id)}/thermal`;

  return (
    <section className="card card-pad flex flex-col gap-4" aria-labelledby="thermal-h">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <h2 id="thermal-h" className="t-head">Thermal camera</h2>
          <InfoTip term="thermal" />
        </div>
        <span className="t-micro text-mute">ESP32, 80 × 62 sensor</span>
        <Tip term="thermalAlign" asChild>
          <Link to={alignHref} className="btn btn-ghost h-8 px-2 ml-auto">
            <Crosshair className="h-4 w-4" aria-hidden />{reg?.registered ? 'Realign' : 'Align with colour camera'}
          </Link>
        </Tip>
      </div>

      {!grid ? (
        <div className="flex gap-3 items-start rounded-lg hatch p-4">
          <ThermometerSnowflake className="h-5 w-5 text-mute shrink-0 mt-0.5" aria-hidden />
          <div className="flex flex-col gap-1">
            <p className="t-body text-ink">{th ? 'Thermal camera offline.' : 'Connecting to the thermal camera.'}</p>
            {th?.link_detail && <p className="t-label text-ink max-w-[58ch]">{th.link_detail}</p>}
            <p className="t-label text-ink2 max-w-[58ch]">
              Respiratory rate continues from the depth camera alone. Check that the ESP32 is powered
              and on the same Wi-Fi as this computer (it answers at 192.168.4.1).
            </p>
          </div>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="flex flex-col gap-2">
            <ThermalCanvas grid={grid} range={range} roi={th?.roi}
                           label={`Live thermal image. Range ${range[0].toFixed(1)} to ${range[1].toFixed(1)} degrees.`} />
            <Tip term="thermalScale" className="block"><ThermalLegend lo={range[0]} hi={range[1]} /></Tip>
          </div>

          <div className="flex flex-col gap-3 min-w-0">
            <Tip term="breathTrace" className="flex flex-col gap-1">
              <span className="t-label text-ink2">
                {atNostrils ? 'Breath temperature at the nostrils' : 'Temperature at the warmest spot'}
              </span>
              <span className="t-micro text-mute">last 30 s; warms on each exhale, cools on each inhale</span>
            </Tip>
            <div className="h-[96px] rounded-lg border border-line" role="img"
                 aria-label="Breath temperature over the last 30 seconds">
              {trace.length > 4 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trace} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <YAxis hide domain={['dataMin - 0.15', 'dataMax + 0.15']} />
                    <Tooltip isAnimationActive={false} cursor={{ stroke: 'var(--ink-2)', strokeWidth: 1 }}
                             content={({ active, payload }) => active && payload?.length ? (
                               <div className="rounded-md border border-line bg-surface2 px-2 py-1 t-micro tnum">
                                 {(payload[0].value as number).toFixed(2)} °C
                               </div>) : null} />
                    <Line type="monotone" dataKey="c" stroke="var(--rr-line)" strokeWidth={2} dot={false}
                          isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="t-micro text-mute p-3">Collecting signal.</p>
              )}
            </div>

            {reg?.registered ? (
              <p className="t-label text-ink2">
                {atNostrils
                  ? 'Aligned with the colour camera; breathing is read at the nostrils.'
                  : 'Aligned, but no face in view right now; using the warmest spot.'}
                <span className="flex">
                  <Tip term="alignQuality" className="t-micro text-mute tnum">
                    Alignment {QUALITY_TEXT[reg.quality ?? 'unchecked']}
                    {reg.error_px != null && reg.n_pairs > 4 ? `, average error ${reg.error_px.toFixed(1)} px` : ''}
                    {reg.source === 'camera.yaml' ? ', from camera.yaml' : ''}
                  </Tip>
                </span>
              </p>
            ) : (
              <p className="t-label text-amberink flex gap-2 items-start">
                <TriangleAlert className="h-4 w-4 shrink-0 mt-0.5" aria-hidden />
                Not aligned yet: breathing is read from the warmest spot, which may be the forehead
                rather than the nose. Align once to read the nostrils.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
