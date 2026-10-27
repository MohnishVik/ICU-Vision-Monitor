// Home.tsx — what VisionICU is, what it reads, and how to run it. Written for the
// person seeing the monitor for the first time; the live ward sits beside the
// explanation so the claims can be checked against the real feed.
import { ArrowRight, ArrowUpRight, Camera, Crosshair, Thermometer } from 'lucide-react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/Logo';
import { Experimental, Quality } from '../components/ui';
import { useWard } from '../context/WardContext';
import type { PatientSummary } from '../types';

// ------------------------------------------------------------------ live ward preview

function bedWord(p: PatientSummary): { text: string; tone: string } {
  if (p.top_alarm?.state === 'active') {
    return { text: p.top_alarm.title, tone: p.top_alarm.severity === 'red' ? 'text-red' : 'text-amber' };
  }
  if (p.status === 'awaiting_calibration') return { text: 'Waiting for chair setup', tone: 'text-ink2' };
  if (p.status === 'camera_offline') return { text: 'Camera not connected', tone: 'text-ink2' };
  if (p.status === 'error') return { text: 'Pipeline error', tone: 'text-red' };
  if (p.status === 'starting') return { text: 'Starting', tone: 'text-ink2' };
  if (p.ews.level === 'amber' || p.ews.level === 'red') return { text: 'Watch closely', tone: 'text-amber' };
  return { text: 'Stable', tone: 'text-ink2' };
}

function Num({ v, tone }: { v: PatientSummary['hr']; tone: 'text-hr' | 'text-rr' }) {
  const shown = v.value !== null && !v.stale;
  return (
    <span className={`font-cond font-semibold text-[26px] leading-none tnum ${shown ? tone : 'text-mute'}`}>
      {shown ? Math.round(v.value as number) : '–'}
    </span>
  );
}

function LiveWard() {
  const { ward, link, error } = useWard();

  if (!ward) {
    return (
      <div className="card card-pad flex flex-col gap-3">
        <p className="t-head">Waiting for the monitor server</p>
        <p className="t-label text-ink2 max-w-[48ch]">
          The page is up, but nothing is answering on the API yet. Start the server in a terminal
          and this panel fills in by itself.
        </p>
        <code className="well px-3 py-2 w-fit text-ink">python main.py --demo</code>
        {error && <p className="t-micro text-mute break-words">{error}</p>}
      </div>
    );
  }

  return (
    <section aria-labelledby="live-h" className="card overflow-hidden">
      <div className="flex items-center gap-3 px-5 pt-4 pb-3 rule-b">
        <h2 id="live-h" className="t-head">On the ward now</h2>
        <span className="ml-auto inline-flex items-center gap-2 t-micro text-mute">
          <span className={`h-2 w-2 rounded-full ${link === 'live' ? 'bg-hr' : 'bg-amber'}`} aria-hidden />
          {link === 'live' ? 'Live, updates every second' : 'Reconnecting'}
        </span>
      </div>
      <ul>
        {ward.map((p) => {
          const w = bedWord(p);
          return (
            <li key={p.patient_id} className="rule-b last:border-b-0">
              <Link to={`/patients/${p.patient_id}`}
                    className="grid grid-cols-[1fr_auto_auto_20px] items-center gap-x-5 px-5 py-3.5 hover:bg-surface2 transition-colors">
                <span className="min-w-0">
                  <span className="block font-cond font-semibold text-[17px]">{p.label}</span>
                  <span className={`block t-micro truncate ${w.tone}`}>{w.text}</span>
                </span>
                <span className="flex items-baseline gap-1.5" aria-label="Heart rate">
                  <Num v={p.hr} tone="text-hr" /><span className="t-micro text-mute">bpm</span>
                </span>
                <span className="flex items-baseline gap-1.5" aria-label="Breathing rate">
                  <Num v={p.rr} tone="text-rr" /><span className="t-micro text-mute">br/min</span>
                </span>
                <ArrowRight className="h-4 w-4 text-mute" aria-hidden />
              </Link>
            </li>
          );
        })}
      </ul>
      {ward.some((p) => p.simulated) && (
        <p className="px-5 py-3 t-micro text-mute rule-t">
          These beds are simulated (the server was started with <code>--demo</code>). With a camera
          connected, one real bed appears here instead.
        </p>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ content

const READS: { name: string; swatch?: string; how: ReactNode; role: ReactNode }[] = [
  {
    name: 'Heart rate',
    swatch: 'var(--hr)',
    how: <>Each heartbeat pushes a little more blood into the face, which makes the skin very slightly
      redder for a moment. The colour camera tracks that change across thousands of pixels and
      counts the beats per minute.</>,
    role: 'Raises alarms',
  },
  {
    name: 'Breathing rate',
    swatch: 'var(--rr)',
    how: <>The depth camera sees the chest move by a few millimetres with each breath. The thermal
      camera, when connected, sees the air below the nose warm on every breath out. Both are
      counted, then cross-checked.</>,
    role: 'Raises alarms',
  },
  {
    name: 'Falls and chair exits',
    how: <>A pose model finds 17 points on the body (head, shoulders, hips, knees). A sudden drop
      followed by lying still is a fall; standing up from the chair is an exit.</>,
    role: 'Raises alarms',
  },
  {
    name: 'Breathing synchrony',
    how: <>When breathing is easy, the chest and belly rise together. When it is laboured they start
      to see-saw. The depth camera compares the two.</>,
    role: 'Flags on the bed view',
  },
  {
    name: 'Pain expression',
    how: <>A face model estimates how likely the current expression is to be pain. It was trained on
      synthetic faces, so it is shown for research and never raises an alarm.</>,
    role: <Experimental />,
  },
];

const PIPELINE: { step: string; text: string }[] = [
  { step: 'Capture', text: 'Colour and depth at 30 frames a second, thermal at about 8.' },
  { step: 'Find the patient', text: 'Lock on to the person in the chair and ignore staff walking past.' },
  { step: 'Measure', text: 'Each method gives a number plus how much it trusts that number.' },
  { step: 'Combine', text: 'Readings are weighted by trust and smoothed so one bad frame cannot jump the value.' },
  { step: 'Warn', text: 'A NEWS2-style score, then highlight, chime and escalate if nobody responds.' },
];

const STEPS: { title: string; body: ReactNode; link?: { to: string; label: string } }[] = [
  {
    title: 'Point the cameras at the bed',
    body: <>Plug the RealSense D435 into a USB 3 port, about 1.5–2 m from the patient at chest-to-face
      height. Turn on the thermal camera and join its Wi-Fi if you are using it.</>,
  },
  {
    title: 'Start the monitor',
    body: <>Run <code>python main.py</code> from the project folder. With no cameras, run{' '}
      <code>python main.py --demo</code> to get four simulated beds. Then open this page at{' '}
      <code>http://localhost:8000</code>.</>,
  },
  {
    title: 'Show it where the chair is',
    body: <>Open the bed and choose <em>Set up chair</em>. Click the four corners of the chair or bed;
      fall and exit detection use this outline.</>,
    link: { to: '/ward', label: 'Open the ward' },
  },
  {
    title: 'Line up the thermal camera',
    body: <>Optional. Choose <em>Align thermal</em> and click the same few points in both pictures, so
      breathing is read at the nostrils instead of the warmest spot.</>,
  },
  {
    title: 'Watch, and let it call you',
    body: <>Turn on alarm sound (browsers keep pages silent until you allow it). The ward page puts
      the bed that needs you first.</>,
  },
  {
    title: 'Check it against a reference',
    body: <>Under <em>Record &amp; compare</em>, record a session while you note the true rates from a
      pulse oximeter or by counting. The page scores every method against your numbers.</>,
    link: { to: '/compare', label: 'Record & compare' },
  },
];

// ------------------------------------------------------------------ page

export default function Home() {
  return (
    <div className="flex flex-col gap-20 pb-10">
      {/* opening */}
      <section className="grid lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] gap-10 lg:gap-14 items-start pt-4">
        <div className="flex flex-col gap-7">
          <Logo height={132} className="unveil" />
          <div className="flex flex-col gap-4">
            <h1 className="font-cond font-semibold text-[44px] sm:text-[60px] leading-[.98] tracking-[-0.025em] max-w-[14ch] text-balance">
              Vital signs from across the room.
            </h1>
            <p className="t-body text-ink2 max-w-[56ch] text-[17.5px] leading-[1.6]">
              VisionICU watches an ICU bed through a colour, depth and thermal camera, and reads heart
              rate, breathing, falls and chair exits without a single wire on the patient. It warns the
              nurse when the numbers drift, and shows how sure it is about every one of them.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link to="/ward" className="btn btn-primary min-h-[46px] px-5 text-[15.5px]">
              Open the ward<ArrowRight className="h-4 w-4" aria-hidden />
            </Link>
            <a href="#start" className="btn min-h-[46px] px-5 text-[15.5px]">How to run it</a>
          </div>
        </div>
        <LiveWard />
      </section>

      {/* what it reads */}
      <section aria-labelledby="reads-h" className="flex flex-col gap-6">
        <div className="flex flex-col gap-2 max-w-[62ch]">
          <h2 id="reads-h" className="t-title">What it reads</h2>
          <p className="t-body text-ink2">
            Every number on the monitor comes from one of these. Heart rate is always green and
            breathing is always yellow, the same as a bedside monitor.
          </p>
        </div>
        <dl className="rule-t">
          {READS.map((r) => (
            <div key={r.name} className="grid md:grid-cols-[220px_minmax(0,1fr)_200px] gap-x-8 gap-y-2 py-5 rule-b">
              <dt className="self-start flex items-center gap-2.5 font-cond font-semibold text-[19px]">
                {r.swatch
                  ? <span className="h-2.5 w-2.5 rounded-full" style={{ background: r.swatch }} aria-hidden />
                  : <span className="h-2.5 w-2.5" aria-hidden />}
                {r.name}
              </dt>
              <dd className="t-body text-ink2 max-w-[68ch]">{r.how}</dd>
              <dd className="t-label text-mute md:text-right">{r.role}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* how a reading is made */}
      <section aria-labelledby="flow-h" className="flex flex-col gap-6">
        <div className="flex flex-col gap-2 max-w-[62ch]">
          <h2 id="flow-h" className="t-title">How a reading is made</h2>
          <p className="t-body text-ink2">
            Like a nurse who glances at the patient, counts, compares with the last count, and only
            then decides whether to worry. Every second, each frame goes through five stages.
          </p>
        </div>
        <ol className="grid sm:grid-cols-2 lg:grid-cols-5 gap-px bg-line border border-line rounded-[10px] overflow-hidden">
          {PIPELINE.map((s, i) => (
            <li key={s.step} className="bg-surface p-5 flex flex-col gap-2">
              <span className="flex items-center gap-2 font-cond font-semibold text-[17px]">
                {s.step}
                {i < PIPELINE.length - 1 && <ArrowRight className="h-4 w-4 text-mute hidden lg:block ml-auto" aria-hidden />}
              </span>
              <span className="t-label text-ink2">{s.text}</span>
            </li>
          ))}
        </ol>
        <div className="flex flex-wrap gap-x-8 gap-y-3 t-label text-ink2">
          <span className="inline-flex items-center gap-2"><Camera className="h-4 w-4 text-mute" aria-hidden />Intel RealSense D435: colour + depth</span>
          <span className="inline-flex items-center gap-2"><Thermometer className="h-4 w-4 text-mute" aria-hidden />Waveshare ESP32-S3: 80 × 62 thermal</span>
          <span className="inline-flex items-center gap-2"><Crosshair className="h-4 w-4 text-mute" aria-hidden />YOLOv8 pose, POS rPPG, Kalman fusion</span>
        </div>
      </section>

      {/* getting started */}
      <section id="start" aria-labelledby="start-h" className="flex flex-col gap-6 scroll-mt-24">
        <div className="flex flex-col gap-2 max-w-[62ch]">
          <h2 id="start-h" className="t-title">Running it, start to finish</h2>
          <p className="t-body text-ink2">Six steps. Only the first two are needed to see it working.</p>
        </div>
        <ol className="grid md:grid-cols-2 gap-x-12">
          {STEPS.map((s, i) => (
            <li key={s.title} className="grid grid-cols-[36px_minmax(0,1fr)] gap-x-3 py-5 rule-t">
              <span className="font-cond font-semibold text-[22px] leading-[1.15] text-brand tnum">{i + 1}</span>
              <div className="flex flex-col gap-1.5">
                <h3 className="font-cond font-semibold text-[18px]">{s.title}</h3>
                <p className="t-label text-ink2 leading-[1.6]">{s.body}</p>
                {s.link && (
                  <Link to={s.link.to} className="t-label text-ink inline-flex items-center gap-1 w-fit underline hover:text-brand">
                    {s.link.label}<ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
                  </Link>
                )}
              </div>
            </li>
          ))}
        </ol>
      </section>

      {/* reading the screen */}
      <section aria-labelledby="legend-h" className="flex flex-col gap-6">
        <div className="flex flex-col gap-2 max-w-[62ch]">
          <h2 id="legend-h" className="t-title">Reading the screen</h2>
          <p className="t-body text-ink2">Every colour and mark has one job, and is always backed by words.</p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-x-8 gap-y-6">
          <Legend sample={<span className="font-cond font-semibold text-[30px] leading-none text-hr tnum">72</span>}
                  title="Green and yellow" text="Heart rate and breathing rate. Identity, not good or bad." />
          <Legend sample={<span className="chip band-amber">Watch closely</span>}
                  title="Orange" text="Drifting out of range. Look more often." />
          <Legend sample={<span className="chip band-red">Fall detected</span>}
                  title="Red" text="Act now. Red alarms escalate until someone acknowledges them." />
          <Legend sample={<Quality sqi={0.3} />}
                  title="Signal bars" text="How much the camera trusts the reading. Below 40% they turn orange." />
          <Legend sample={<span className="hatch block h-7 w-24 rounded" />}
                  title="Hatched area" text="Cannot measure right now (face turned, out of frame). Never shown as a number." />
          <Legend sample={<Experimental />}
                  title="Flask, dashed line" text="A learned model still being validated. Shown, but never used for alarms." />
        </div>
      </section>

      {/* limits */}
      <section aria-labelledby="limits-h" className="rule-t pt-8 grid md:grid-cols-[220px_minmax(0,1fr)] gap-x-8 gap-y-3">
        <h2 id="limits-h" className="font-cond font-semibold text-[19px]">What it is not</h2>
        <div className="flex flex-col gap-2 t-label text-ink2 max-w-[72ch] leading-[1.6]">
          <p>
            A research prototype, not a certified medical device. It does not replace a bedside monitor
            or a nurse’s judgement. The early-warning score uses only the camera’s heart and breathing
            rates; oxygen saturation, temperature, blood pressure and consciousness are not measured.
          </p>
          <p>No video or images are saved. Recorded sessions keep only numbers, one row per second.</p>
        </div>
      </section>
    </div>
  );
}

function Legend({ sample, title, text }: { sample: ReactNode; title: string; text: string }) {
  return (
    <div className="flex flex-col gap-2.5">
      <div className="h-9 flex items-center">{sample}</div>
      <p className="font-cond font-semibold text-[16px]">{title}</p>
      <p className="t-label text-ink2">{text}</p>
    </div>
  );
}
