// AppShell.tsx — top bar, global alarm bar, page frame.
import { Activity, LayoutGrid, Server, Volume2, VolumeX } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { useWard } from '../context/WardContext';
import { enableSound } from '../lib/sound';
import AlarmNotification from './AlarmNotification';
import StatusBadge from './StatusBadge';

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const h = window.setInterval(() => setNow(new Date()), 15000);
    return () => window.clearInterval(h);
  }, []);
  return now;
}

const navCls = ({ isActive }: { isActive: boolean }) =>
  `inline-flex items-center gap-2 h-9 px-3 rounded-lg t-label transition-colors ${
    isActive ? 'bg-raised text-ink' : 'text-ink2 hover:text-ink hover:bg-raised/60'}`;

export default function AppShell({ children }: { children: ReactNode }) {
  const { ward, link } = useWard();
  const [soundOn, setSoundOn] = useState(false);
  const now = useClock();
  const simulated = !!ward?.some((p) => p.simulated);

  const toggleSound = () => {
    if (soundOn) setSoundOn(false);
    else setSoundOn(enableSound());
  };

  return (
    <div className="min-h-full flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 btn">
        Skip to content
      </a>
      <header className="sticky top-0 z-20 bg-ward/95 backdrop-blur rule-b">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 sm:px-6 py-2.5">
          <NavLink to="/" className="flex items-center gap-2.5 mr-2" aria-label="VisionICU ward view">
            <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
              <rect width="32" height="32" rx="7" fill="var(--panel)" />
              <path d="M4 17h6l2.5-6 4 12 3-9 1.5 3H28" fill="none" stroke="var(--hr)" strokeWidth="2.4"
                    strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="font-semi font-semibold text-[21px] tracking-[-0.01em] text-ink">VisionICU</span>
          </NavLink>

          <nav aria-label="Main" className="flex items-center gap-1">
            <NavLink to="/" end className={navCls}><LayoutGrid className="h-4 w-4" aria-hidden />Ward</NavLink>
            <NavLink to="/system" className={navCls}><Server className="h-4 w-4" aria-hidden />System</NavLink>
          </nav>

          <div className="ml-auto flex flex-wrap items-center gap-2">
            {simulated && (
              <StatusBadge tone="demo" title="Started with --demo: no camera, simulated patients">
                Demo, simulated data
              </StatusBadge>
            )}
            <StatusBadge tone={link === 'live' ? 'neutral' : 'amber'}
                         icon={<Activity className={link === 'live' ? 'text-hr' : ''} />}>
              {link === 'live' ? 'Live' : link === 'connecting' ? 'Connecting' : 'Reconnecting'}
            </StatusBadge>
            <button type="button" onClick={toggleSound} className="btn btn-ghost h-8 px-2.5"
                    aria-pressed={soundOn}
                    title={soundOn ? 'Alarm tones on' : 'Browsers block sound until you allow it'}>
              {soundOn ? <Volume2 className="h-4 w-4" aria-hidden /> : <VolumeX className="h-4 w-4 text-amber" aria-hidden />}
              {soundOn ? 'Sound on' : 'Turn on alarm sound'}
            </button>
            <time className="t-label text-ink2 tnum pl-1" dateTime={now.toISOString()}>
              {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </time>
          </div>
        </div>
      </header>

      <AlarmNotification ward={ward} soundOn={soundOn} />

      <main id="main" className="flex-1 px-4 sm:px-6 py-5 w-full max-w-[1600px] mx-auto">
        {children}
      </main>
    </div>
  );
}
