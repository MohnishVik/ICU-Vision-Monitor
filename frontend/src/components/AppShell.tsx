// AppShell.tsx — side rail (desktop) / top bar (narrow), ward-wide alarm bar, page frame.
import { Activity, House, LayoutGrid, Moon, Server, Sun, TestTubeDiagonal, Volume2, VolumeX } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { useWard } from '../context/WardContext';
import { enableSound } from '../lib/sound';
import { useTheme } from '../lib/theme';
import AlarmBar from './AlarmBar';
import Logo from './Logo';
import { Chip } from './ui';

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const h = window.setInterval(() => setNow(new Date()), 10000);
    return () => window.clearInterval(h);
  }, []);
  return now;
}

const NAV = [
  { to: '/', label: 'Home', icon: House, end: true },
  { to: '/ward', label: 'Ward', icon: LayoutGrid, end: false },
  { to: '/compare', label: 'Record & compare', short: 'Compare', icon: TestTubeDiagonal, end: false },
  { to: '/system', label: 'System', icon: Server, end: false },
];


export default function AppShell({ children }: { children: ReactNode }) {
  const { ward, link } = useWard();
  const [soundOn, setSoundOn] = useState(false);
  const [theme, setTheme] = useTheme();
  const now = useClock();
  const simulated = !!ward?.some((p) => p.simulated);
  const toggleSound = () => (soundOn ? setSoundOn(false) : setSoundOn(enableSound()));

  const railLink = ({ isActive }: { isActive: boolean }) =>
    `group flex flex-col items-center gap-1 w-full rounded-lg py-2.5 transition-colors ${
      isActive ? 'bg-surface2 text-ink' : 'text-mute hover:text-ink hover:bg-[color-mix(in_srgb,var(--surface-2)_60%,transparent)]'}`;
  const topLink = ({ isActive }: { isActive: boolean }) =>
    `inline-flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-2 min-h-[48px] sm:min-h-[40px] px-2 sm:px-3 rounded-lg text-[13px] sm:text-[14px] transition-colors ${
      isActive ? 'bg-surface2 text-ink' : 'text-ink2 hover:text-ink'}`;

  return (
    <div className="min-h-full flex">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 btn btn-primary">
        Skip to content
      </a>

      {/* rail: desktop */}
      <aside className="hidden lg:flex sticky top-0 h-screen w-[88px] shrink-0 flex-col items-center gap-2 py-4 px-2 bg-surface rule-r border-r border-line">
        <NavLink to="/" aria-label="VisionICU home" className="mb-3 mt-1 rounded-md"><Logo height={44} label={null} /></NavLink>
        <nav aria-label="Main" className="flex flex-col gap-1 w-full">
          {NAV.map(({ to, label, short, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={railLink} title={label}>
              <Icon className="h-5 w-5" aria-hidden />
              <span className="font-cond font-semibold text-[12.5px] leading-tight text-center">{short ?? label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto flex flex-col items-center gap-1 w-full">
          <button type="button" className="btn btn-ghost w-full px-0" onClick={toggleSound} aria-pressed={soundOn}
                  title={soundOn ? 'Alarm tones on' : 'Browsers block sound until you allow it'}>
            {soundOn ? <Volume2 className="h-5 w-5" aria-hidden /> : <VolumeX className="h-5 w-5 text-amber" aria-hidden />}
            <span className="sr-only">{soundOn ? 'Alarm sound on' : 'Turn on alarm sound'}</span>
          </button>
          <button type="button" className="btn btn-ghost w-full px-0"
                  onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                  aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
            {theme === 'dark' ? <Sun className="h-5 w-5" aria-hidden /> : <Moon className="h-5 w-5" aria-hidden />}
          </button>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="sm:sticky sm:top-0 z-20 bg-[color-mix(in_srgb,var(--bg)_90%,transparent)] sm:backdrop-blur-md rule-b">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 sm:px-6 py-2.5">
            <NavLink to="/" className="flex items-center gap-2.5 rounded-md" aria-label="VisionICU home">
              <Logo height={26} label={null} className="lg:hidden" />
              <span className="font-cond font-semibold text-[20px] tracking-[-0.01em]">VisionICU</span>
            </NavLink>
            <nav aria-label="Main" className="lg:hidden grid grid-cols-4 gap-1 order-last w-full sm:flex sm:items-center sm:order-none sm:w-auto">
              {NAV.map(({ to, label, short, icon: Icon, end }) => (
                <NavLink key={to} to={to} end={end} className={topLink}>
                  <Icon className="h-4 w-4" aria-hidden />{short ?? label}
                </NavLink>
              ))}
            </nav>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {simulated && <Chip tone="sim" title="Started with --demo: no camera, simulated patients">Demo · simulated data</Chip>}
              <Chip tone={link === 'live' ? 'neutral' : 'amber'}
                    icon={<Activity className={link === 'live' ? 'text-hr' : ''} />}>
                {link === 'live' ? 'Live' : link === 'connecting' ? 'Connecting' : 'Reconnecting'}
              </Chip>
              <button type="button" onClick={toggleSound} className="lg:hidden btn btn-ghost btn-sm" aria-pressed={soundOn}>
                {soundOn ? <Volume2 className="h-4 w-4" aria-hidden /> : <VolumeX className="h-4 w-4 text-amber" aria-hidden />}
                {soundOn ? 'Sound on' : 'Sound off'}
              </button>
              <button type="button" className="lg:hidden btn btn-ghost btn-sm"
                      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                      aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}>
                {theme === 'dark' ? <Sun className="h-4 w-4" aria-hidden /> : <Moon className="h-4 w-4" aria-hidden />}
              </button>
              {!soundOn && (
                <button type="button" onClick={toggleSound} className="hidden lg:inline-flex btn btn-sm">
                  <VolumeX className="h-4 w-4 text-amber" aria-hidden />Turn on alarm sound
                </button>
              )}
              <time className="font-cond font-semibold text-[20px] tnum pl-2" dateTime={now.toISOString()}>
                {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </time>
            </div>
          </div>
        </header>

        <AlarmBar ward={ward} soundOn={soundOn} />

        <main id="main" className="flex-1 px-4 sm:px-6 py-6 w-full max-w-[1680px] mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
