// ModuleHealth.tsx — which parts of the pipeline are running, and why not if not.
import { CircleCheck, CircleDashed, CircleX, Hourglass, TriangleAlert } from 'lucide-react';
import { MODULE_TEXT } from '../lib/format';
import type { ModuleStatus } from '../types';

const ORDER = ['pose', 'rr', 'rr_learned', 'hr', 'hr_learned', 'fall', 'taa', 'pain'];

const ICON = {
  ok: <CircleCheck className="h-4 w-4 text-ink2" aria-hidden />,
  degraded: <TriangleAlert className="h-4 w-4 text-amber" aria-hidden />,
  unavailable: <CircleX className="h-4 w-4 text-amber" aria-hidden />,
  error: <CircleX className="h-4 w-4 text-red" aria-hidden />,
  pending: <Hourglass className="h-4 w-4 text-mute" aria-hidden />,
};
const WORD = { ok: 'Running', degraded: 'Degraded', unavailable: 'Not available', error: 'Error', pending: 'Pending' };

export default function ModuleHealth({ modules }: { modules: Record<string, ModuleStatus> | null | undefined }) {
  const names = Object.keys(modules ?? {}).sort((a, b) =>
    (ORDER.indexOf(a) + 99) % 99 - (ORDER.indexOf(b) + 99) % 99);
  if (!names.length) {
    return <p className="t-label text-mute inline-flex items-center gap-2"><CircleDashed className="h-4 w-4" />Loading modules</p>;
  }
  return (
    <ul className="flex flex-col">
      {names.map((n) => {
        const m = modules![n];
        const state = (m.state in ICON ? m.state : 'error') as keyof typeof ICON;
        return (
          <li key={n} className="flex items-start gap-3 py-2.5 rule-b last:border-b-0">
            <span className="mt-0.5">{ICON[state]}</span>
            <div className="flex flex-col min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="t-label text-ink">{MODULE_TEXT[n] ?? n}</span>
                <span className="t-micro text-mute">{WORD[state]}</span>
                {m.errors > 0 && <span className="t-micro text-mute tnum ml-auto">{m.errors} errors</span>}
              </div>
              {m.detail && <span className="t-micro text-ink2 break-words">{m.detail}</span>}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
