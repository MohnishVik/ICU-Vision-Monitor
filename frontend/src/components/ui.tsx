// ui.tsx — small shared pieces: chips, signal-quality meter, experimental tag, section heading.
// Each piece that carries a meaning also carries its explanation (Tooltip.tsx, lib/glossary.ts).
import { FlaskConical } from 'lucide-react';
import type { ReactNode } from 'react';
import type { GlossKey } from '../lib/glossary';
import { InfoTip, Tip } from './Tooltip';

export type ChipTone = 'neutral' | 'amber' | 'red' | 'ok' | 'sim';

const TONE: Record<ChipTone, string> = {
  neutral: '',
  amber: 'band-amber',
  red: 'band-red',
  ok: 'text-ok',
  sim: 'border-dashed',
};

/** Icon + text pill. Colour never carries meaning alone; `term` / `tip` explain it on hover and focus. */
export function Chip({ tone = 'neutral', icon, children, term, tip }: {
  tone?: ChipTone; icon?: ReactNode; children: ReactNode; term?: GlossKey; tip?: ReactNode;
}) {
  const body = (
    <>
      {icon && <span aria-hidden className="inline-flex">{icon}</span>}
      {children}
    </>
  );
  if (!term && !tip) return <span className={`chip ${TONE[tone]}`}>{body}</span>;
  return <Tip term={term} content={tip} className={`chip ${TONE[tone]}`}>{body}</Tip>;
}

/** Marks anything produced by a learned model that has not been validated yet. */
export function Experimental({ compact = false }: { compact?: boolean }) {
  return (
    <Tip term="learned" className="chip border-dashed">
      <FlaskConical aria-hidden />{compact ? 'Exp.' : 'Experimental'}
    </Tip>
  );
}

/** Signal quality 0–1 as five rising bars + a number. Orange below 40 %. */
export function Quality({ sqi, label = 'Signal', compact = false, term = 'sqi' }: {
  sqi: number | null | undefined; label?: string; compact?: boolean; term?: GlossKey;
}) {
  const q = sqi === null || sqi === undefined || !Number.isFinite(sqi) ? 0 : Math.max(0, Math.min(1, sqi));
  const lit = Math.round(q * 5);
  const weak = q > 0 && q < 0.4;
  const now = q > 0 ? `Now ${Math.round(q * 100)} %${weak ? ', below 40 %: less certain.' : '.'}` : 'Now: none.';
  return (
    <Tip term={term} content={now} asChild>
    <span className="tip-target inline-flex items-center gap-2" role="meter" aria-label={`${label} quality`} tabIndex={0}
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(q * 100)}>
      <span className="flex items-end gap-[2px] h-3.5" aria-hidden>
        {[0, 1, 2, 3, 4].map((i) => (
          <span key={i} className="w-[4px] rounded-[1px]"
                style={{ height: `${40 + i * 15}%`,
                         background: i < lit ? (weak ? 'var(--amber)' : 'var(--ink-2)') : 'var(--line-strong)' }} />
        ))}
      </span>
      {!compact && (
        <span className={`t-micro tnum ${weak ? 'text-amber' : 'text-mute'}`}>
          {label} {q > 0 ? `${Math.round(q * 100)}%` : 'none'}
        </span>
      )}
    </span>
    </Tip>
  );
}

/** Section heading, with an optional (i) that explains what the section shows. */
export function SectionHead({ id, title, info, children }: {
  id?: string; title: string; info?: GlossKey; children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
      <div className="flex items-center gap-2 min-w-0">
        <h2 id={id} className="t-head text-ink min-w-0">{title}</h2>
        {info && <InfoTip term={info} />}
      </div>
      {children && <div className="ml-auto flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

/** Horizontal 0..1 bar used in the comparison tables. */
export function Bar({ value, max = 1, tone = 'ink', label }: {
  value: number | null; max?: number; tone?: 'ink' | 'amber' | 'hr' | 'rr'; label: string;
}) {
  const frac = value === null ? 0 : Math.max(0, Math.min(1, value / max));
  const color = tone === 'amber' ? 'var(--amber)' : tone === 'hr' ? 'var(--hr-line)' : tone === 'rr' ? 'var(--rr-line)'
    : 'var(--ink-2)';
  return (
    <span className="block h-1.5 w-full rounded-full bg-surface3 overflow-hidden" role="img" aria-label={label}>
      <span className="block h-full rounded-full" style={{ width: `${frac * 100}%`, background: color }} />
    </span>
  );
}
