// ui.tsx — small shared pieces: chips, signal-quality meter, experimental tag, section heading.
import { FlaskConical } from 'lucide-react';
import type { ReactNode } from 'react';

export type ChipTone = 'neutral' | 'amber' | 'red' | 'ok' | 'sim';

const TONE: Record<ChipTone, string> = {
  neutral: '',
  amber: 'band-amber',
  red: 'band-red',
  ok: 'text-ok',
  sim: 'border-dashed',
};

/** Icon + text pill. Colour never carries meaning alone. */
export function Chip({ tone = 'neutral', icon, children, title }: {
  tone?: ChipTone; icon?: ReactNode; children: ReactNode; title?: string;
}) {
  return (
    <span title={title} className={`chip ${TONE[tone]}`}>
      {icon && <span aria-hidden className="inline-flex">{icon}</span>}
      {children}
    </span>
  );
}

/** Marks anything produced by a learned model that has not been validated yet. */
export function Experimental({ compact = false }: { compact?: boolean }) {
  return (
    <span className="chip border-dashed" title="Learned model, not yet validated against references. Never drives alarms.">
      <FlaskConical aria-hidden />{compact ? 'Exp.' : 'Experimental'}
    </span>
  );
}

/** Signal quality 0–1 as five rising bars + a number. Orange below 40 %. */
export function Quality({ sqi, label = 'Signal', compact = false }: {
  sqi: number | null | undefined; label?: string; compact?: boolean;
}) {
  const q = sqi === null || sqi === undefined || !Number.isFinite(sqi) ? 0 : Math.max(0, Math.min(1, sqi));
  const lit = Math.round(q * 5);
  const weak = q > 0 && q < 0.4;
  return (
    <span className="inline-flex items-center gap-2" role="meter" aria-label={`${label} quality`}
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
  );
}

export function SectionHead({ id, eyebrow, title, children }: {
  id?: string; eyebrow?: string; title: string; children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
      <div className="flex flex-col gap-1 min-w-0">
        {eyebrow && <span className="t-eyebrow text-mute">{eyebrow}</span>}
        <h2 id={id} className="t-head text-ink">{title}</h2>
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
