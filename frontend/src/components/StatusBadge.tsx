// StatusBadge.tsx — icon + text pill. Colour never carries meaning alone.
import type { ReactNode } from 'react';

export type BadgeTone = 'neutral' | 'ok' | 'info' | 'amber' | 'red' | 'demo';

const TONE: Record<BadgeTone, string> = {
  neutral: 'border-hairline text-ink2 bg-transparent',
  ok: 'border-hairline text-ink bg-transparent',
  info: 'border-[#2c5a60] text-[#9fe0e6] bg-[#12343a]',
  amber: 'border-[#6b4419] text-[#ffc488] bg-[#3a2611]',
  red: 'border-[#7a2630] text-[#ffb3ba] bg-[#3d141a]',
  demo: 'border-[#5a4a8a] text-[#d8ccff] bg-[#261f3f]',
};

export default function StatusBadge({ tone = 'neutral', icon, children, title }: {
  tone?: BadgeTone;
  icon?: ReactNode;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-full border t-micro whitespace-nowrap ${TONE[tone]}`}
    >
      {icon && <span aria-hidden className="inline-flex [&>svg]:h-3.5 [&>svg]:w-3.5">{icon}</span>}
      {children}
    </span>
  );
}
