// SignalQualityBar.tsx — Signal Quality Index (0–1) as five steps + a number.
// Neutral when the signal is usable; amber below 0.4 so weak data is noticed.

export default function SignalQualityBar({ sqi, label = 'Signal', compact = false }: {
  sqi: number | null | undefined;
  label?: string;
  compact?: boolean;
}) {
  const q = sqi === null || sqi === undefined || !Number.isFinite(sqi) ? 0 : Math.max(0, Math.min(1, sqi));
  const lit = Math.round(q * 5);
  const weak = q > 0 && q < 0.4;
  const on = weak ? 'bg-amber' : 'bg-ink2';
  return (
    <div
      className="inline-flex items-center gap-2"
      role="meter"
      aria-label={`${label} quality`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(q * 100)}
    >
      <span className="flex items-end gap-[2px] h-3.5" aria-hidden>
        {[0, 1, 2, 3, 4].map((i) => (
          <span
            key={i}
            className={`w-[4px] rounded-[1px] ${i < lit ? on : 'bg-hairline'}`}
            style={{ height: `${40 + i * 15}%` }}
          />
        ))}
      </span>
      {!compact && (
        <span className={`t-micro tnum ${weak ? 'text-amber' : 'text-mute'}`}>
          {label} {q > 0 ? `${Math.round(q * 100)}%` : 'none'}
        </span>
      )}
    </div>
  );
}
