// Logo.tsx — the ICU Vision Monitor emblem (winged caduceus + camera aperture), drawn
// in one colour. The artwork is an alpha mask, so it takes --brand and follows the theme.
import mark from '../assets/logo-mark.png';

const RATIO = 414 / 320;   // width / height of the mask asset

export default function Logo({ height = 32, className = '', label = 'ICU Vision Monitor' }: {
  height?: number; className?: string; label?: string | null;
}) {
  const mask = `url(${mark}) center / contain no-repeat`;
  return (
    <span
      role={label ? 'img' : undefined}
      aria-label={label ?? undefined}
      aria-hidden={label ? undefined : true}
      className={`inline-block shrink-0 bg-brand ${className}`}
      style={{ height, width: Math.round(height * RATIO), mask, WebkitMask: mask }}
    />
  );
}
