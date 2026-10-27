// Tooltip.tsx — how the monitor explains itself.
//
// Any number, badge or control can carry a short, plain-language note. It opens on hover after
// 250 ms, at once on keyboard focus, and on tap for touch; Esc, blur, a tap elsewhere or moving
// away closes it. The note is portalled to <body> with fixed positioning, placed above the
// trigger (or below / to the right) and flipped or slid sideways so it never leaves the viewport.
// aria-describedby always points at the note's text, so a screen reader hears it on focus.
// Wording comes from lib/glossary.ts wherever a definition exists: one meaning, one sentence.
import { Info } from 'lucide-react';
import {
  cloneElement, isValidElement, useCallback, useEffect, useId, useLayoutEffect, useRef, useState,
  type FocusEvent, type KeyboardEvent, type MouseEvent, type PointerEvent, type ReactElement, type ReactNode,
  type Ref,
} from 'react';
import { createPortal } from 'react-dom';
import { GLOSSARY, type GlossKey } from '../lib/glossary';

type Side = 'top' | 'bottom' | 'right';
type Pos = { top: number; left: number; side: Side };

const OPEN_DELAY = 250;
const CLOSE_DELAY = 120;     // long enough to move the pointer onto the note (WCAG 1.4.13 hoverable)
const GAP = 8;
const PAD = 8;
const VOID = new Set(['input', 'select', 'textarea', 'img']);

// Focus opens a note only when it came from the keyboard, so clicking a button or into a
// field never drops a note over the work.
let keyboardModality = false;
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => { if (!e.metaKey && !e.ctrlKey && !e.altKey) keyboardModality = true; }, true);
  window.addEventListener('pointerdown', () => { keyboardModality = false; }, true);
}

// One note at a time: opening one closes whichever was open.
let closeCurrent: (() => void) | null = null;

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(lo, hi));

/** Collision-aware placement: preferred side if it fits, else the opposite, always inside the viewport. */
function place(t: DOMRect, b: DOMRect, side: Side): Pos {
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  if (side === 'right' && t.right + GAP + b.width <= vw - PAD) {
    return { side, left: t.right + GAP, top: clamp(t.top + t.height / 2 - b.height / 2, PAD, vh - PAD - b.height) };
  }
  const above = t.top - GAP - b.height >= PAD;
  const below = t.bottom + GAP + b.height <= vh - PAD;
  const s: Side = side === 'bottom' ? (below || !above ? 'bottom' : 'top') : (above || !below ? 'top' : 'bottom');
  return {
    side: s,
    top: clamp(s === 'top' ? t.top - GAP - b.height : t.bottom + GAP, PAD, vh - PAD - b.height),
    left: clamp(t.left + t.width / 2 - b.width / 2, PAD, vw - PAD - b.width),
  };
}

function setRef<T>(ref: Ref<T> | undefined, value: T) {
  if (typeof ref === 'function') ref(value);
  else if (ref && typeof ref === 'object') (ref as { current: T }).current = value;
}

function Note({ term, title, content }: { term?: GlossKey; title?: string; content?: ReactNode }) {
  const g = term ? GLOSSARY[term] : null;
  const head = title ?? g?.title;
  return (
    <>
      {head && <span className="block t-strong text-ink">{head}</span>}
      {g && <span className="block">{g.body}</span>}
      {content != null && content !== false && <span className="block">{content}</span>}
    </>
  );
}

export interface TipProps {
  /** Glossary entry: its title and definition head the note. */
  term?: GlossKey;
  /** Heading override (defaults to the glossary title). */
  title?: string;
  /** Extra or one-off text, shown after the glossary definition. */
  content?: ReactNode;
  children: ReactNode;
  /** Attach to the single child element (a button, link, meter...) instead of wrapping it in a focusable span. */
  asChild?: boolean;
  /** Tap and keyboard activation toggle the note (InfoTip, and every wrapped value). */
  toggle?: boolean;
  side?: Side;
  /** Classes for the wrapping span (wrap mode only). */
  className?: string;
  /** Fixed id, for when another element (an input) must also point at the note. */
  id?: string;
}

export function Tip({ term, title, content, children, asChild = false, toggle, side = 'top', className = '', id: fixedId }: TipProps) {
  const auto = useId();
  const id = fixedId ?? `tip-${auto.replace(/:/g, '')}`;
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<Pos | null>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const bubble = useRef<HTMLDivElement | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const pointer = useRef('mouse');
  const tapToggles = toggle ?? !asChild;

  const cancel = () => window.clearTimeout(timer.current);
  const hide = useCallback(() => {
    window.clearTimeout(timer.current);
    setOpen(false);
    setPos(null);
    if (closeCurrent === hide) closeCurrent = null;
  }, []);
  const show = useCallback(() => {
    window.clearTimeout(timer.current);
    if (closeCurrent && closeCurrent !== hide) closeCurrent();
    closeCurrent = hide;
    setOpen(true);
  }, [hide]);
  const later = (fn: () => void, ms: number) => { cancel(); timer.current = window.setTimeout(fn, ms); };

  const measure = useCallback(() => {
    const t = trigger.current;
    const b = bubble.current;
    if (!t || !b) return;
    const next = place(t.getBoundingClientRect(), b.getBoundingClientRect(), side);
    setPos((p) => (p && p.side === next.side && Math.abs(p.top - next.top) < 0.5 && Math.abs(p.left - next.left) < 0.5
      ? p : next));
  }, [side]);

  // place before paint, and again whenever the note's content changes size
  useLayoutEffect(() => { if (open) measure(); });

  useEffect(() => {
    if (!open) return;
    let raf = 0;
    const follow = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(measure); };
    const onKey = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') hide(); };
    const onDown = (e: globalThis.PointerEvent) => {
      const n = e.target as Node;
      if (!trigger.current?.contains(n) && !bubble.current?.contains(n)) hide();
    };
    window.addEventListener('scroll', follow, true);
    window.addEventListener('resize', follow);
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', follow, true);
      window.removeEventListener('resize', follow);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open, measure, hide]);

  useEffect(() => () => { window.clearTimeout(timer.current); if (closeCurrent === hide) closeCurrent = null; }, [hide]);

  const handlers = {
    onPointerEnter: (e: PointerEvent<HTMLElement>) => {
      pointer.current = e.pointerType;
      if (e.pointerType !== 'touch') later(show, OPEN_DELAY);
    },
    onPointerLeave: (e: PointerEvent<HTMLElement>) => { if (e.pointerType !== 'touch') later(hide, CLOSE_DELAY); },
    onPointerDown: (e: PointerEvent<HTMLElement>) => { pointer.current = e.pointerType; },
    onClick: (e: MouseEvent<HTMLElement>) => {
      const fromKeyboard = e.detail === 0;
      if (tapToggles && (fromKeyboard || pointer.current === 'touch')) (open ? hide() : show());
      else if (tapToggles) show();
    },
    onFocus: () => { if (keyboardModality) show(); },
    onBlur: (e: FocusEvent<HTMLElement>) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) hide();
    },
    onKeyDown: (e: KeyboardEvent<HTMLElement>) => {
      if (e.key === 'Escape' && open) { e.stopPropagation(); hide(); }
      else if (tapToggles && !asChild && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); (open ? hide() : show()); }
    },
  };

  const note = <Note term={term} title={title} content={content} />;
  const quiet = !open && <span id={id} hidden>{note}</span>;
  const floating = open && typeof document !== 'undefined' && createPortal(
    <div ref={bubble} id={id} role="tooltip" className="tip" data-side={pos?.side ?? side}
         style={{ top: pos?.top ?? 0, left: pos?.left ?? 0, visibility: pos ? 'visible' : 'hidden' }}
         onPointerEnter={cancel}
         onPointerLeave={(e) => { if (e.pointerType !== 'touch') later(hide, CLOSE_DELAY); }}>
      {note}
    </div>,
    document.body,
  );

  if (asChild && isValidElement(children)) {
    const child = children as ReactElement<Record<string, unknown>> & { ref?: Ref<HTMLElement> };
    const p = child.props;
    const merged: Record<string, unknown> = {
      ref: (el: HTMLElement | null) => { trigger.current = el; setRef(child.ref, el); },
      'aria-describedby': [p['aria-describedby'], id].filter(Boolean).join(' '),
      'data-tip-open': open || undefined,
    };
    for (const [k, fn] of Object.entries(handlers)) {
      const own = p[k] as ((e: unknown) => void) | undefined;
      merged[k] = (e: never) => { own?.(e); fn(e); };
    }
    const isVoid = typeof child.type === 'string' && VOID.has(child.type);
    return isVoid
      ? <>{cloneElement(child, merged)}{quiet}{floating}</>
      : <>{cloneElement(child, merged, <>{p.children as ReactNode}{quiet}</>)}{floating}</>;
  }

  return (
    <>
      <span ref={trigger} tabIndex={0} aria-describedby={id} data-tip-open={open || undefined}
            className={`tip-target ${className}`} {...handlers}>
        {children}{quiet}
      </span>
      {floating}
    </>
  );
}

/** The small (i) beside a section heading: 2–4 sentences on what the section shows and how to read it. */
export function InfoTip({ term, label, content, title, side = 'bottom' }: {
  term?: GlossKey; label?: string; content?: ReactNode; title?: string; side?: Side;
}) {
  const name = title ?? (term ? GLOSSARY[term].title : 'this section');
  return (
    <Tip term={term} title={title} content={content} side={side} asChild toggle>
      <button type="button" className="infotip" aria-label={label ?? `About ${name.charAt(0).toLowerCase()}${name.slice(1)}`}>
        <Info aria-hidden />
      </button>
    </Tip>
  );
}
