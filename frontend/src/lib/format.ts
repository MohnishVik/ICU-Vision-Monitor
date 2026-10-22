// format.ts — wording and number formatting shared across the UI
import type { EngineStatus, PatientState, Vital } from '../types';

export const fmtRate = (v: number | null | undefined): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '--' : Math.round(v).toString();

export const pct = (x: number | null | undefined): string =>
  x === null || x === undefined || !Number.isFinite(x) ? '--' : `${Math.round(x * 100)}%`;

export function ago(ts: number | null | undefined, now = Date.now() / 1000): string {
  if (!ts) return 'never';
  const s = Math.max(0, Math.round(now - ts));
  if (s < 5) return 'just now';
  if (s < 60) return `${s} s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  return `${Math.round(m / 60)} h ago`;
}

export function clock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function duration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return '--';
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export const PATIENT_STATE_TEXT: Record<PatientState, string> = {
  IN_CHAIR: 'In chair',
  SITTING_EDGE: 'On chair edge',
  OUT_OF_CHAIR: 'Out of chair',
  ON_FLOOR: 'On the floor',
  UNKNOWN: 'Not in view',
};

export const STATUS_TEXT: Record<EngineStatus, string> = {
  starting: 'Starting',
  running: 'Monitoring',
  awaiting_calibration: 'Needs chair setup',
  camera_offline: 'Camera offline',
  error: 'Engine error',
};

/** Why a vital has no number, in words a nurse can act on. */
export function vitalUnavailableText(v: Pick<Vital, 'reason' | 'stale' | 'value'> | null | undefined,
                                     apnea = false): string {
  if (apnea) return 'Possible apnoea';
  if (!v) return 'Waiting for data';
  switch (v.reason) {
    case 'camera_offline': return 'Camera offline';
    case 'awaiting_calibration': return 'Chair not set up';
    case 'starting': return 'Starting up';
    case 'warming_up': return 'Collecting signal';
    case 'paths_disagree': return 'Sensors disagree';
    case 'no_valid_input': return 'Signal too weak';
    default: return v.stale ? 'No recent reading' : 'Waiting for data';
  }
}

export const MODULE_TEXT: Record<string, string> = {
  pose: 'Pose tracking',
  rr: 'Respiratory rate',
  hr: 'Heart rate',
  fall: 'Fall detection',
  taa: 'Breathing synchrony',
  pain: 'Pain expression (experimental)',
  rr_learned: 'Respiratory rate, learned (experimental)',
  hr_learned: 'Heart rate, learned (experimental)',
};

export interface SourceInfo { name: string; how: string; learned: boolean; color: string; dash?: string }

/** Every RR / HR source, in display order. Learned paths are dashed in charts. */
export const SOURCES: Record<string, SourceInfo> = {
  'rr.fused':         { name: 'Fused', how: 'Inverse-variance blend of the fused paths', learned: false, color: 'var(--rr-line)' },
  'rr.depth':         { name: 'Chest depth', how: 'D435 depth over the sternum, spectral peak', learned: false, color: 'var(--s1)' },
  'rr.thermal':       { name: 'Nostril thermal', how: 'ESP32 breath temperature, spectral peak', learned: false, color: 'var(--s2)' },
  'rr.cnn_lbrd':      { name: 'CNN · LBRD-IC', how: '1-D CNN on whole-frame green, trained on LBRD-IC', learned: true, color: 'var(--s3)', dash: '6 4' },
  'rr.cnn_d435':      { name: 'CNN · D435 fine-tune', how: '1-D CNN on chest depth, fine-tuned on your D435 clips', learned: true, color: 'var(--s4)', dash: '2 3' },
  'hr.fused':         { name: 'Fused', how: 'Inverse-variance blend of the fused paths', learned: false, color: 'var(--hr-line)' },
  'hr.pos':           { name: 'Skin colour (POS)', how: 'rPPG plane-orthogonal-to-skin, spectral peak', learned: false, color: 'var(--s1)' },
  'hr.efficientphys': { name: 'EfficientPhys', how: 'Temporal-shift CNN, trained on UBFC-rPPG', learned: true, color: 'var(--s3)', dash: '6 4' },
};

export const sourceName = (s: string) => SOURCES[s]?.name ?? s;

export const fmt1 = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(digits);
