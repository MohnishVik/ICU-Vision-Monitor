// types.ts — TypeScript mirror of the server contract (src/types.py + src/serving/state.py)
//
// The one rule the UI must keep: `value: null` means "cannot measure right now",
// and `stale: true` means "no fresh value for > 5 s". Neither is ever shown as
// a live normal reading.

export interface Measurement {
  value: number | null;    // null = "cannot measure right now"
  sqi: number;             // Signal Quality Index 0.0–1.0
  ts: number;              // Unix timestamp
  source: string;          // e.g. "rr.depth", "hr.pos"
  latency_ms: number;
  meta: Record<string, unknown>;
}

export type PatientState = 'IN_CHAIR' | 'SITTING_EDGE' | 'OUT_OF_CHAIR' | 'ON_FLOOR' | 'UNKNOWN';
export type LightingState = 'bright' | 'dim' | 'dark' | 'unknown';
export type EWSLevel = 'green' | 'amber' | 'red' | 'unknown';
export type EngineStatus =
  | 'starting' | 'running' | 'awaiting_calibration' | 'camera_offline' | 'error';
export type ModuleState = 'ok' | 'degraded' | 'unavailable' | 'error' | 'pending';
export type VitalKey = 'hr' | 'rr';

export interface Vital {
  value: number | null;       // Kalman-smoothed, null when stale or unmeasurable
  raw: number | null;         // fused, pre-smoothing
  sqi: number;
  stale: boolean;
  age_s: number | null;
  unit: string;
  reason: string | null;      // why value is null: no_valid_input | paths_disagree | warming_up
  subscore: number | null;    // NEWS2 sub-score
  sources: Record<string, SourceReading>;
  learned_fused: boolean;
  apnea?: boolean;
}

/** A source's reading plus whether it feeds the fused value or is shown alongside it. */
export interface SourceReading extends Measurement {
  role: 'fused' | 'display';
}

export interface PainInfo {
  available: boolean;
  experimental: boolean;
  state?: ModuleState;
  reason?: string | null;
  probability?: number | null;     // mean P(pain expression) over the last few seconds
  probability_raw?: number | null;
  threshold?: number | null;
  pain_detected?: boolean | null;
  sqi?: number;
  stale?: boolean;
  model?: string;
  trained_on?: string;
}

export interface SessionStatus {
  active: boolean;
  id?: string;
  label?: string;
  condition?: string;
  started?: number;
  rows?: number;
  elapsed_s?: number;
  reference?: { rr: number | null; hr: number | null };
  error?: string | null;
}

export interface EWS {
  total: number | null;
  level: EWSLevel;
  subscores: Record<string, number | null>;
  missing: string[];
  action: string;
  action_text: string;
  partial: boolean;
  explanation: string;
}

export interface FallInfo {
  state: number | null;
  state_name: 'NORMAL' | 'FALLING' | 'LYING_DOWN' | 'UNKNOWN';
  patient_state: PatientState;
  sqi: number;
  stale: boolean;
  ml_probability: number | null;
  provisional: boolean;
  last_confirmed: number | null;
}

export interface TAAInfo {
  flag: boolean;
  phase_rad: number | null;
  sustain_s: number | null;
  sqi: number;
  stale: boolean;
  reason: string | null;
}

export interface Scene {
  has_patient: boolean;
  in_chair_frac: number;
  staff_count: number;
  lighting: LightingState;
  h_centroid: number;
  thermal_online: boolean;
  calibrated: boolean;
  image_size: [number, number] | null;      // [h, w]
  keypoints: [number, number][] | null;
  keypoints_conf: number[] | null;
  person_bbox: [number, number, number, number] | null;
  others: [number, number, number, number][];
  chair: [number, number][] | null;
  redaction: 'faces' | 'unavailable' | 'disabled';
  nostril_box: [number, number, number, number] | null;   // RGB px, breathing region
  thermal_roi: [number, number, number, number] | null;   // thermal px
  thermal_registered: boolean;
  thermal_quality: AlignmentQuality | null;
}

// ------------------------------------------------------------------ thermal (ESP32)
export type AlignmentQuality = 'good' | 'fair' | 'poor' | 'unchecked';

export interface ThermalGrid {
  w: number;
  h: number;
  min: number;
  max: number;
  data: number[];            // centi-degC, row-major, length w*h
}

export interface PointPair { rgb: [number, number]; thermal: [number, number] }

export interface ThermalRegistrationStatus {
  registered: boolean;
  source: 'session' | 'camera.yaml' | 'demo' | null;
  error_px: number | null;
  n_pairs: number;
  quality: AlignmentQuality | null;
  ts: number | null;
  pairs: PointPair[];
}

export interface ThermalPayload extends Partial<ThermalGrid> {
  online: boolean;
  ts: number | null;
  roi: [number, number, number, number] | null;
  roi_source: 'nostril' | 'warmest_patch';
  registration: ThermalRegistrationStatus;
  trace: { t: number[]; c: number[] };
}

export interface ThermalSnapshot {
  snapshot_id: string;
  rgb_size: [number, number];
  thermal: ThermalGrid;
}

export type AlarmKind = 'ews' | 'fall' | 'apnea' | 'taa';
export interface Alarm {
  id: number;
  kind: AlarmKind;
  severity: 'amber' | 'red';
  title: string;
  message: string;
  state: 'pending' | 'active' | 'acknowledged' | 'cleared';
  first_seen: number;
  active_since: number | null;
  last_seen: number;
  stage: number;
  action: 'log_only' | 'dashboard_highlight' | 'chime' | 'phone_push' | string;
  suppressed: boolean;
  acked_at: number | null;
  cleared_at: number | null;
}

export interface TrendStats {
  window_s: number;
  n: number;
  mean: number | null;
  std: number | null;
  slope_per_min: number | null;
  first: number | null;
  last: number | null;
  span_s: number;
}

export interface ModuleStatus {
  ok: boolean;
  state: ModuleState;
  detail: string;
  errors: number;
}

export interface PatientSnapshot {
  patient_id: string;
  label: string;
  mode: 'camera' | 'demo';
  simulated: boolean;
  ts: number;
  status: EngineStatus;
  status_detail: string;
  vitals: Record<VitalKey, Vital> | null;
  ews: EWS | null;
  fall: FallInfo | null;
  taa: TAAInfo | null;
  pain: PainInfo;
  session: SessionStatus | null;
  scene: Scene;
  alarms: Alarm[];
  alarm_history: Alarm[];
  trends: Record<VitalKey, Record<string, TrendStats>> | null;
  modules: Record<string, ModuleStatus>;
  metrics: { fps?: number; latency_ms?: number; uptime_s?: number; actionable_alerts_shift?: number };
  version: number;
}

export interface VitalSummary {
  value: number | null;
  stale: boolean;
  sqi: number;
  subscore: number | null;
  reason: string | null;
}

export interface PatientSummary {
  patient_id: string;
  label: string;
  mode: 'camera' | 'demo';
  simulated: boolean;
  status: EngineStatus;
  status_detail: string;
  ts: number;
  version: number;
  rr: VitalSummary;
  hr: VitalSummary;
  apnea: boolean;
  ews: { total: number | null; level: EWSLevel; explanation: string; partial: boolean };
  fall: { state_name: string | null; patient_state: PatientState | null };
  taa_flag: boolean;
  scene: {
    has_patient: boolean | null; staff_count: number | null; thermal_online: boolean | null;
    calibrated: boolean | null; lighting: LightingState | null;
  };
  alarm_count: number;
  top_alarm: Alarm | null;
  pain: { available: boolean | null; probability: number | null; pain_detected: boolean | null };
  recording: boolean;
}

export interface History {
  patient_id: string;
  minutes: number;
  t: number[];
  rr: (number | null)[];
  hr: (number | null)[];
  rr_sqi: (number | null)[];
  hr_sqi: (number | null)[];
  ews: (number | null)[];
  fall: (number | null)[];
  taa: (boolean | null)[];
  pain: (number | null)[];
  sources: Record<string, (number | null)[]>;
}

export interface ModelFile {
  file: string;
  used_by: string;
  required: boolean;
  present: boolean;
  size_bytes: number | null;
}

export interface ModelsReport { mode: string; files: ModelFile[]; missing_required: string[] }

export interface SessionMeta {
  id: string;
  patient_id: string;
  label: string;
  condition: string;
  notes: string;
  started: number;
  ended: number | null;
  rows: number;
  simulated: boolean;
  mode: string;
  fuse_learned: boolean;
  sources: string[];
}

export interface SourceScore {
  n_scored: number;
  coverage: number | null;
  mae: number | null;
  rmse: number | null;
  bias: number | null;
  within_tol: number | null;
  tolerance: number;
  pearson_r: number | null;
  mae_unfiltered: number | null;
  coverage_unfiltered: number | null;
}

export interface SessionReport {
  session: Partial<SessionMeta>;
  settings: { warmup_s: number; settle_s: number; tolerance: Record<VitalKey, number> };
  rr: Record<string, SourceScore>;
  hr: Record<string, SourceScore>;
}

export interface Health {
  status: 'ok' | 'degraded';
  version: string;
  mode: 'camera' | 'demo';
  uptime_s: number;
  server_time: number;
  patients: {
    patient_id: string;
    status: EngineStatus;
    status_detail: string;
    modules: Record<string, ModuleStatus>;
    stale: Record<string, boolean>;
    metrics: PatientSnapshot['metrics'];
  }[];
}

export interface CalibrationStatus {
  calibrated: boolean;
  corners: [number, number][] | null;
  seat_depth_m: number | null;
  label: 'chair' | 'bed' | null;
  image_size: [number, number] | null;
  has_saved: boolean;
}

// Backward-compatible alias for the original scaffold name.
export type PatientVitals = PatientSnapshot;
