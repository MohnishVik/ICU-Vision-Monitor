// types.ts — TypeScript mirror of src/types.py Measurement contract

export interface Measurement {
  value: number | null;    // null = "cannot measure right now"
  sqi: number;             // Signal Quality Index 0.0–1.0
  ts: number;              // Unix timestamp
  source: string;          // e.g. "rr.fused", "hr.efficientphys"
  latency_ms: number;
  meta: Record<string, unknown>;
}

export type PatientState = "IN_CHAIR" | "SITTING_EDGE" | "OUT_OF_CHAIR" | "ON_FLOOR";
export type LightingState = "bright" | "dim" | "dark";
export type EWSLevel = "green" | "amber" | "red";

export interface PatientVitals {
  patient_id: string;
  hr: Measurement;
  rr: Measurement;
  fall_state: PatientState;
  pain: Measurement;
  ews_score: number;
  ews_level: EWSLevel;
  taa_flag: boolean;
  last_updated: number;
}
