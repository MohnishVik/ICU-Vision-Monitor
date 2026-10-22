// api.ts — typed client for the FastAPI server (src/serving/api.py)
import type {
  CalibrationStatus, Health, History, ModelsReport, PatientSnapshot, PatientSummary, PointPair,
  SessionMeta, SessionReport, SessionStatus, ThermalPayload, ThermalRegistrationStatus, ThermalSnapshot,
} from '../types';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, 'Cannot reach the VisionICU server. Check that python main.py is running.');
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  health: () => request<Health>('/api/health'),
  patients: () => request<PatientSummary[]>('/api/patients'),
  vitals: (id: string) => request<PatientSnapshot>(`/api/patients/${enc(id)}/vitals`),
  history: (id: string, minutes: number, maxPoints = 720) =>
    request<History>(`/api/patients/${enc(id)}/history?minutes=${minutes}&max_points=${maxPoints}`),
  ack: (id: string, alarmId: number | null = null) =>
    request<{ acknowledged: number }>(`/api/patients/${enc(id)}/ack`, {
      method: 'POST',
      body: JSON.stringify({ alarm_id: alarmId }),
    }),
  calibration: (id: string) => request<CalibrationStatus>(`/api/patients/${enc(id)}/calibration`),
  setCalibration: (id: string, corners: [number, number][], label: 'chair' | 'bed') =>
    request<CalibrationStatus>(`/api/patients/${enc(id)}/calibration`, {
      method: 'POST',
      body: JSON.stringify({ corners, label }),
    }),
  reuseCalibration: (id: string) =>
    request<CalibrationStatus>(`/api/patients/${enc(id)}/calibration/reuse`, { method: 'POST' }),
  clearCalibration: (id: string) =>
    request<CalibrationStatus>(`/api/patients/${enc(id)}/calibration`, { method: 'DELETE' }),

  thermal: (id: string) => request<ThermalPayload>(`/api/patients/${enc(id)}/thermal`),
  thermalRegistration: (id: string) =>
    request<ThermalRegistrationStatus>(`/api/patients/${enc(id)}/thermal/registration`),
  setThermalRegistration: (id: string, pairs: PointPair[]) =>
    request<ThermalRegistrationStatus>(`/api/patients/${enc(id)}/thermal/registration`, {
      method: 'POST',
      body: JSON.stringify({ pairs }),
    }),
  clearThermalRegistration: (id: string) =>
    request<ThermalRegistrationStatus>(`/api/patients/${enc(id)}/thermal/registration`, { method: 'DELETE' }),
  models: () => request<ModelsReport>('/api/models'),
  session: (id: string) => request<SessionStatus>(`/api/patients/${enc(id)}/session`),
  startSession: (id: string, body: { label: string; condition: string; notes: string;
                                     reference: { rr: number | null; hr: number | null } }) =>
    request<SessionStatus>(`/api/patients/${enc(id)}/session`, { method: 'POST', body: JSON.stringify(body) }),
  addReference: (id: string, body: { rr: number | null; hr: number | null; note: string }) =>
    request<SessionStatus>(`/api/patients/${enc(id)}/session/reference`,
                           { method: 'POST', body: JSON.stringify(body) }),
  stopSession: (id: string) => request<SessionStatus>(`/api/patients/${enc(id)}/session`, { method: 'DELETE' }),
  sessions: () => request<SessionMeta[]>('/api/sessions'),
  report: (sid: string, warmup: number, settle: number) =>
    request<SessionReport>(`/api/sessions/${enc(sid)}/report?warmup_s=${warmup}&settle_s=${settle}`),
  thermalSnapshot: (id: string) =>
    request<ThermalSnapshot>(`/api/patients/${enc(id)}/thermal/snapshot`, { method: 'POST' }),
};

export const urls = {
  stream: (id: string) => `/api/patients/${enc(id)}/stream.mjpg`,
  frame: (id: string) => `/api/patients/${enc(id)}/frame.jpg`,
  calibrationFrame: (id: string) => `/api/patients/${enc(id)}/calibration/frame`,
  snapshotRgb: (id: string, sid: string) => `/api/patients/${enc(id)}/thermal/snapshot/${enc(sid)}/rgb.jpg`,
  thermalOverlay: (id: string) => `/api/patients/${enc(id)}/thermal/overlay.jpg`,
  ws: (path: string) => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}${path}`;
  },
};
