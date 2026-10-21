// usePatientData.ts — REST polling for /history, /health and /calibration
import { useCallback, useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { CalibrationStatus, Health, History, ThermalPayload } from '../types';

function usePoll<T>(fetcher: (() => Promise<T>) | null, everyMs: number, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!fetcher) return;
    let alive = true;
    let busy = false;                 // never stack requests on a slow network
    const run = () => {
      if (busy) return;
      busy = true;
      fetcher()
        .then((d) => {
          if (!alive) return;
          setData(d);
          setError(null);
        })
        .catch((e: Error) => alive && setError(e.message))
        .finally(() => { busy = false; });
    };
    void run();
    const h = window.setInterval(run, everyMs);
    return () => {
      alive = false;
      window.clearInterval(h);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick, everyMs]);

  return { data, error, refresh };
}

export function usePatientHistory(id: string | undefined, minutes: number) {
  return usePoll<History>(id ? () => api.history(id, minutes) : null, minutes <= 15 ? 2000 : 5000,
                          [id, minutes]);
}

export function useHealth(everyMs = 5000) {
  return usePoll<Health>(() => api.health(), everyMs, []);
}

/** Thermal grid + breath trace at ~4 Hz (the ESP32 delivers ~8 frames/s). */
export function useThermal(id: string | undefined, everyMs = 250) {
  return usePoll<ThermalPayload>(id ? () => api.thermal(id) : null, everyMs, [id]);
}

export function useCalibration(id: string | undefined) {
  return usePoll<CalibrationStatus>(id ? () => api.calibration(id) : null, 4000, [id]);
}
