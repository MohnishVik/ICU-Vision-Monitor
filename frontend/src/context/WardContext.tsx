// WardContext.tsx — one ward WebSocket shared by the whole app.
// The alarm bar (every page) and the ward grid read the same feed, so the
// browser never opens duplicate connections. Falls back to REST polling
// while the socket is down.
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useWardStream, type LinkState } from '../hooks/useWebSocket';
import { api } from '../lib/api';
import type { PatientSummary } from '../types';

interface WardValue {
  ward: PatientSummary[] | null;
  link: LinkState;
  lastMessageAt: number | null;
  error: string | null;
}

const Ctx = createContext<WardValue>({ ward: null, link: 'connecting', lastMessageAt: null, error: null });

export function WardProvider({ children }: { children: ReactNode }) {
  const { data, link, lastMessageAt } = useWardStream();
  const [polled, setPolled] = useState<PatientSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (link === 'live') return;
    let alive = true;
    const run = () => api.patients()
      .then((p) => { if (alive) { setPolled(p); setError(null); } })
      .catch((e: Error) => alive && setError(e.message));
    void run();
    const h = window.setInterval(run, 3000);
    return () => { alive = false; window.clearInterval(h); };
  }, [link]);

  const ward = link === 'live' ? data ?? polled : polled ?? data;
  return <Ctx.Provider value={{ ward, link, lastMessageAt, error: link === 'live' ? null : error }}>
    {children}
  </Ctx.Provider>;
}

export const useWard = () => useContext(Ctx);
