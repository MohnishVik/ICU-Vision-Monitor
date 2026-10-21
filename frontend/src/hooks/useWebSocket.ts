// useWebSocket.ts — WS /ws/stream/{id} and /ws/ward with automatic reconnect.
//
// Connection state is surfaced to the UI: when the socket is down, the last
// snapshot is kept on screen but marked as not live, never silently presented
// as current.
import { useEffect, useRef, useState } from 'react';
import { urls } from '../lib/api';
import type { PatientSnapshot, PatientSummary } from '../types';

export type LinkState = 'connecting' | 'live' | 'reconnecting';

export function useWebSocket<T>(path: string | null, expectType: string) {
  const [data, setData] = useState<T | null>(null);
  const [link, setLink] = useState<LinkState>('connecting');
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const retry = useRef(0);

  useEffect(() => {
    if (!path) return;
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      setLink(retry.current === 0 ? 'connecting' : 'reconnecting');
      ws = new WebSocket(urls.ws(path));
      ws.onopen = () => {
        retry.current = 0;
        setLink('live');
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as { type: string; data: T };
          if (msg.type === expectType) {
            setData(msg.data);
            setLastMessageAt(Date.now());
          }
        } catch {
          /* ignore malformed frame */
        }
      };
      ws.onclose = () => {
        if (closed) return;
        setLink('reconnecting');
        const delay = Math.min(10000, 500 * 2 ** retry.current);
        retry.current += 1;
        timer = window.setTimeout(connect, delay);
      };
      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      closed = true;
      if (timer) window.clearTimeout(timer);
      ws?.close();
    };
  }, [path, expectType]);

  return { data, link, lastMessageAt };
}

export const usePatientStream = (id: string | undefined) =>
  useWebSocket<PatientSnapshot>(id ? `/ws/stream/${encodeURIComponent(id)}` : null, 'tick');

export const useWardStream = () => useWebSocket<PatientSummary[]>('/ws/ward', 'ward');
