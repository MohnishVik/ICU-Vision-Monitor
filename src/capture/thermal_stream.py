"""
thermal_stream.py — Waveshare ESP32-S3 thermal camera reader
Binary HTTP polling at http://192.168.4.1/thermal/frame

Hardware confirmed:
  Frame: 62x80 pixels, int16 raw data
  Payload: 9920 bytes = 62 * 80 * 2
  Celsius: raw_int16 / 100.0  (e.g. 3103 -> 31.03 C)
  AP SSID: WSThermal

Why this reader is deliberately gentle
  The ESP32-S3 has little RAM. Earlier versions, whenever one request to the frame
  endpoint failed, fired requests at a dozen other paths back to back over a kept-alive
  connection (and repeated that every 10 s). Each unknown path makes the board look the
  file up and buffer a reply; a burst of them exhausts its RAM and the board powers off,
  which then looks like "thermal camera not connected" for the rest of the session.

  So, like Camera_Configurations/test_waveshare_thermal.py (the reader that works):
    - only ONE endpoint is ever requested: camera.yaml -> waveshare_thermal.endpoint.
      Finding the endpoint is a one-off job for scripts/scan_esp32.py, which closes the
      connection after every request and waits between them.
    - one persistent session while frames flow (no new socket per frame);
    - polls are paced to max_poll_hz (the sensor makes ~8 frames/s; asking faster only
      makes the board send the same frame twice);
    - after a failure the session is closed and rebuilt (a board that rebooted leaves
      the old socket dead) and the wait grows from retry_delay_s to max_backoff_s, so an
      unreachable board is never hammered.

If nothing answers, thermal=None propagates; every consumer handles that.
"""

from __future__ import annotations
import threading
import time

import numpy as np
import requests


class ThermalStream:
    """
    Background-thread HTTP poller for the Waveshare ESP32-S3 thermal camera.

    Usage:
        stream = ThermalStream(cfg)
        stream.start()
        frame, ts = stream.get_latest_frame()  # (62,80) float32 C, or (None, None)
        stream.stop()
    """

    FRAME_WIDTH    = 80
    FRAME_HEIGHT   = 62
    EXPECTED_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 2   # 9920
    CELSIUS_SCALE  = 100.0

    def __init__(self, cfg: dict, session_factory=requests.Session):
        """
        Args:
            cfg: the waveshare_thermal sub-dict from camera.yaml
            session_factory: builds the HTTP session (replaceable in tests)
        """
        self.cfg = cfg
        base = str(cfg.get("base_ip", "http://192.168.4.1")).rstrip("/")
        endpoint = cfg.get("endpoint")
        if endpoint:                       # base_ip + endpoint is what scan_esp32.py tells you to set
            self.url = f"{base}{endpoint if str(endpoint).startswith('/') else '/' + str(endpoint)}"
        else:
            self.url = cfg.get("full_url") or f"{base}/thermal/frame"
        self.timeout_s = float(cfg.get("timeout_s", 2.0))
        self.retry_s = float(cfg.get("retry_delay_s", 1.0))
        self.max_backoff_s = float(cfg.get("max_backoff_s", 5.0))
        self.min_interval_s = 1.0 / max(0.5, float(cfg.get("max_poll_hz", 10.0)))
        self._session_factory = session_factory

        self._latest_frame: np.ndarray | None = None
        self._latest_ts:    float | None      = None
        self._lock    = threading.Lock()
        self._running = False
        self._wake    = threading.Event()        # interrupts waits on stop()
        self._thread: threading.Thread | None = None
        self._session: requests.Session | None = None
        self.frames_ok = 0
        self.failures = 0
        self.last_error: str | None = None       # why the last request failed (shown on the dashboard)
        self.started_at: float | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start polling in a background thread and return at once."""
        self._running = True
        self.started_at = time.monotonic()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, name="thermal-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the HTTP session."""
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=self.timeout_s + 1.0)
        self._close_session()
        print("[Thermal] Stream stopped.")

    # ── public API ─────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> tuple[np.ndarray | None, float | None]:
        """Most recent frame and its monotonic timestamp, or (None, None)."""
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

    def is_online(self) -> bool:
        """True if a valid frame was received within the last 3 seconds."""
        with self._lock:
            return self._latest_ts is not None and (time.monotonic() - self._latest_ts) < 3.0

    def link_detail(self) -> str:
        """One line on the connection state, in plain words, for the dashboard."""
        if self.is_online():
            return f"Receiving frames from {self.url}."
        with self._lock:
            last = self._latest_ts
        since = last if last is not None else self.started_at
        secs = 0 if since is None else int(time.monotonic() - since)
        why = f" ({self.last_error})" if self.last_error else ""
        if last is None and self.failures == 0:
            return f"Connecting to {self.url}..."
        return (f"No answer from {self.url} for {secs} s{why}. Check the ESP32 is powered and this "
                f"computer is on its 'WSThermal' Wi-Fi; retrying every {self.max_backoff_s:.0f} s.")

    # ── poll loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        print(f"[Thermal] Polling {self.url} (max {1.0 / self.min_interval_s:.0f} Hz)")
        backoff = self.retry_s
        was_online = False
        while self._running:
            t0 = time.monotonic()
            ok = self._poll_once()
            if ok:
                if not was_online:
                    print("[Thermal] Frames arriving.")
                was_online, backoff = True, self.retry_s
                wait = self.min_interval_s - (time.monotonic() - t0)
            else:
                if was_online:
                    print(f"[Thermal] WARN: frame request failed; retrying {self.url} with back-off.")
                was_online = False
                self._close_session()                # a rebooted board leaves the socket dead
                wait, backoff = backoff, min(self.max_backoff_s, backoff * 2)
            if wait > 0:
                self._wake.wait(wait)

    def _poll_once(self) -> bool:
        session = self._session                      # local: stop() may clear the attribute
        if session is None:
            session = self._session = self._session_factory()
        try:
            r = session.get(self.url, timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - RequestException, or a session closed by stop()
            self.failures += 1
            self.last_error = ("timed out" if isinstance(exc, requests.exceptions.Timeout)
                               else "no connection" if isinstance(exc, requests.exceptions.ConnectionError)
                               else type(exc).__name__)
            return False
        if r.status_code != 200:
            self.failures += 1
            self.last_error = f"HTTP {r.status_code}"
            return False
        frame = self._decode_frame(r.content)
        if frame is None:
            self.failures += 1                       # wrong size: wrong endpoint or a cut-off reply
            self.last_error = f"{len(r.content)} bytes instead of {self.EXPECTED_BYTES}"
            return False
        self.last_error = None
        with self._lock:
            self._latest_frame, self._latest_ts = frame, time.monotonic()
        self.frames_ok += 1
        return True

    def _close_session(self) -> None:
        s, self._session = self._session, None
        if s is not None:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass

    # ── frame decode ───────────────────────────────────────────────────────────

    def _decode_frame(self, raw_bytes: bytes) -> np.ndarray | None:
        """Raw ESP32 payload -> float32 Celsius (62, 80); None if the length is wrong."""
        if len(raw_bytes) != self.EXPECTED_BYTES:
            return None
        raw = np.frombuffer(raw_bytes, dtype=np.int16)
        mat = raw.reshape((self.FRAME_HEIGHT, self.FRAME_WIDTH)).astype(np.float32)
        mat /= self.CELSIUS_SCALE
        return mat


# ── display helper (used by diagnose_cameras.py) ──────────────────────────────

def make_display_frame(
    thermal_c:    np.ndarray,
    display_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Float32 Celsius frame -> colourised BGR display image (COLORMAP_INFERNO)."""
    import cv2
    t_min, t_max = thermal_c.min(), thermal_c.max()
    if t_max > t_min:
        norm = ((thermal_c - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(thermal_c, dtype=np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.resize(coloured, display_size, interpolation=cv2.INTER_CUBIC)
