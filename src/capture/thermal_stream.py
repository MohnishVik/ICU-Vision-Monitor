"""
thermal_stream.py — Waveshare ESP32-S3 thermal camera reader
Binary HTTP polling at http://192.168.4.1/thermal/frame

Hardware confirmed:
  Frame: 62x80 pixels, int16 raw data
  Payload: 9920 bytes = 62 * 80 * 2
  Celsius: raw_int16 / 100.0  (e.g. 3103 -> 31.03 C)
  AP SSID: WSThermal

Endpoint auto-discovery:
  On first start, tries the cached endpoint from camera.yaml.
  If it fails, scans all known endpoints automatically.
  Writes the working endpoint back to camera.yaml (targeted line replace —
  preserves all comments and formatting).
  If nothing works, thermal=None propagates — all consumers handle this.
"""

from __future__ import annotations
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests
import yaml


# All endpoints to try during auto-discovery, in priority order
_SCAN_ENDPOINTS = [
    "/thermal/frame",     # confirmed working — tried first
    "/thermal/raw",
    "/thermal",
    "/thermal/",
    "/raw",
    "/stream",
    "/capture",
    "/",
    "/index.html",
    "/debug",
    "/thermal/debug",
    "/status",
]

_CAMERA_YAML = Path("config/camera.yaml")


class ThermalStream:
    """
    Background-thread HTTP poller for the Waveshare ESP32-S3 thermal camera.

    On first start:
      1. Tries the cached endpoint from camera.yaml.
      2. If that fails, scans all known endpoints automatically.
      3. Writes the working endpoint back to camera.yaml (line-targeted replace).
      4. If nothing works, runs without thermal (thermal=None everywhere).

    The latest decoded frame is available via get_latest_frame().

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

    def __init__(self, cfg: dict, config_path: Path = _CAMERA_YAML):
        """
        Args:
            cfg:         the waveshare_thermal sub-dict from camera.yaml
            config_path: path to camera.yaml for endpoint write-back
        """
        self.cfg         = cfg
        self.config_path = config_path
        self.base_ip     = cfg.get("base_ip", "http://192.168.4.1")
        self.timeout_s   = cfg.get("timeout_s", 2.0)
        self.retry_s     = cfg.get("retry_delay_s", 1.0)

        # Active endpoint — resolved during start()
        self._endpoint: str | None = cfg.get("endpoint", "/thermal/frame")
        self._url: str | None = None

        self._latest_frame: np.ndarray | None = None
        self._latest_ts:    float | None      = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._session: requests.Session | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Resolve the working endpoint, then start the background poll thread.
        Blocks briefly during endpoint resolution (< 5 s in the normal case).
        """
        self._session = requests.Session()

        working_url = self._resolve_endpoint()

        if working_url is None:
            print("[Thermal] WARN: no working endpoint found. "
                  "Thermal stream disabled — continuing without it.")
            self._running = False
            return

        self._url     = working_url
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop,
            name="thermal-poll",
            daemon=True,
        )
        self._thread.start()
        print(f"[Thermal] Polling started → {self._url}")

    def stop(self) -> None:
        """Stop background thread and close HTTP session."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._session:
            self._session.close()
        print("[Thermal] Stream stopped.")

    # ── public API ─────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> tuple[np.ndarray | None, float | None]:
        """
        Returns the most recent thermal frame and its monotonic timestamp.
        Returns (None, None) when ESP32 is offline or stream not started.
        """
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

    def is_online(self) -> bool:
        """True if a valid frame was received within the last 3 seconds."""
        with self._lock:
            if self._latest_ts is None:
                return False
            return (time.monotonic() - self._latest_ts) < 3.0

    # ── endpoint resolution ────────────────────────────────────────────────────

    def _resolve_endpoint(self) -> str | None:
        """
        Try cached endpoint first. If it fails, scan all known endpoints.
        Writes the winner back to camera.yaml (targeted line replace).

        Returns:
            Full working URL string, or None if nothing responds.
        """
        # 1. Try cached endpoint first
        cached = self._endpoint or "/thermal/frame"
        cached_url = f"{self.base_ip}{cached}"

        if self._probe(cached_url):
            print(f"[Thermal] Endpoint OK (cached): {cached_url}")
            return cached_url

        # 2. Cached endpoint failed — auto-scan
        print(f"[Thermal] Cached endpoint {cached_url!r} not responding. "
              "Auto-scanning endpoints...")

        for path in _SCAN_ENDPOINTS:
            if path == cached:
                continue   # already tried
            url = f"{self.base_ip}{path}"
            if self._probe(url):
                print(f"[Thermal] Found working endpoint: {url}")
                self._write_endpoint_to_yaml(path)
                return url

        return None   # nothing worked

    def _probe(self, url: str) -> bool:
        """
        Try one HTTP GET. Returns True only if status=200 AND
        payload is exactly EXPECTED_BYTES (9920).
        Pure byte-count check — no decoding needed here.
        """
        try:
            r = self._session.get(url, timeout=self.timeout_s)
            return r.status_code == 200 and len(r.content) == self.EXPECTED_BYTES
        except requests.exceptions.RequestException:
            return False

    def _write_endpoint_to_yaml(self, new_endpoint: str) -> None:
        """
        Update only the `endpoint:` line in camera.yaml.
        All comments, whitespace, and other values are preserved exactly.

        Strategy: read the file as plain text, replace just the matching line
        with a regex targeted at the endpoint key, write back.
        """
        if not self.config_path.exists():
            print(f"[Thermal] WARN: {self.config_path} not found — "
                  "endpoint not persisted.")
            return

        try:
            text = self.config_path.read_text(encoding="utf-8")

            # Match:  endpoint: "/thermal/frame"   or  endpoint: /thermal/frame
            # Preserves leading whitespace (indentation inside waveshare_thermal block)
            pattern = r'^(\s*endpoint:\s*).*$'
            replacement = rf'\g<1>"{new_endpoint}"'
            new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)

            if n == 0:
                print("[Thermal] WARN: could not find 'endpoint:' key in "
                      f"{self.config_path} — endpoint not persisted.")
                return

            self.config_path.write_text(new_text, encoding="utf-8")
            print(f"[Thermal] camera.yaml updated → endpoint: \"{new_endpoint}\"")

        except OSError as e:
            print(f"[Thermal] WARN: could not write {self.config_path}: {e}")

    # ── poll loop ──────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Runs in background daemon thread.
        Persistent HTTP session prevents ESP32 socket exhaustion.
        On connection loss: retries silently, does NOT clear the last frame
        (consumers check is_online() if they care about freshness).
        On repeated failure: attempts endpoint re-discovery once, then keeps retrying.
        """
        consecutive_failures = 0
        rediscovery_done     = False

        while self._running:
            try:
                response = self._session.get(self._url, timeout=self.timeout_s)

                if response.status_code == 200:
                    frame = self._decode_frame(response.content)
                    if frame is not None:
                        with self._lock:
                            self._latest_frame = frame
                            self._latest_ts    = time.monotonic()
                        consecutive_failures = 0
                    # else: bad payload length — skip, don't count as failure
                else:
                    consecutive_failures += 1

            except requests.exceptions.RequestException:
                consecutive_failures += 1

            # If we've failed 10 times in a row and haven't retried discovery yet
            if consecutive_failures >= 10 and not rediscovery_done:
                print("[Thermal] WARN: 10 consecutive failures. "
                      "Re-running endpoint discovery...")
                new_url = self._resolve_endpoint()
                if new_url and new_url != self._url:
                    self._url = new_url
                    print(f"[Thermal] Switched to new endpoint: {self._url}")
                rediscovery_done     = True
                consecutive_failures = 0

            if consecutive_failures > 0:
                time.sleep(self.retry_s)

    # ── frame decode ───────────────────────────────────────────────────────────

    def _decode_frame(self, raw_bytes: bytes) -> np.ndarray | None:
        """
        Decode raw ESP32 binary payload → float32 Celsius matrix (62, 80).
        Returns None if payload length is wrong.
        """
        if len(raw_bytes) != self.EXPECTED_BYTES:
            return None

        raw  = np.frombuffer(raw_bytes, dtype=np.int16)
        mat  = raw.reshape((self.FRAME_HEIGHT, self.FRAME_WIDTH)).astype(np.float32)
        mat /= self.CELSIUS_SCALE   # e.g. 3103 → 31.03 C
        return mat


# ── display helper (used by diagnose_cameras.py) ──────────────────────────────

def make_display_frame(
    thermal_c:    np.ndarray,
    display_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """
    Convert float32 Celsius frame → colourised BGR display image.
    Uses COLORMAP_INFERNO (matches original test_waveshare_thermal.py).
    """
    import cv2
    t_min, t_max = thermal_c.min(), thermal_c.max()
    if t_max > t_min:
        norm = ((thermal_c - t_min) / (t_max - t_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(thermal_c, dtype=np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return cv2.resize(coloured, display_size, interpolation=cv2.INTER_CUBIC)