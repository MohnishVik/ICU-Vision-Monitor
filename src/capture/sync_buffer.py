"""
sync_buffer.py — joins the D435 and the Waveshare thermal camera into one FrameBundle stream

D435 produces a bundle every ~33 ms (30 fps).
The thermal camera produces a frame every ~125 ms (about 8 fps), polled in a background thread.

Rule: every D435 bundle gets the NEWEST thermal frame, provided that frame is at most
camera.yaml -> sync.thermal_max_age_ms old (250 ms = two thermal frames). Otherwise thermal=None.
bundle.thermal_ts says when that thermal frame arrived, so the same frame showing up in
several bundles can be recognised and skipped by the thermal RR path.

This is the only place camera data is combined. Nothing else reads the cameras directly.
"""

from __future__ import annotations
import time
from typing import Iterator

from src.types import FrameBundle
from src.capture.d435_stream import D435Stream
from src.capture.thermal_stream import ThermalStream


class SyncBuffer:
    """
    Wraps D435Stream and ThermalStream and emits aligned FrameBundle objects
    at the D435 frame rate.

    Usage:
        cfg = load_camera_config()
        with SyncBuffer(cfg) as buf:
            for bundle in buf.frames():
                process(bundle)
    """

    def __init__(self, camera_cfg: dict):
        """
        Args:
            camera_cfg: the full parsed camera.yaml dict
        """
        self.d435_cfg    = camera_cfg["d435"]
        self.thermal_cfg = camera_cfg["waveshare_thermal"]
        self.max_thermal_age_s = camera_cfg["sync"]["thermal_max_age_ms"] / 1000.0

        self._d435    = D435Stream(self.d435_cfg)
        self._thermal = ThermalStream(self.thermal_cfg) \
            if self.thermal_cfg.get("enabled", False) else None

        self._thermal_online_logged = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start both hardware streams."""
        # Thermal first: it runs in a background thread and does not block
        if self._thermal:
            self._thermal.start()
            time.sleep(0.5)   # give the ESP32 a moment to answer

        # D435 blocks until the hardware is ready (+1 s settling)
        self._d435.start()

    def stop(self) -> None:
        """Stop both streams cleanly."""
        self._d435.stop()
        if self._thermal:
            self._thermal.stop()

    def __enter__(self) -> "SyncBuffer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── frame generator ───────────────────────────────────────────────────────

    def frames(self) -> Iterator[FrameBundle]:
        """
        Yields one complete FrameBundle per D435 tick.

        thermal is the newest thermal frame if it is at most thermal_max_age_ms old,
        otherwise None (ESP32 offline, disabled, or lagging).
        """
        for d435_bundle in self._d435.frames():
            thermal_array  = None
            thermal_ts_out = None

            if self._thermal is not None:
                thermal_frame, thermal_ts = self._thermal.get_latest_frame()

                if thermal_frame is not None and thermal_ts is not None:
                    age = abs(d435_bundle.ts_mono - thermal_ts)
                    if age <= self.max_thermal_age_s:
                        thermal_array  = thermal_frame
                        thermal_ts_out = thermal_ts

                        # Log once when thermal comes online
                        if not self._thermal_online_logged:
                            print("[SyncBuffer] Thermal stream synchronised")
                            self._thermal_online_logged = True

            yield FrameBundle(
                rgb=d435_bundle.rgb,
                depth=d435_bundle.depth,
                ir=d435_bundle.ir,
                thermal=thermal_array,
                ts_mono=d435_bundle.ts_mono,
                thermal_ts=thermal_ts_out,
            )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def thermal_online(self) -> bool:
        """True if the thermal stream received a valid frame in the last 3 seconds."""
        if self._thermal is None:
            return False
        return self._thermal.is_online()

    def mount_distance_m(self) -> float | None:
        """Measured centre-pixel depth in metres (startup sanity check)."""
        return self._d435.get_center_depth_m()


# ── convenience loader ────────────────────────────────────────────────────────

def load_camera_config(path: str = "config/camera.yaml") -> dict:
    """Load and return the full camera.yaml as a dict."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
