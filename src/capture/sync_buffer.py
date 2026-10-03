"""
sync_buffer.py — cross-sensor temporal alignment for D435 + Waveshare thermal

D435 pushes FrameBundle objects at 30fps via its frame generator.
ThermalStream updates its latest frame at ~8fps in a background thread.

Strategy:
  For each D435 FrameBundle, attach the most recent thermal frame if:
    |ts_d435 - ts_thermal| < tolerance_ms
  Otherwise attach thermal=None (stream late, offline, or lagging).

The sync_buffer is the single place that creates the complete FrameBundle
consumed by all downstream layers. Nothing else reads from cameras directly.
"""

from __future__ import annotations
import time
from typing import Iterator

from src.types import FrameBundle
from src.capture.d435_stream import D435Stream
from src.capture.thermal_stream import ThermalStream


class SyncBuffer:
    """
    Wraps D435Stream and ThermalStream.
    Emits temporally-aligned FrameBundle objects at the D435 frame rate.

    Usage:
        buf = SyncBuffer(d435_cfg, thermal_cfg)
        with buf:
            for bundle in buf.frames():
                process(bundle)
    """

    def __init__(self, camera_cfg: dict):
        """
        Args:
            camera_cfg: full parsed camera.yaml dict
        """
        self.d435_cfg    = camera_cfg["d435"]
        self.thermal_cfg = camera_cfg["waveshare_thermal"]
        self.tolerance_s = camera_cfg["sync"]["tolerance_ms"] / 1000.0

        self._d435    = D435Stream(self.d435_cfg)
        self._thermal = ThermalStream(self.thermal_cfg) \
            if self.thermal_cfg.get("enabled", False) else None

        self._thermal_online_logged = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start both hardware streams."""
        # Start thermal first (background thread, non-blocking)
        if self._thermal:
            self._thermal.start()
            # Give ESP32 a moment to respond before D435 starts dominating the loop
            time.sleep(0.5)

        # Start D435 (blocks until hardware is ready + 1s stabilisation)
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
        Yields one complete FrameBundle per D435 frame tick.
        Attaches the latest thermal frame when it is within tolerance_ms.
        Sets thermal=None when ESP32 is offline, lagging, or disabled.

        This is the only generator consumers should use.
        """
        for d435_bundle in self._d435.frames():
            thermal_array = None

            if self._thermal is not None:
                thermal_frame, thermal_ts = self._thermal.get_latest_frame()

                if thermal_frame is not None and thermal_ts is not None:
                    dt = abs(d435_bundle.ts_mono - thermal_ts)
                    if dt <= self.tolerance_s:
                        thermal_array = thermal_frame

                        # Log once when thermal comes online
                        if not self._thermal_online_logged:
                            print("[SyncBuffer] Thermal stream synchronised ✓")
                            self._thermal_online_logged = True
                    # else: thermal frame too old — attach None, no warning
                    # (thermal is slow at ~8fps; gap is expected between D435 ticks)

            # Yield a new FrameBundle with the thermal field filled in
            yield FrameBundle(
                rgb=d435_bundle.rgb,
                depth=d435_bundle.depth,
                ir=d435_bundle.ir,
                thermal=thermal_array,
                ts_mono=d435_bundle.ts_mono,
            )

    # ── diagnostics ───────────────────────────────────────────────────────────

    def thermal_online(self) -> bool:
        """True if thermal stream received a valid frame within last 3 seconds."""
        if self._thermal is None:
            return False
        return self._thermal.is_online()

    def mount_distance_m(self) -> float | None:
        """Return measured centre-pixel depth in metres (startup sanity check)."""
        return self._d435.get_center_depth_m()


# ── convenience loader ────────────────────────────────────────────────────────

def load_camera_config(path: str = "config/camera.yaml") -> dict:
    """Load and return the full camera.yaml as a dict."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
