"""
d435_stream.py — Intel RealSense D435 acquisition
Locks exposure and white balance, applies the High Accuracy depth preset,
and emits FrameBundle objects via a generator.

Settings come from config/camera.yaml (d435 block):
  RGB   640x480 @ 30 fps, bgr8
  Depth 424x240 @ 30 fps, z16, aligned to the RGB grid
  exposure = 156, white balance = 4600 (LOCKED: same as the subject recordings)
  depth visual preset = 3 (High Accuracy, same as record_session.py)
"""

from __future__ import annotations
import time
import numpy as np
import pyrealsense2 as rs
from src.types import FrameBundle


class D435Stream:
    """
    Manages the RealSense D435 pipeline lifecycle.

    Usage:
        stream = D435Stream(cfg["d435"])
        with stream:
            for bundle in stream.frames():
                process(bundle)
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: the d435 sub-dict from camera.yaml (loaded with yaml.safe_load)
        """
        self.cfg = cfg
        self.pipeline: rs.pipeline | None = None
        self.align: rs.align | None = None
        self.depth_scale: float = 0.001
        self._running = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the pipeline, lock sensor settings, wait for the hardware to settle."""
        self.pipeline = rs.pipeline()
        config = rs.config()

        rgb = self.cfg["rgb"]
        depth = self.cfg["depth"]

        config.enable_stream(
            rs.stream.color,
            rgb["width"], rgb["height"],
            rs.format.bgr8, rgb["fps"]
        )
        config.enable_stream(
            rs.stream.depth,
            depth["width"], depth["height"],
            rs.format.z16, depth["fps"]
        )

        # Align depth frames to the RGB lens perspective
        self.align = rs.align(rs.stream.color)

        print("[D435] Starting RealSense pipeline...")
        profile = self.pipeline.start(config)

        # ── sensor identification ─────────────────────────────────────────────
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # High Accuracy preset: the same one record_session.py applied during recording
        preset = depth.get("visual_preset")
        if preset is not None:
            depth_sensor.set_option(rs.option.visual_preset, preset)

        color_sensor = self._get_color_sensor(device)

        # ── lock exposure and white balance ───────────────────────────────────
        # Must happen AFTER pipeline.start(). Auto-exposure creates brightness
        # swings that look exactly like a pulse signal to rPPG.
        print("[D435] Locking RGB sensor settings...")
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, self.cfg["locked_exposure"])
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, self.cfg["locked_white_balance"])

        self._print_diagnostics(device, color_sensor, depth_sensor)

        # Let the hardware registers settle before streaming
        time.sleep(1.0)
        self._running = True
        print("[D435] Stream ready.")

    def stop(self) -> None:
        """Stop the pipeline cleanly."""
        self._running = False
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        print("[D435] Pipeline stopped.")

    def __enter__(self) -> "D435Stream":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── frame generator ───────────────────────────────────────────────────────

    def frames(self):
        """
        Generator that yields one FrameBundle per synchronised colour+depth pair.
        Waits up to 5 s per frame; incomplete pairs are skipped silently.

        Yields:
            FrameBundle with rgb, depth (aligned to RGB), ir=None, thermal=None.
            thermal is filled in later by SyncBuffer.
        """
        if not self._running or self.pipeline is None:
            raise RuntimeError("Call start() before iterating frames.")

        while self._running:
            try:
                raw_frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # Timeout: hardware glitch or USB bandwidth spike
                print("[D435] WARN: frame timeout, skipping tick.")
                continue

            aligned = self.align.process(raw_frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue  # incomplete pair

            rgb   = np.asanyarray(color_frame.get_data())    # (480, 640, 3) uint8
            depth = np.asanyarray(depth_frame.get_data())    # (480, 640)    uint16

            yield FrameBundle(
                rgb=rgb,
                depth=depth,
                ir=None,         # IR stream not enabled
                thermal=None,    # filled in by sync_buffer
                ts_mono=time.monotonic(),
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_color_sensor(device: rs.device) -> rs.sensor:
        """Return the RGB camera sensor, raise if it is not found."""
        for sensor in device.query_sensors():
            if "RGB Camera" in sensor.get_info(rs.camera_info.name):
                return sensor
        raise RuntimeError(
            "[D435] RGB Camera sensor not found. "
            "Check the USB connection and firmware version."
        )

    def _print_diagnostics(self, device, color_sensor, depth_sensor) -> None:
        """Print a hardware verification summary after the settings are applied."""
        print("-" * 55)
        print(f"  Device Name      : {device.get_info(rs.camera_info.name)}")
        print(f"  Serial Number    : {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware Version : {device.get_info(rs.camera_info.firmware_version)}")
        print(f"  Depth Scale      : {self.depth_scale:.6f} m/unit")
        print(f"  Depth Preset     : {depth_sensor.get_option(rs.option.visual_preset)}"
              f"  (expected {self.cfg['depth'].get('visual_preset')})")
        print(f"  Auto-Exposure    : {color_sensor.get_option(rs.option.enable_auto_exposure)}"
              "  (must be 0)")
        print(f"  Exposure         : {color_sensor.get_option(rs.option.exposure)}"
              f"  (must be {self.cfg['locked_exposure']})")
        print(f"  Auto-WB          : {color_sensor.get_option(rs.option.enable_auto_white_balance)}"
              "  (must be 0)")
        print(f"  White Balance    : {color_sensor.get_option(rs.option.white_balance)}"
              f"  (must be {self.cfg['locked_white_balance']})")
        print("-" * 55)

    def get_center_depth_m(self) -> float | None:
        """
        One-shot depth reading at the image centre, in metres.
        Handy for checking the mount distance at startup.
        Returns None if the pipeline is not running.
        """
        if not self._running or self.pipeline is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if depth_frame:
                return depth_frame.get_distance(320, 240)
        except RuntimeError:
            pass
        return None
