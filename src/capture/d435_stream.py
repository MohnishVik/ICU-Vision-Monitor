"""
d435_stream.py — Intel RealSense D435 acquisition
Locks exposure and white balance, applies the High Accuracy depth preset,
and emits FrameBundle objects via a generator.

Settings come from config/camera.yaml (d435 block):
  RGB   640x480 @ 30 fps, bgr8
  Depth 424x240 @ 30 fps, z16, aligned to the RGB grid
  exposure / gain / white balance LOCKED for the run (auto_lock meters the room
  once at startup; fixed uses the configured values)
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
        self.locked: dict = {}
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

        # The IR dot projector gives the two IR cameras texture to match on plain surfaces
        # (gowns, blankets); without it chest depth gets holes and noise. The device keeps
        # whatever another app (e.g. RealSense Viewer) last set, so set it every start.
        if self._supports(depth_sensor, rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if depth.get("emitter_enabled", True) else 0)
        power = depth.get("laser_power")
        if power is not None and self._supports(depth_sensor, rs.option.laser_power):
            self._set(depth_sensor, rs.option.laser_power, float(power))

        color_sensor = self._get_color_sensor(device)

        # ── lock exposure and white balance ───────────────────────────────────
        # Must happen AFTER pipeline.start(). Auto-exposure creates brightness
        # swings that look exactly like a pulse signal to rPPG, so it is never
        # left running: auto_lock meters the room once, then freezes the values.
        print("[D435] Locking RGB sensor settings...")
        self.locked = self._lock_rgb_settings(color_sensor, rgb["fps"])

        self._print_diagnostics(device, color_sensor, depth_sensor)

        # Let the hardware registers settle before streaming
        time.sleep(1.0)
        self._running = True
        print("[D435] Stream ready.")

    def stop(self) -> None:
        """Stop the pipeline cleanly."""
        self._running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass        # start() never completed (no device): nothing to stop
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

        timeouts = 0
        while self._running:
            try:
                raw_frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                timeouts = 0
            except RuntimeError:
                # One timeout is a glitch or USB bandwidth spike; three in a row (15 s) means
                # the camera is gone, so raise and let the engine reconnect.
                timeouts += 1
                if timeouts >= 3:
                    raise RuntimeError("D435 delivered no frames for 15 s (unplugged?)")
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

    def _lock_rgb_settings(self, color_sensor: rs.sensor, fps: int) -> dict:
        """
        Freeze exposure, gain and white balance; return the values applied.

        auto_lock: let the camera's own auto-exposure and auto white balance run for
        auto_lock_settle_s and note how bright the picture became and its colour
        cast. Many hosts (Windows without the RealSense metadata driver patch) do
        not report the exposure the camera chose, so the result is reproduced by
        measurement instead: with auto off, exposure/gain are stepped until the frame
        is as bright as auto made it, and white balance is searched until the
        red/blue balance matches. fixed: use the configured values.
        Exposure is capped at one frame period so the stream stays at `fps`.
        """
        mode = self.cfg.get("exposure_mode", "fixed")
        exposure = float(self.cfg["locked_exposure"])
        gain = self.cfg.get("locked_gain")
        wb = float(self.cfg["locked_white_balance"])
        e_max = 10000.0 / max(int(fps), 1)          # exposure units are 0.1 ms

        if self._supports(color_sensor, rs.option.auto_exposure_priority):
            # 0 = keep the frame rate; never stretch exposure past a frame
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
        # gamma/brightness persist in the camera between runs: start every run neutral
        self._reset_tone(color_sensor)

        applied = {"mode": mode}
        if mode == "auto_lock":
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            settle_until = time.monotonic() + float(self.cfg.get("auto_lock_settle_s", 2.5))
            frames = []
            while time.monotonic() < settle_until:
                img = self._next_color()
                if img is not None:
                    frames = (frames + [img])[-5:]
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

            if frames:
                auto_luma = float(np.mean([self._luma(f) for f in frames]))
                auto_rb = float(np.mean([self._red_blue(f) for f in frames]))
                target = self.cfg.get("target_luma") or auto_luma
                target = float(np.clip(target, 30.0, 200.0))
                gain_floor = float(gain) if gain is not None else 64.0
                exposure, gain = self._match_brightness(color_sensor, target, exposure,
                                                        gain_floor, gain_floor, e_max)
                applied.update(self._lift_tone(color_sensor, target))
                wb = self._match_white_balance(color_sensor, auto_rb, wb)
                applied.update(auto_luma=round(auto_luma, 1), target_luma=round(target, 1))
            else:
                print("[D435] WARN: no frames while metering; using fixed values.")

        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        for name, option, value in (("exposure", rs.option.exposure, min(exposure, e_max)),
                                    ("gain", rs.option.gain, gain),
                                    ("white_balance", rs.option.white_balance, wb)):
            if value is None or not self._supports(color_sensor, option):
                continue
            applied[name] = self._set(color_sensor, option, value)
        return applied

    # ── metering by measurement ───────────────────────────────────────────────

    def _match_brightness(self, sensor, target: float, exposure: float, gain: float,
                          gain_floor: float, e_max: float) -> tuple[float, float]:
        """
        Step exposure (first) and gain (only once exposure is at its cap) until the
        mean luma is within 6 % of `target`. Brightness is close to proportional to
        exposure x gain, so each step scales by target / measured.
        """
        has_gain = self._supports(sensor, rs.option.gain)
        g_max = sensor.get_option_range(rs.option.gain).max if has_gain else gain
        for _ in range(8):
            exposure = self._set(sensor, rs.option.exposure, min(exposure, e_max))
            if has_gain:
                gain = self._set(sensor, rs.option.gain, gain)
            luma = self._measure(self._luma)
            if luma is None or abs(luma - target) <= 0.06 * target:
                break
            want = exposure * gain * float(np.clip(target / max(luma, 1.0), 0.25, 4.0))
            if want > exposure * gain:                     # brighter: exposure, then gain
                exposure = min(want / gain, e_max)
                if has_gain and exposure >= e_max:
                    gain = min(want / exposure, g_max)
            else:                                          # darker: gain down to floor, then exposure
                if has_gain and gain > gain_floor:
                    gain = max(want / exposure, gain_floor)
                exposure = max(want / gain, 1.0)
            if exposure >= e_max and gain >= g_max and luma < target:
                exposure, gain = e_max, g_max              # as bright as 30 fps allows
                self._set(sensor, rs.option.exposure, exposure)
                if has_gain:
                    self._set(sensor, rs.option.gain, gain)
                break
        return exposure, gain

    def _reset_tone(self, sensor) -> None:
        """Gamma and brightness back to the configured (neutral) values."""
        for key, option, default in (("locked_gamma", rs.option.gamma, 300),
                                     ("locked_brightness", rs.option.brightness, 0)):
            if self._supports(sensor, option):
                self._set(sensor, option, self.cfg.get(key, default))

    def _lift_tone(self, sensor, target: float) -> dict:
        """
        In a dim room the camera's auto mode is brighter than any manual exposure x gain
        at full frame rate (it adds an internal boost it does not expose). When the
        picture is still too dark at the exposure and gain caps, raise gamma (lifts
        shadows and mid-tones) and then brightness (a constant offset) just enough.
        Both are fixed for the run, so they add no flicker.
        """
        lifted = {}
        luma = self._measure(self._luma)
        for name, option in (("gamma", rs.option.gamma), ("brightness", rs.option.brightness)):
            if luma is None or luma >= 0.94 * target or not self._supports(sensor, option):
                break
            rng = sensor.get_option_range(option)
            lo, hi = sensor.get_option(option), float(rng.max)
            best = lo
            for _ in range(6):
                mid = self._set(sensor, option, (lo + hi) / 2.0)
                luma = self._measure(self._luma)
                if luma is None:
                    break
                if luma < target:
                    lo, best = mid, mid
                else:
                    hi = mid
                if abs(luma - target) <= 0.06 * target or hi - lo <= max(rng.step, 1.0):
                    best = mid
                    break
            if luma is not None and luma < target:
                best = hi                                   # still short: take the brighter end
            lifted[name] = self._set(sensor, option, best)
            luma = self._measure(self._luma)
        return lifted

    def _match_white_balance(self, sensor, target_rb: float, wb: float) -> float:
        """Bisect the white-balance setting until red/blue matches what auto WB produced."""
        if not self._supports(sensor, rs.option.white_balance):
            return wb
        rng = sensor.get_option_range(rs.option.white_balance)
        lo, hi = float(rng.min), float(rng.max)

        def ratio_at(k: float) -> float | None:
            self._set(sensor, rs.option.white_balance, k)
            return self._measure(self._red_blue)

        r_lo, r_hi = ratio_at(lo), ratio_at(hi)
        if r_lo is None or r_hi is None or r_lo == r_hi:
            return wb
        rising = r_hi > r_lo
        if (target_rb <= min(r_lo, r_hi)) or (target_rb >= max(r_lo, r_hi)):
            return lo if abs(target_rb - r_lo) < abs(target_rb - r_hi) else hi
        for _ in range(8):
            mid = (lo + hi) / 2.0
            r = ratio_at(mid)
            if r is None:
                break
            if (r < target_rb) == rising:
                lo = mid
            else:
                hi = mid
            if hi - lo <= max(rng.step, 10.0) * 2:
                break
        return (lo + hi) / 2.0

    def _next_color(self) -> np.ndarray | None:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError:
            return None
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def _measure(self, stat, skip: int = 4, keep: int = 2) -> float | None:
        """Let a new setting reach the sensor (`skip` frames), then average `keep` frames."""
        for _ in range(skip):
            self._next_color()
        vals = [stat(img) for img in (self._next_color() for _ in range(keep)) if img is not None]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _luma(bgr: np.ndarray) -> float:
        m = bgr.reshape(-1, 3).mean(axis=0)
        return float(0.114 * m[0] + 0.587 * m[1] + 0.299 * m[2])

    @staticmethod
    def _red_blue(bgr: np.ndarray) -> float:
        m = bgr.reshape(-1, 3).mean(axis=0)
        return float((m[2] + 1.0) / (m[0] + 1.0))

    @staticmethod
    def _set(sensor, option, value: float) -> float:
        """Clamp to the option's range, snap to its step, write it, return what was written."""
        rng = sensor.get_option_range(option)
        value = float(np.clip(value, rng.min, rng.max))
        if rng.step > 0:
            value = rng.min + round((value - rng.min) / rng.step) * rng.step
        sensor.set_option(option, value)
        return value

    @staticmethod
    def _supports(sensor: rs.sensor, option) -> bool:
        try:
            return bool(sensor.supports(option))
        except RuntimeError:
            return False
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
        if self._supports(depth_sensor, rs.option.emitter_enabled):
            print(f"  IR projector     : {'on' if depth_sensor.get_option(rs.option.emitter_enabled) else 'OFF'}"
                  + (f", {depth_sensor.get_option(rs.option.laser_power):.0f} mW"
                     if self._supports(depth_sensor, rs.option.laser_power) else ""))
        print(f"  Auto-Exposure    : {color_sensor.get_option(rs.option.enable_auto_exposure)}"
              "  (must be 0)")
        print(f"  Exposure mode    : {self.locked.get('mode')}")
        print(f"  Exposure         : {color_sensor.get_option(rs.option.exposure)}"
              f"  (locked at {self.locked.get('exposure')})")
        print(f"  Gain             : {self.locked.get('gain', 'n/a')}")
        if "auto_luma" in self.locked:
            print(f"  Brightness       : target {self.locked['target_luma']} (camera auto reached "
                  f"{self.locked['auto_luma']}), gamma {self.locked.get('gamma', 'default')}, "
                  f"offset {self.locked.get('brightness', 'default')}")
        print(f"  Auto-WB          : {color_sensor.get_option(rs.option.enable_auto_white_balance)}"
              "  (must be 0)")
        print(f"  White Balance    : {color_sensor.get_option(rs.option.white_balance)}"
              f"  (locked at {self.locked.get('white_balance')})")
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