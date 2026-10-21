"""
engine.py — background engines that feed the web API.

    CameraEngine  one D435 (+ thermal) camera -> one patient. Runs the full
                  pipeline in a background thread:
                      SyncBuffer -> pose -> tracker -> ROIs -> RR/HR/Fall/TAA
                      -> VitalsPipeline (fusion, EWS, alarms) -> snapshot
                  Calibration happens in the browser: the operator clicks the
                  four chair corners on a still frame (POST /calibration).

    DemoEngine    SIMULATED patients for UI development without hardware.
                  Every snapshot carries simulated=true and the UI labels it.
                  Simulated measurements still go through the real fusion,
                  EWS and alarm code.

Robustness rules (the server must never crash because hardware or weights are
missing):
  * every module is loaded in isolation; a failure marks it "unavailable" with
    the reason and the rest keep running;
  * no camera / no pyrealsense2 -> status "camera_offline", retried every 5 s;
  * device "cuda" without CUDA -> pose falls back to CPU, reported in status;
  * D435 frames point into recycled SDK buffers, so they are copied before
    being shared with the web threads.
"""
from __future__ import annotations

import importlib
import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.fusion import load_rules
from src.scene import ChairCalibration, load_chair_calibration, load_privacy_guard
from src.scene.pose import PosePerson
from src.serving.state import SceneInfo, VitalsPipeline, to_jsonable
from src.types import FrameBundle, Measurement

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.engine")

STREAM_FPS = 12.0
PREVIEW_PERIOD_S = 1.0          # keep a 1 Hz preview even with no viewers (calibration, thumbnails)


def _read_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _encode_jpeg(img: np.ndarray, quality: int = 72) -> Optional[bytes]:
    if cv2 is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def _validate_corners(corners: Any, h: int, w: int) -> List[List[float]]:
    if not isinstance(corners, (list, tuple)) or len(corners) != 4:
        raise ValueError("Exactly 4 corners are required.")
    pts = []
    for c in corners:
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise ValueError("Each corner must be [x, y].")
        x, y = float(c[0]), float(c[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Corner coordinates must be finite numbers.")
        if not (0 <= x <= w and 0 <= y <= h):
            raise ValueError(f"Corner ({x:.0f}, {y:.0f}) is outside the {w}x{h} frame.")
        pts.append([x, y])
    a = np.array(pts)
    area = 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1)))
    if area < 0.01 * h * w:
        raise ValueError("The four corners enclose almost no area - click the corners of the seat.")
    return pts


# =========================================================================== #
# Base
# =========================================================================== #
class PatientSource:
    mode = "base"

    def __init__(self, patient_id: str, label: str, config_dir: str, simulated: bool):
        self.patient_id, self.label = patient_id, label
        self.config_dir = Path(config_dir)
        self.rules = load_rules(self.config_dir / "ews_rules.yaml")
        self.signal_cfg = _read_yaml(self.config_dir / "signal.yaml")
        self.pipeline = VitalsPipeline(patient_id, label, self.rules, self.signal_cfg,
                                       mode=self.mode, simulated=simulated)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._jpeg_version = 0
        self._jpeg_lock = threading.Lock()
        self._last_encode = 0.0
        self.stream_clients = 0
        self._chair: Optional[ChairCalibration] = None

    # lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, name=f"engine-{self.patient_id}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - last line of defence, never kill the server
            log.exception("Engine %s crashed: %s", self.patient_id, exc)
            self.pipeline.touch_status("error", f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # reads
    def snapshot(self) -> dict:
        return self.pipeline.snapshot()

    def history(self, minutes: float, max_points: int) -> dict:
        return self.pipeline.history(minutes, max_points)

    def acknowledge(self, alarm_id: Optional[int]) -> int:
        return self.pipeline.acknowledge(alarm_id)

    def latest_jpeg(self) -> tuple[Optional[bytes], int]:
        with self._jpeg_lock:
            return self._jpeg, self._jpeg_version

    def _publish_jpeg(self, data: Optional[bytes]) -> None:
        if data is None:
            return
        with self._jpeg_lock:
            self._jpeg = data
            self._jpeg_version += 1

    def _want_frame(self, now: float) -> bool:
        period = (1.0 / STREAM_FPS) if self.stream_clients > 0 else PREVIEW_PERIOD_S
        if now - self._last_encode >= period:
            self._last_encode = now
            return True
        return False

    # calibration (overridden where it matters)
    def calibration_status(self) -> dict:
        c = self._chair
        return to_jsonable({
            "calibrated": c is not None,
            "corners": c.corners.tolist() if c is not None else None,
            "seat_depth_m": c.seat_depth_m if c is not None else None,
            "label": c.label if c is not None else None,
            "image_size": self.pipeline.scene.image_size,
            "has_saved": self._saved_path().exists(),
        })

    def _saved_path(self) -> Path:
        return self.config_dir / "sessions" / f"chair_{self.patient_id}.json"

    def set_calibration(self, corners: Any, label: str = "chair") -> dict:
        raise NotImplementedError

    def reuse_calibration(self) -> dict:
        raise NotImplementedError

    def clear_calibration(self) -> dict:
        raise NotImplementedError


# =========================================================================== #
# Real camera
# =========================================================================== #
class CameraEngine(PatientSource):
    mode = "camera"

    def __init__(self, patient_id: str, label: str, config_dir: str = "config",
                 fps: float = 30.0, reuse_calibration: bool = False):
        super().__init__(patient_id, label, config_dir, simulated=False)
        self.fps = float(fps)
        self.reuse_on_start = reuse_calibration
        privacy_cfg = (self.signal_cfg.get("privacy") or {})
        self.stream_video = bool(privacy_cfg.get("stream_video", True))
        self.guard = load_privacy_guard(self.config_dir / "signal.yaml")
        self._pose = None
        self._mods: Dict[str, Any] = {}
        self._tracker = None
        self._roi_builder = None
        self._buf = None
        self._frame_lock = threading.Lock()
        self._last_bundle: Optional[FrameBundle] = None
        self._last_people: List[PosePerson] = []
        self._pose_errors = 0
        self._fps_ema = 0.0
        self._lat_ema = 0.0
        self._last_tick = 0.0
        self._last_fall_save = 0.0
        self._thermal_online = False

    # -------------------------------------------------------------- modules --
    def _set_mod(self, name: str, state: str, detail: str = "") -> None:
        prev = self.pipeline.modules.get(name, {})
        self.pipeline.modules[name] = {"ok": state == "ok", "state": state, "detail": detail,
                                       "errors": prev.get("errors", 0)}

    def _load_components(self) -> None:
        models_yaml = str(self.config_dir / "models.yaml")
        signal_yaml = str(self.config_dir / "signal.yaml")

        # pose ------------------------------------------------------------------
        try:
            from src.scene import load_pose_estimator
            est = load_pose_estimator(models_yaml)
            note = ""
            if str(est.device).startswith("cuda") or est.device == 0:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        est.device, note = "cpu", "CUDA not available - running pose on CPU."
                except Exception:  # noqa: BLE001
                    est.device, note = "cpu", "torch not importable - running pose on CPU."
            if not Path(est.weights).exists():
                self._set_mod("pose", "unavailable", f"Weights not found: {est.weights}")
            else:
                try:
                    import ultralytics  # noqa: F401
                    self._pose = est
                    self._set_mod("pose", "ok", note or f"{Path(est.weights).name} on {est.device}")
                except Exception as exc:  # noqa: BLE001
                    self._set_mod("pose", "unavailable", f"ultralytics not installed ({exc})")
        except Exception as exc:  # noqa: BLE001
            self._set_mod("pose", "unavailable", f"{type(exc).__name__}: {exc}")

        # signal modules ---------------------------------------------------------
        mod = importlib.import_module    # imported lazily: torch/scipy load only when needed
        loaders = {
            "rr": lambda: mod("src.modules.rr").load_rr_module(signal_yaml, depth_fs=self.fps),
            "hr": lambda: mod("src.modules.hr").load_hr_module(signal_yaml, fs=self.fps),
            "taa": lambda: mod("src.modules.taa").load_taa_module(signal_yaml, fs=self.fps),
            "fall": lambda: mod("src.modules.fall").load_fall_module(signal_yaml, local_pose=False),
        }
        for name, fn in loaders.items():
            try:
                self._mods[name] = fn()
                self._set_mod(name, "ok", "")
            except FileNotFoundError as exc:
                self._set_mod(name, "unavailable", f"Weights missing - {exc}")
            except Exception as exc:  # noqa: BLE001
                self._set_mod(name, "unavailable", f"{type(exc).__name__}: {exc}")
        if "fall" in self._mods and self._pose is None:
            self._set_mod("fall", "degraded", "Running without pose: needs yolov8n-pose.pt for "
                                              "keypoints.")
        self._set_mod("pain", "pending", "src/modules/pain.py is a scaffold - no weights yet.")
        # report the true redaction state from the start, not the optimistic default
        self.pipeline.scene.redaction = ("disabled" if not self.stream_video
                                         else ("faces" if self._pose is not None else "unavailable"))

    # ---------------------------------------------------------- calibration --
    def _install_chair(self, chair: ChairCalibration) -> None:
        from src.scene import load_person_tracker, load_roi_builder
        signal_yaml = str(self.config_dir / "signal.yaml")
        tracker = load_person_tracker(chair, signal_yaml, image_size=chair.image_size
                                      if chair.image_size != (0, 0) else None)
        roi_builder = load_roi_builder(signal_yaml, patient_id=self.patient_id)
        # single assignment each; the frame loop reads all three at the top of a tick
        self._tracker, self._roi_builder, self._chair = tracker, roi_builder, chair

    def set_calibration(self, corners: Any, label: str = "chair") -> dict:
        with self._frame_lock:
            b = self._last_bundle
        if b is None:
            raise RuntimeError("No camera frame yet - wait for the video to appear, then click.")
        h, w = b.rgb.shape[:2]
        pts = _validate_corners(corners, h, w)
        chair = ChairCalibration.from_corners(pts, depth=b.depth, image_size=(h, w),
                                              label=label if label in ("chair", "bed") else "chair",
                                              depth_already_meters=False)
        chair.save(self._saved_path())
        chair.save(self.config_dir / "sessions" / "chair_calibration.json")   # CLI default path
        self._install_chair(chair)
        self.pipeline.touch_status("running", "")
        return self.calibration_status()

    def reuse_calibration(self) -> dict:
        p = self._saved_path()
        if not p.exists():
            legacy = self.config_dir / "sessions" / "chair_calibration.json"
            if not legacy.exists():
                raise FileNotFoundError("No saved calibration for this bed yet.")
            p = legacy
        self._install_chair(load_chair_calibration(p))
        self.pipeline.touch_status("running", "")
        return self.calibration_status()

    def clear_calibration(self) -> dict:
        self._tracker, self._roi_builder, self._chair = None, None, None
        self.pipeline.touch_status("awaiting_calibration", "Click the four chair corners to start.")
        return self.calibration_status()

    # ------------------------------------------------------------------ run --
    def _run(self) -> None:
        self.pipeline.touch_status("starting", "Loading models...")
        self._load_components()
        if self.reuse_on_start:
            try:
                self.reuse_calibration()
            except Exception as exc:  # noqa: BLE001
                log.info("No saved calibration to reuse (%s)", exc)

        while not self._stop.is_set():
            try:
                from src.capture.sync_buffer import SyncBuffer, load_camera_config
            except Exception as exc:  # noqa: BLE001
                self.pipeline.touch_status(
                    "camera_offline",
                    f"Camera drivers unavailable ({type(exc).__name__}: {exc}). Install pyrealsense2, "
                    f"or start with --demo to explore the dashboard.")
                self._stop.wait(5.0)
                continue
            try:
                cfg = load_camera_config(str(self.config_dir / "camera.yaml"))
                self._buf = SyncBuffer(cfg)
                self.pipeline.touch_status("starting", "Opening RealSense D435...")
                self._buf.start()
            except Exception as exc:  # noqa: BLE001
                self.pipeline.touch_status("camera_offline",
                                           f"Could not open the D435 ({type(exc).__name__}: {exc}). "
                                           f"Retrying every 5 s.")
                self._safe_stop_buf()
                self._stop.wait(5.0)
                continue

            if self._chair is None:
                self.pipeline.touch_status("awaiting_calibration",
                                           "Click the four chair corners on the live frame.")
            try:
                for bundle in self._buf.frames():
                    if self._stop.is_set():
                        break
                    self._on_frame(bundle)
            except Exception as exc:  # noqa: BLE001
                log.exception("Capture loop error: %s", exc)
                self.pipeline.touch_status("camera_offline", f"Stream lost ({type(exc).__name__}: "
                                                             f"{exc}). Reconnecting...")
            finally:
                self._safe_stop_buf()
            self._stop.wait(2.0)

    def stop(self) -> None:
        super().stop()
        self._safe_stop_buf()

    def _safe_stop_buf(self) -> None:
        buf, self._buf = self._buf, None
        if buf is not None:
            try:
                buf.stop()
            except Exception:  # noqa: BLE001
                pass

    def _infer_pose(self, rgb: np.ndarray) -> List[PosePerson]:
        if self._pose is None:
            return []
        try:
            people = self._pose.infer(rgb)
            self._pose_errors = 0
            return people
        except Exception as exc:  # noqa: BLE001
            self._pose_errors += 1
            self._set_mod("pose", "error", f"{type(exc).__name__}: {exc}")
            if self._pose_errors >= 30:
                log.error("Pose failed 30 times in a row - disabling: %s", exc)
                self._pose = None
                self._set_mod("pose", "unavailable", f"Disabled after repeated errors: {exc}")
            return []

    def _on_frame(self, bundle: FrameBundle) -> None:
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(bundle.rgb).copy()        # SDK buffers are recycled
        depth = np.ascontiguousarray(bundle.depth).copy()
        thermal = None if bundle.thermal is None else np.array(bundle.thermal, copy=True)
        b = FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=thermal,
                        ts_mono=bundle.ts_mono, thermal_ts=bundle.thermal_ts)
        with self._frame_lock:
            self._last_bundle = b
        h, w = rgb.shape[:2]
        now = time.time()
        if now - self._last_tick > 2.0 and self._buf is not None:
            try:
                self._thermal_online = bool(self._buf.thermal_online())
            except Exception:  # noqa: BLE001
                self._thermal_online = False
        people = self._infer_pose(rgb)
        self._last_people = people
        redaction = ("disabled" if not self.stream_video
                     else ("faces" if self._pose is not None else "unavailable"))

        chair, tracker, roi_builder = self._chair, self._tracker, self._roi_builder
        if chair is None or tracker is None or roi_builder is None:
            self.pipeline.scene = SceneInfo(calibrated=False, image_size=[h, w],
                                            thermal_online=self._thermal_online, redaction=redaction,
                                            others=[p.bbox.tolist() for p in people])
            if self.pipeline.status != "awaiting_calibration":
                self.pipeline.touch_status("awaiting_calibration",
                                           "Click the four chair corners on the live frame.")
            else:
                self.pipeline.touch_status("awaiting_calibration", self.pipeline.status_detail)
            self._maybe_encode(rgb, people, now)
            return

        track = tracker.update(people, ts=b.ts_mono, image_size=(h, w))
        roi = roi_builder.build(b, track)
        raw: Dict[str, Measurement] = {}
        for name, mod in self._mods.items():
            try:
                out = mod.process(b, roi)
                if out:
                    raw.update(out)
                st = self.pipeline.modules.get(name, {})
                if st.get("state") == "error":
                    self._set_mod(name, "ok", "recovered")
            except Exception as exc:  # noqa: BLE001 - one module must not stop the others
                st = self.pipeline.modules.setdefault(name, {})
                st["errors"] = int(st.get("errors", 0)) + 1
                if st["errors"] in (1, 10, 100) or st["errors"] % 1000 == 0:
                    log.exception("Module %s error #%d: %s", name, st["errors"], exc)
                self._set_mod(name, "error", f"{type(exc).__name__}: {exc}")

        person = track.person
        scene = SceneInfo(
            has_patient=person is not None, in_chair_frac=float(track.in_chair_frac),
            staff_count=int(track.staff_count), lighting=str(roi.lighting),
            h_centroid=float(track.h_centroid), thermal_online=self._thermal_online, calibrated=True,
            image_size=[h, w],
            keypoints=person.keypoints.tolist() if person is not None else None,
            keypoints_conf=person.keypoints_conf.tolist() if person is not None else None,
            person_bbox=person.bbox.tolist() if person is not None else None,
            others=[p.bbox.tolist() for p in people if p is not person],
            chair=chair.corners.tolist(), redaction=redaction,
        )

        fall = raw.get("fall.classifier")
        if fall is not None and fall.meta.get("confirmed_fall") and now - self._last_fall_save > 10:
            self._last_fall_save = now
            try:
                self.guard.save_frame(rgb, f"fall_{self.patient_id}", person=person)
            except Exception as exc:  # noqa: BLE001
                log.warning("Fall snapshot not saved: %s", exc)

        dt = now - self._last_tick if self._last_tick else 0.0
        self._last_tick = now
        lat = (time.perf_counter() - t0) * 1000.0
        if dt > 0:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * (1.0 / dt) if self._fps_ema else 1.0 / dt
        self._lat_ema = 0.9 * self._lat_ema + 0.1 * lat if self._lat_ema else lat
        self.pipeline.metrics = {"fps": round(self._fps_ema, 1), "latency_ms": round(self._lat_ema, 1)}
        self.pipeline.status, self.pipeline.status_detail = "running", ""
        self.pipeline.step(now, raw, scene)
        self._maybe_encode(rgb, people, now)

    def _maybe_encode(self, rgb: np.ndarray, people: List[PosePerson], now: float) -> None:
        if not self.stream_video or not self._want_frame(now):
            return
        frame = self.guard.deidentify_many(rgb, people) if people else rgb
        self._publish_jpeg(_encode_jpeg(frame))

    def calibration_frame(self) -> Optional[bytes]:
        data, _ = self.latest_jpeg()
        if data is not None:
            return data
        with self._frame_lock:
            b = self._last_bundle
        if b is None:
            return None
        people = self._last_people
        return _encode_jpeg(self.guard.deidentify_many(b.rgb, people) if people else b.rgb)


# =========================================================================== #
# Demo (simulated) patients
# =========================================================================== #
_SEATED = np.array([                                              # frontal view, seated:
    [320, 148], [310, 140], [330, 140], [298, 146], [342, 146],      # head
    [282, 200], [358, 200], [266, 262], [374, 262], [296, 300], [344, 300],   # arms, hands in lap
    [294, 318], [346, 318], [272, 352], [368, 352], [278, 438], [362, 438],   # thighs foreshortened
], dtype=np.float32)
_LYING = np.array([
    [150, 418], [146, 410], [146, 426], [140, 404], [140, 432],
    [196, 398], [200, 440], [236, 380], [244, 452], [268, 390], [276, 446],
    [318, 404], [318, 436], [390, 402], [392, 438], [460, 404], [462, 436],
], dtype=np.float32)
_DEMO_CHAIR = [[226, 212], [414, 212], [436, 404], [204, 404]]
_SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


class DemoEngine(PatientSource):
    """Simulated bed. scenario: stable | deteriorating | fall | respiratory."""
    mode = "demo"

    def __init__(self, patient_id: str, label: str, scenario: str, config_dir: str = "config",
                 seed: int = 0, time_offset_s: float = 0.0):
        super().__init__(patient_id, label, config_dir, simulated=True)
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.t0 = time.time() - time_offset_s
        self._chair = ChairCalibration.from_corners(_DEMO_CHAIR, image_size=(480, 640), label="chair")
        self._last_emit = 0.0
        self._fall_emitted_cycle = -1
        self._sustain = 0.0
        for name in ("pose", "rr", "hr", "fall", "taa"):
            self.pipeline.modules[name] = {"ok": True, "state": "ok", "detail": "simulated",
                                           "errors": 0}
        self.pipeline.modules["pain"] = {"ok": False, "state": "pending",
                                         "detail": "src/modules/pain.py is a scaffold - no weights yet.",
                                         "errors": 0}

    # calibration endpoints work, but only move the drawn polygon
    def set_calibration(self, corners: Any, label: str = "chair") -> dict:
        pts = _validate_corners(corners, 480, 640)
        self._chair = ChairCalibration.from_corners(pts, image_size=(480, 640), label=label)
        return self.calibration_status()

    def reuse_calibration(self) -> dict:
        self._chair = ChairCalibration.from_corners(_DEMO_CHAIR, image_size=(480, 640))
        return self.calibration_status()

    def clear_calibration(self) -> dict:
        return self.reuse_calibration()      # a demo bed is always calibrated

    def calibration_frame(self) -> Optional[bytes]:
        return self._render(_SEATED, 0.0, 0.0)

    # ---------------------------------------------------------- simulation --
    def _truth(self, t: float) -> dict:
        n = self.rng.gauss
        s: dict = {"rr": 15.0, "hr": 74.0, "rr_sqi": 0.82, "hr_sqi": 0.74, "thermal": True,
                   "fall": 0, "confirm": False, "in_chair": 0.92, "pose": _SEATED, "taa": False,
                   "taa_phase": 0.35, "apnea": False, "staff": 0, "hr_ok": True, "lighting": "bright"}
        if self.scenario == "stable":
            s["rr"] = 15 + 0.8 * math.sin(t / 40) + n(0, 0.4)
            s["hr"] = 74 + 3 * math.sin(t / 55) + n(0, 1.2)
            if (t % 300) > 250:
                s["staff"] = 1                                    # a nurse visits
        elif self.scenario == "deteriorating":
            k = 0.0 if t < 90 else min(1.0, (t - 90) / 600)       # slow, one-way decline
            s["rr"] = 16 + 11 * k + n(0, 0.5)
            s["hr"] = 86 + 32 * k + n(0, 1.5)
            s["rr_sqi"] = 0.78 - 0.1 * k
        elif self.scenario == "fall":
            c = t % 900                                           # one fall every 15 min
            s["rr"] = 17 + n(0, 0.5)
            s["hr"] = 80 + n(0, 1.5)
            if 150 <= c < 152:
                s["fall"], s["in_chair"] = 1, 0.4
                s["pose"] = _SEATED + (_LYING - _SEATED) * ((c - 150) / 2.0)
            elif 152 <= c < 260:
                s["fall"], s["in_chair"], s["pose"] = 2, 0.0, _LYING
                s["hr"] = 104 + n(0, 2)
                cycle = int(t // 900)
                if c >= 153 and self._fall_emitted_cycle != cycle:
                    self._fall_emitted_cycle, s["confirm"] = cycle, True
            if 175 <= c < 260:
                s["staff"] = 2                                    # staff respond
        elif self.scenario == "respiratory":
            c = t % 240
            s["rr"] = 18 + n(0, 0.6)
            s["hr"] = 92 + n(0, 2)
            if 50 <= c < 120:
                s["taa"], s["taa_phase"], s["rr"] = True, 1.55 + n(0, 0.08), 24 + n(0, 0.6)
            if 140 <= c < 165:
                s["apnea"] = True
            if 190 <= c < 215:
                s["hr_ok"], s["thermal"], s["lighting"] = False, False, "dim"   # poor signal
        return s

    def _backfill(self, seconds: int = 1200) -> None:
        """Seed 20 min of simulated history so trends are meaningful immediately."""
        now = time.time()
        for k in range(seconds, 0, -1):
            ts = now - k
            s = self._truth(ts - self.t0)
            self.pipeline.backfill(ts, None if s["apnea"] else s["rr"], s["hr"] if s["hr_ok"] else None,
                                   s["rr_sqi"], s["hr_sqi"], s["fall"], bool(s["taa"]))

    def _run(self) -> None:
        self._backfill()
        self.pipeline.status, self.pipeline.status_detail = "running", ""
        while not self._stop.is_set():
            now = time.time()
            t = now - self.t0
            s = self._truth(t)
            raw: Dict[str, Measurement] = {}
            if now - self._last_emit >= 1.0:                     # modules report at 1 Hz
                self._last_emit = now
                if s["apnea"]:
                    raw["rr.depth"] = Measurement(None, 0.0, now, "rr.depth", 3.1,
                                                  {"apnea": True, "fs_hz": 30.0})
                else:
                    raw["rr.depth"] = Measurement(s["rr"], s["rr_sqi"], now, "rr.depth", 3.1,
                                                  {"apnea": False, "fs_hz": 30.0,
                                                   "peak_power_frac": s["rr_sqi"]})
                    if s["thermal"]:
                        raw["rr.thermal"] = Measurement(s["rr"] + self.rng.gauss(0, 0.7), 0.55, now,
                                                        "rr.thermal", 2.2, {"fs_hz": 8.0})
                raw["hr.pos"] = (Measurement(s["hr"], s["hr_sqi"], now, "hr.pos", 4.6, {"fs_hz": 30.0})
                                 if s["hr_ok"] else
                                 Measurement(None, 0.12, now, "hr.pos", 4.6,
                                             {"reason": "low_peak_power"}))
                self._sustain = self._sustain + 1.0 if s["taa"] else 0.0
                raw["taa.depth"] = Measurement(s["taa_phase"], 0.7, now, "taa.depth", 2.0,
                                               {"taa_flag": bool(s["taa"] and self._sustain >= 10),
                                                "sustain_s": self._sustain})
            raw["fall.classifier"] = Measurement(
                float(s["fall"]), 1.0, now, "fall.classifier", 6.0,
                {"patient_state": ("NORMAL", "FALLING", "LYING_DOWN")[s["fall"]],
                 "ml_probability": 0.94 if s["fall"] else 0.03, "provisional_active": s["fall"] == 1,
                 "confirmed_fall": ({"time": now, "confidence": 0.94} if s["confirm"] else None)})

            breath = 1.6 * math.sin(2 * math.pi * (s["rr"] / 60.0) * t)
            pose = s["pose"].copy()
            pose[5:7, 1] += breath
            jitter = np.array([[self.rng.gauss(0, 0.6), self.rng.gauss(0, 0.6)] for _ in range(17)])
            pose = pose + jitter
            x1, y1 = pose.min(axis=0) - 12
            x2, y2 = pose.max(axis=0) + 12
            others = [[470.0, 120.0, 560.0, 450.0]] if s["staff"] >= 1 else []
            if s["staff"] >= 2:
                others.append([60.0, 130.0, 150.0, 455.0])
            scene = SceneInfo(
                has_patient=True, in_chair_frac=s["in_chair"], staff_count=s["staff"],
                lighting=s["lighting"], h_centroid=float(pose[5:13, 1].mean() / 480.0),
                thermal_online=s["thermal"], calibrated=True, image_size=[480, 640],
                keypoints=pose.tolist(), keypoints_conf=[0.9] * 17,
                person_bbox=[float(x1), float(y1), float(x2), float(y2)], others=others,
                chair=self._chair.corners.tolist(), redaction="faces")
            self.pipeline.metrics = {"fps": round(29.6 + self.rng.gauss(0, 0.3), 1),
                                     "latency_ms": round(38 + self.rng.gauss(0, 3), 1)}
            self.pipeline.step(now, raw, scene)
            if self._want_frame(now):
                self._publish_jpeg(self._render(pose, s["staff"], t))
            self._stop.wait(0.1)

    def _render(self, pose: np.ndarray, staff: float, t: float) -> Optional[bytes]:
        """Schematic placeholder frame (no person imagery) so overlays have a backdrop."""
        if cv2 is None:
            return None
        img = np.full((480, 640, 3), (22, 17, 13), np.uint8)
        for x in range(0, 640, 32):
            cv2.line(img, (x, 0), (x, 479), (30, 24, 20), 1)
        for y in range(0, 480, 32):
            cv2.line(img, (0, y), (639, y), (30, 24, 20), 1)
        cv2.rectangle(img, (0, 462), (639, 479), (34, 28, 24), -1)
        for a, b in _SKELETON:
            cv2.line(img, tuple(int(v) for v in pose[a]), tuple(int(v) for v in pose[b]),
                     (70, 62, 56), 14, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in pose[0]), 24, (70, 62, 56), -1, cv2.LINE_AA)
        if staff:
            cv2.rectangle(img, (480, 140), (550, 440), (40, 34, 30), -1)
        cv2.putText(img, "SIMULATED FEED - NO CAMERA", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 110, 100), 1, cv2.LINE_AA)
        return _encode_jpeg(img, 70)


# =========================================================================== #
# Manager
# =========================================================================== #
DEMO_BEDS = (
    ("bed-1", "Bed 1", "stable"),
    ("bed-2", "Bed 2", "deteriorating"),
    ("bed-3", "Bed 3", "fall"),
    ("bed-4", "Bed 4", "respiratory"),
)


class EngineManager:
    def __init__(self, sources: List[PatientSource]):
        self.sources: Dict[str, PatientSource] = {s.patient_id: s for s in sources}
        self.started = time.time()

    @classmethod
    def from_args(cls, demo: bool, patient_id: str = "bed-1", label: str = "Bed 1",
                  config_dir: str = "config", fps: float = 30.0,
                  reuse_calibration: bool = False) -> "EngineManager":
        if demo:
            # stagger the scenarios so the ward looks lived-in from the first second
            offsets = {"stable": 400.0, "deteriorating": 300.0, "fall": 130.0, "respiratory": 30.0}
            return cls([DemoEngine(pid, lbl, sc, config_dir, seed=i, time_offset_s=offsets[sc])
                        for i, (pid, lbl, sc) in enumerate(DEMO_BEDS)])
        return cls([CameraEngine(patient_id, label, config_dir, fps, reuse_calibration)])

    @property
    def mode(self) -> str:
        return "demo" if any(isinstance(s, DemoEngine) for s in self.sources.values()) else "camera"

    def get(self, patient_id: str) -> PatientSource:
        return self.sources[patient_id]

    def start(self) -> None:
        for s in self.sources.values():
            s.start()

    def stop(self) -> None:
        for s in self.sources.values():
            s.stop()
