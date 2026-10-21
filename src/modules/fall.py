"""
fall.py — Fall Detection module (Option A: full port, no ROS / scripts dependency)

Ported from https://github.com/MohnishVik/Fall_Detection
  - FallGRUClassifier       (model.py)
  - 62-D feature extractor  (hybrid_detector.extract_62_features)
  - 3-state refined FSM     (fall_detector_refined.RefinedFallDetector)
  - Strategy H5 hybrid gate (hybrid_detector.process_feature_step)

Pipeline per tick
  1. Scene layer or local YOLO -> keypoints + bbox
  2. Kinematics derivation     -> aspect ratio, torso angle, v_down, bottom_y, raw_score
  3. 62-D feature vector       -> rolling 32-frame buffer
  4. FSM step                  -> NORMAL / FALLING / LYING_DOWN
  5. GRU inference every stride frames when buffer full -> provisional ML trigger
  6. Strategy H5 confirmation gate                        -> CONFIRMED or REJECTED

Output
  Measurement(value=state_code, sqi, ts, source="fall.classifier", meta={...})
  value codes: 0 NORMAL / 1 FALLING / 2 LYING_DOWN
  meta.confirmed_fall carries the full event dict when the gate confirms a fall.

Weights
  Primary:   models/public/fall_gru_pseudo.pt       (best_fall_gru.pt, production)
  Fallback:  models/public/fall_gru_pseudo_e1.pt    (best_model.pt, stronger standalone)
  YOLO:      models/public/yolov8n-pose.pt
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.types import FrameBundle, Measurement, ROIBundle

STATE_NORMAL     = 0
STATE_FALLING    = 1
STATE_LYING_DOWN = 2
STATE_NAME = {0: "NORMAL", 1: "FALLING", 2: "LYING_DOWN"}
NAME_STATE = {v: k for k, v in STATE_NAME.items()}


# ─────────────────────────────────────────────────────────────────────────────
# GRU classifier — IDENTICAL to Fall_Detection/scripts/model.py
# ─────────────────────────────────────────────────────────────────────────────

class FallGRUClassifier(nn.Module):
    """2-layer GRU, binary head. Input: [B, T, 62]. Output: logits [B]."""
    def __init__(self, input_size=62, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(p=dropout), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)          # [B, T, H]
        return self.classifier(out[:, -1, :]).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _KinCfg:
    upright_ar: float      = 1.20
    upright_angle: float   = 30.0
    horiz_ar: float        = 0.90
    horiz_angle: float     = 50.0
    vel_thresh: float      = 0.25
    ground_y: float        = 0.70
    min_fall_vel: float    = 0.15
    early_angle: float     = 22.0
    early_ar: float        = 1.25
    w_angle: float = 0.40; w_ar: float = 0.30; w_floor: float = 0.15; w_vel: float = 0.15


@dataclass
class _FSMCfg:
    upright_lookback: float = 1.20
    stab_vel: float         = 0.08
    stab_dur: float         = 0.60
    max_dur: float          = 3.50
    frame_thresh: float     = 0.50


@dataclass
class _MLCfg:
    temporal_window: int    = 32
    stride: int             = 4
    threshold: float        = 0.50
    postprocess_M: int      = 2             # consecutive +ve windows -> provisional
    confirmation_window_s: float = 1.0      # Strategy H5
    cooldown_s: float       = 5.0           # suppress new confirmed after a confirmed fall
    suppress_during_lying: bool = True


@dataclass
class FallConfig:
    kin:  _KinCfg = field(default_factory=_KinCfg)
    fsm:  _FSMCfg = field(default_factory=_FSMCfg)
    ml:   _MLCfg  = field(default_factory=_MLCfg)
    gru_weights_primary:   Path = Path("models/public/fall_gru_pseudo.pt")
    gru_weights_fallback:  Path = Path("models/public/fall_gru_pseudo_e1.pt")
    yolo_weights:          Path = Path("models/public/yolov8n-pose.pt")
    yolo_conf: float = 0.25
    yolo_iou:  float = 0.45
    yolo_img_size: int = 640
    device: str = "auto"

    @classmethod
    def from_dict(cls, cfg: dict) -> "FallConfig":
        f = cfg.get("fall", {}) if isinstance(cfg, dict) else {}
        out = cls()
        for k, v in f.get("kinematics", {}).items():
            if hasattr(out.kin, k): setattr(out.kin, k, float(v))
        for k, v in f.get("fsm", {}).items():
            if hasattr(out.fsm, k): setattr(out.fsm, k, float(v))
        for k, v in f.get("ml", {}).items():
            if hasattr(out.ml, k):
                setattr(out.ml, k, type(getattr(out.ml, k))(v))
        for pk in ("gru_weights_primary", "gru_weights_fallback", "yolo_weights"):
            if pk in f: setattr(out, pk, Path(f[pk]))
        for k in ("yolo_conf", "yolo_iou", "yolo_img_size", "device"):
            if k in f: setattr(out, k, type(getattr(out, k))(f[k]))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Kinematics extractor — port of RefinedFallDetector.analyze_frame_kinematics
# ─────────────────────────────────────────────────────────────────────────────

class _Kinematics:
    """
    Derives posture / velocity features from pose keypoints + bbox.
    Keeps a short centre-of-gravity history so v_down is well-defined.
    """

    def __init__(self, cfg: _KinCfg):
        self.cfg = cfg
        self.cog_history: List[Tuple[float, float]] = []     # (ts, cog_y_norm)
        self.last_valid: Optional[Dict[str, Any]] = None

    def reset(self):
        self.cog_history.clear()
        self.last_valid = None

    def analyze(self,
                kpts:  Optional[np.ndarray],       # (17, 2) pixel coords
                confs: Optional[np.ndarray],       # (17,)
                bbox:  Optional[np.ndarray],       # [x1, y1, x2, y2]
                frame_hw: Tuple[int, int],
                ts: float) -> Dict[str, Any]:
        cfg = self.cfg
        h, w = frame_hw
        default = {
            "timestamp_s": ts, "has_person": False, "box": None, "keypoints": None,
            "keypoints_conf": None,
            "ar": 1.5, "angle_deg": 10.0, "bottom_y": 0.5, "v_down": 0.0,
            "is_upright": True, "is_horizontal": False, "raw_score": 0.0,
        }

        if bbox is None:
            if self.last_valid:   # carry over last posture, decay score, zero velocity
                d = dict(default)
                for k in ("ar", "angle_deg", "bottom_y", "is_upright", "is_horizontal"):
                    d[k] = self.last_valid[k]
                d["raw_score"] = self.last_valid["raw_score"] * 0.95
                return d
            return default

        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
        ar = bh / bw
        cog_y = (y1 + y2) / (2.0 * h)
        bottom_y = y2 / float(h)

        self.cog_history.append((ts, cog_y))
        if len(self.cog_history) > 12: self.cog_history.pop(0)
        v_down = 0.0
        if len(self.cog_history) >= 4:
            dt = max(0.01, self.cog_history[-1][0] - self.cog_history[0][0])
            v_down = (self.cog_history[-1][1] - self.cog_history[0][1]) / dt

        has_kpts = False; angle_deg = 0.0
        if kpts is not None and confs is not None:
            sh = [kpts[i] for i in (5, 6) if confs[i] > 0.25 and kpts[i].sum() > 0]
            hp = [kpts[i] for i in (11, 12) if confs[i] > 0.25 and kpts[i].sum() > 0]
            if sh and hp:
                has_kpts = True
                dx = abs(float(np.mean(sh, axis=0)[0] - np.mean(hp, axis=0)[0]))
                dy = abs(float(np.mean(sh, axis=0)[1] - np.mean(hp, axis=0)[1]))
                angle_deg = float(np.degrees(np.arctan2(dx, dy + 1e-5)))

        is_upright    = (ar >= cfg.upright_ar) and (not has_kpts or angle_deg <= cfg.upright_angle)
        is_horizontal = ((ar <= cfg.horiz_ar) or (has_kpts and angle_deg >= cfg.horiz_angle)) \
                        and (bottom_y >= cfg.ground_y)

        s_ar = float(np.clip((cfg.upright_ar - ar) / (cfg.upright_ar - cfg.horiz_ar), 0, 1))
        if has_kpts:
            s_angle = float(np.clip((angle_deg - cfg.upright_angle) /
                                    (cfg.horiz_angle - cfg.upright_angle), 0, 1))
        else:
            s_angle = s_ar
        s_floor = float(np.clip((bottom_y - 0.50) / (cfg.ground_y - 0.50), 0, 1))
        s_vel   = float(np.clip(v_down / cfg.vel_thresh, 0, 1))
        if has_kpts:
            raw = (cfg.w_angle*s_angle + cfg.w_ar*s_ar + cfg.w_floor*s_floor + cfg.w_vel*s_vel)
        else:
            raw = ((cfg.w_ar + cfg.w_angle*0.5)*s_ar
                   + (cfg.w_floor + cfg.w_angle*0.25)*s_floor
                   + (cfg.w_vel + cfg.w_angle*0.25)*s_vel)

        data = {
            "timestamp_s": ts, "has_person": True,
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "keypoints": kpts, "keypoints_conf": confs,
            "ar": ar, "angle_deg": angle_deg, "bottom_y": bottom_y, "v_down": v_down,
            "is_upright": is_upright, "is_horizontal": is_horizontal,
            "raw_score": float(np.clip(raw, 0, 1)),
        }
        self.last_valid = data
        return data


# ─────────────────────────────────────────────────────────────────────────────
# 62-D feature extractor — IDENTICAL to hybrid_detector.extract_62_features
# ─────────────────────────────────────────────────────────────────────────────

class _Features62D:
    def __init__(self):
        self.last_valid_norm_box  = np.array([0.3, 0.3, 0.7, 0.7], dtype=np.float32)
        self.last_valid_kpts_norm = np.zeros((17, 2), dtype=np.float32)

    def reset(self):
        self.last_valid_norm_box  = np.array([0.3, 0.3, 0.7, 0.7], dtype=np.float32)
        self.last_valid_kpts_norm = np.zeros((17, 2), dtype=np.float32)

    def extract(self, frame_hw: Tuple[int, int], data: Dict[str, Any]) -> np.ndarray:
        h, w = frame_hw
        has = bool(data.get("has_person", False))

        if has and data.get("box") is not None:
            b = data["box"]
            box_norm = np.array([b[0]/max(1., w), b[1]/max(1., h),
                                 b[2]/max(1., w), b[3]/max(1., h)], dtype=np.float32)
            self.last_valid_norm_box = box_norm.copy()
        else:
            box_norm = self.last_valid_norm_box.copy()
        box_w = float(box_norm[2] - box_norm[0]); box_h = float(box_norm[3] - box_norm[1])

        if has and data.get("keypoints") is not None:
            raw = data["keypoints"]
            kpts_norm = np.zeros((17, 2), dtype=np.float32)
            kpts_norm[:, 0] = raw[:, 0] / max(1., w)
            kpts_norm[:, 1] = raw[:, 1] / max(1., h)
            confs = data["keypoints_conf"].astype(np.float32) if data.get("keypoints_conf") is not None \
                    else np.ones(17, dtype=np.float32)
            self.last_valid_kpts_norm = kpts_norm.copy()
        else:
            kpts_norm = self.last_valid_kpts_norm.copy()
            confs = np.zeros(17, dtype=np.float32)

        feat = np.zeros(62, dtype=np.float32)
        feat[0:34]  = kpts_norm.reshape(34)
        feat[34:51] = confs
        feat[51:55] = box_norm
        feat[55]    = box_w; feat[56] = box_h
        feat[57]    = float(data.get("ar", 1.5))
        feat[58]    = float(data.get("angle_deg", 10.0))
        feat[59]    = float(data.get("v_down", 0.0))
        feat[60]    = float(data.get("bottom_y", 0.5))
        feat[61]    = 1.0 if has else 0.0
        return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3-state FSM — port of RefinedFallDetector step logic
# ─────────────────────────────────────────────────────────────────────────────

class _RefinedFSM:
    def __init__(self, kin_cfg: _KinCfg, fsm_cfg: _FSMCfg):
        self.k, self.f = kin_cfg, fsm_cfg
        self.reset()

    def reset(self):
        self.state = STATE_NORMAL
        self.upright_hist: List[Tuple[float, bool]] = []
        self.downward_frames = 0
        self.fall_start_ts: Optional[float] = None
        self.fall_peak_score: float = 0.0
        self.stable_since_ts: Optional[float] = None

    def step(self, data: Dict[str, Any], ts: float, fps: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Advance the FSM by one tick. Returns an event dict on FALLING -> LYING_DOWN
        (i.e. heuristic confirmation that a fall completed), else None.
        """
        k, f = self.k, self.f

        self.upright_hist.append((ts, bool(data["is_upright"])))
        self.upright_hist = [it for it in self.upright_hist if it[0] >= ts - f.upright_lookback]
        was_upright = any(it[1] for it in self.upright_hist)

        meaningful = data["v_down"] >= k.min_fall_vel
        self.downward_frames = (self.downward_frames + 1) if meaningful else max(0, self.downward_frames - 1)
        vel_spike = data["v_down"] >= k.vel_thresh

        is_early_fall = (
            was_upright and meaningful
            and (data["angle_deg"] >= k.early_angle or data["ar"] <= k.early_ar)
            and self.downward_frames >= 2
        )

        event = None
        if self.state == STATE_NORMAL:
            if (data["raw_score"] >= f.frame_thresh and
                (vel_spike or (was_upright and data["is_horizontal"]))) \
               or (is_early_fall and (data["raw_score"] >= 0.35 or vel_spike)):
                self.state = STATE_FALLING
                self.fall_start_ts = ts
                self.fall_peak_score = max(data["raw_score"], 0.6)
                self.stable_since_ts = None
            elif data["is_horizontal"]:
                self.state = STATE_LYING_DOWN

        elif self.state == STATE_FALLING:
            self.fall_peak_score = max(self.fall_peak_score, data["raw_score"])
            dur = ts - (self.fall_start_ts or ts)

            if data["is_horizontal"] and abs(data["v_down"]) <= f.stab_vel:
                if self.stable_since_ts is None:
                    self.stable_since_ts = ts
                elif (ts - self.stable_since_ts) >= f.stab_dur:
                    self.state = STATE_LYING_DOWN
                    event = {"type": "FALL_STABILIZED_LYING",
                             "start": self.fall_start_ts, "end": ts,
                             "confidence": self.fall_peak_score}
            else:
                self.stable_since_ts = None

            if self.state == STATE_FALLING and dur >= f.max_dur:
                self.state = STATE_LYING_DOWN
                event = {"type": "FALL_TIMEOUT_LYING",
                         "start": self.fall_start_ts, "end": ts,
                         "confidence": self.fall_peak_score}

            if data["is_upright"]:
                self.state = STATE_NORMAL
                self.stable_since_ts = None
                self.fall_start_ts = None

        elif self.state == STATE_LYING_DOWN:
            if data["is_upright"]:
                self.state = STATE_NORMAL
                self.stable_since_ts = None

        return event, is_early_fall


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid gate — Strategy H5
# ─────────────────────────────────────────────────────────────────────────────

class _HybridGate:
    """Pairs provisional ML triggers with FSM confirmations inside a 1 s window."""
    def __init__(self, cfg: _MLCfg):
        self.cfg = cfg
        self.pos_hist: Deque[bool] = deque(maxlen=max(2, cfg.postprocess_M))
        self.provisional: List[Dict[str, Any]] = []
        self.recent_fsm_events: Deque[Dict[str, Any]] = deque(maxlen=16)
        self.recent_early: Deque[Dict[str, Any]] = deque(maxlen=16)
        self.last_confirmed_ts: Optional[float] = None
        self.event_log: List[Dict[str, Any]] = []

    def register_ml_prob(self, prob: float, ts: float):
        pos = prob >= self.cfg.threshold
        self.pos_hist.append(pos)
        if sum(self.pos_hist) >= self.cfg.postprocess_M and pos:
            if not self.provisional or (ts - self.provisional[-1]["ml_end_time"] > 0.5):
                self.provisional.append({"ml_trigger_time": ts, "ml_end_time": ts,
                                         "probabilities": [prob]})
            else:
                self.provisional[-1]["ml_end_time"] = ts
                self.provisional[-1]["probabilities"].append(prob)

    def register_fsm_event(self, event: Dict[str, Any]):
        self.recent_fsm_events.append(event)

    def register_early_fall(self, ts: float):
        self.recent_early.append({"start": ts - 0.1, "time": ts})

    def check(self, ts: float, fsm_state: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Returns (confirmed, rejected) events this tick, if any."""
        confirmed, rejected = None, None
        is_static = (fsm_state == STATE_LYING_DOWN
                     and self.last_confirmed_ts is not None
                     and (ts - self.last_confirmed_ts) < self.cfg.cooldown_s
                     and self.cfg.suppress_during_lying)

        if not is_static:
            for p in list(self.provisional):
                ms, me = p["ml_trigger_time"], p["ml_end_time"]
                matched = None
                for h in list(self.recent_fsm_events) + list(self.recent_early):
                    hs = h.get("start", h.get("time", ms))
                    he = h.get("end",   h.get("time", ms))
                    if max(hs, ms) <= min(max(he, hs + 0.1), me + self.cfg.confirmation_window_s):
                        matched = h; break
                if matched:
                    confirmed = {
                        "status": "CONFIRMED_FALL",
                        "ml_trigger_time": round(ms, 3),
                        "ml_end_time":     round(me, 3),
                        "final_alert_time": round(ms, 3),   # preserve ML timestamp
                        "peak_ml_probability":  round(max(p["probabilities"]), 4),
                    }
                    self.event_log.append(confirmed)
                    self.last_confirmed_ts = ts
                    self.provisional.clear()
                    self.pos_hist.clear()
                    break

        for p in list(self.provisional):
            if ts > p["ml_end_time"] + self.cfg.confirmation_window_s:
                rejected = {"status": "REJECTED", "ml_trigger_time": round(p["ml_trigger_time"], 3),
                            "peak_ml_probability": round(max(p["probabilities"]), 4)}
                self.event_log.append(rejected)
                self.provisional.remove(p)
        return confirmed, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Weight loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_gru(primary: Path, fallback: Path, device: torch.device) -> Tuple[FallGRUClassifier, Dict]:
    """Load primary weights; fall back if missing. Returns (model, info)."""
    for path, tag in ((primary, "primary"), (fallback, "fallback")):
        if path.exists():
            ck = torch.load(path, map_location=device)
            cfg = ck.get("model_config", {})
            model = FallGRUClassifier(
                input_size=cfg.get("input_size", 62),
                hidden_size=cfg.get("hidden_size", ck.get("hidden_size", 128)),
                num_layers=cfg.get("num_layers",  ck.get("num_layers", 2)),
                dropout=cfg.get("dropout", ck.get("dropout", 0.2)),
            ).to(device).eval()
            model.load_state_dict(ck["model_state_dict"])
            return model, {"path": str(path), "slot": tag}
    raise FileNotFoundError(f"No GRU weights at {primary} or {fallback}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional local YOLO fallback (when scene layer doesn't provide pose)
# ─────────────────────────────────────────────────────────────────────────────

class _LocalPose:
    def __init__(self, weights: Path, conf: float, iou: float, img_size: int):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.conf, self.iou, self.imgsz = conf, iou, img_size

    def infer(self, rgb_bgr: np.ndarray):
        """Return (kpts(17,2), confs(17,), bbox(4,)) or (None, None, None)."""
        res = self.model.predict(rgb_bgr, verbose=False,
                                 conf=self.conf, iou=self.iou, imgsz=self.imgsz, device="cpu")[0]
        if len(res.boxes) == 0:
            return None, None, None
        bbox = res.boxes[0].xyxy[0].cpu().numpy()
        kpts, confs = None, None
        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            kpts  = res.keypoints.xy[0].cpu().numpy()
            confs = (res.keypoints.conf[0].cpu().numpy()
                     if res.keypoints.conf is not None else np.ones(17, dtype=np.float32))
        return kpts, confs, bbox


# ─────────────────────────────────────────────────────────────────────────────
# Public module
# ─────────────────────────────────────────────────────────────────────────────

class FallModule:
    """
    Per-tick entry: process(bundle, roi) -> {"fall.classifier": Measurement}
    Always returns a Measurement (fall state is a free running value, not a
    gated spectral estimate).
    """

    def __init__(self, cfg: FallConfig, local_pose: bool = True):
        self.cfg = cfg
        dev = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        self.gru, self.gru_info = _load_gru(cfg.gru_weights_primary,
                                            cfg.gru_weights_fallback, self.device)

        self.kin  = _Kinematics(cfg.kin)
        self.feat = _Features62D()
        self.fsm  = _RefinedFSM(cfg.kin, cfg.fsm)
        self.gate = _HybridGate(cfg.ml)

        self._buf: Deque[np.ndarray] = deque(maxlen=cfg.ml.temporal_window)
        self._tick = 0

        self._local = _LocalPose(cfg.yolo_weights, cfg.yolo_conf, cfg.yolo_iou, cfg.yolo_img_size) \
                      if local_pose and cfg.yolo_weights.exists() else None

    # ---- lifecycle ----------------------------------------------------------

    def reset(self):
        self.kin.reset(); self.feat.reset(); self.fsm.reset()
        self.gate = _HybridGate(self.cfg.ml)
        self._buf.clear(); self._tick = 0

    # ---- per-tick ------------------------------------------------------------

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> Dict[str, Measurement]:
        t0 = time.perf_counter()
        self._tick += 1
        ts = bundle.ts_mono
        h, w = bundle.rgb.shape[:2]

        # 1. pose + bbox: scene layer first, local YOLO as fallback
        kpts, confs, bbox = None, None, None
        if roi is not None:
            kpts  = getattr(roi, "pose_keypoints", None)
            confs = getattr(roi, "pose_keypoints_conf", None)
            bbox  = getattr(roi, "person_bbox", None)
        if bbox is None and self._local is not None:
            kpts, confs, bbox = self._local.infer(bundle.rgb)

        # 2. kinematics + 62-D features
        kin_data = self.kin.analyze(kpts, confs, bbox, (h, w), ts)
        feat     = self.feat.extract((h, w), kin_data)
        self._buf.append(feat)

        # 3. FSM step
        fsm_event, is_early = self.fsm.step(kin_data, ts)
        if fsm_event is not None:
            self.gate.register_fsm_event({"start": fsm_event["start"] or ts,
                                          "end":   fsm_event["end"],
                                          "time":  fsm_event["end"]})
        if is_early:
            self.gate.register_early_fall(ts)

        # 4. GRU inference every stride frames when buffer full
        ml_prob: Optional[float] = None
        if len(self._buf) == self.cfg.ml.temporal_window and (self._tick % self.cfg.ml.stride == 0):
            x = torch.from_numpy(np.stack(self._buf, axis=0)[None]).float().to(self.device)
            with torch.no_grad():
                ml_prob = float(torch.sigmoid(self.gru(x)).cpu().item())
            self.gate.register_ml_prob(ml_prob, ts)

        # 5. Strategy H5 gate
        confirmed, rejected = self.gate.check(ts, self.fsm.state)

        has_person = bool(kin_data["has_person"])
        kp_quality = 1.0 if (confs is not None and float(np.isfinite(confs).all()) and confs.mean() > 0.3) else 0.0
        sqi = 1.0 if (has_person and kp_quality) else (0.5 if has_person else 0.0)

        m = Measurement(
            value=float(self.fsm.state),
            sqi=sqi,
            ts=time.time(),
            source="fall.classifier",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={
                "patient_state": STATE_NAME[self.fsm.state],
                "has_person": has_person,
                "ml_probability": ml_prob,
                "raw_kinematic_score": kin_data.get("raw_score"),
                "ar": kin_data.get("ar"),
                "angle_deg": kin_data.get("angle_deg"),
                "v_down": kin_data.get("v_down"),
                "bottom_y": kin_data.get("bottom_y"),
                "provisional_active": bool(self.gate.provisional),
                "confirmed_fall": confirmed,
                "rejected_alarm": rejected,
                "gru_weights": self.gru_info,
            },
        )
        return {"fall.classifier": m}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_fall_module(signal_yaml_path: str = "config/signal.yaml",
                     local_pose: bool = True) -> FallModule:
    """Build a FallModule from config/signal.yaml. Missing fall block => defaults."""
    import yaml
    cfg = {}
    try:
        with open(signal_yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    return FallModule(FallConfig.from_dict(cfg), local_pose=local_pose)