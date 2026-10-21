"""
state.py — per-patient vitals pipeline + JSON-safe snapshots for the API.

VitalsPipeline is the per-patient "brain" between the modules and the UI:

    raw module Measurements (rr.depth, rr.thermal, hr.pos, fall.classifier, taa.depth)
        -> inverse-variance fusion (SQI gate + agreement gate)     src/fusion/inv_variance.py
        -> Kalman smoothing (slew-limited, TAA-aware)              src/fusion/kalman_filter.py
        -> trends (5/15/60/240 min)                               src/fusion/trend.py
        -> EWS sub-score + alarm state machine + explanation       src/fusion/alarm_fsm.py
        -> 1 Hz history ring buffer + a JSON-safe snapshot dict

Both the camera engine and the demo engine feed this same class, so the demo
exercises the real fusion/alarm code paths.

Contract the UI relies on: a vital that cannot be measured is null with a
reason, and a vital with no fresh value for > stale_s is flagged stale. Neither
is ever displayed as a live normal reading.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from src.fusion import (
    HR_KALMAN, RR_KALMAN, AlarmManager, Condition, EWSScorer, FusionConfig,
    StalenessWatchdog, TrendTracker, VitalKalman, explain, fuse,
)
from src.types import Measurement

FALL_STATE_NAME = {0: "NORMAL", 1: "FALLING", 2: "LYING_DOWN"}


# --------------------------------------------------------------------------- #
# JSON safety: numpy types, NaN/inf, nested dicts, dataclasses
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert anything a module might put in meta into strict-JSON-safe data.
    NaN/inf become null (Starlette's JSON encoder rejects them outright)."""
    if _depth > 8:
        return None
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item(), _depth + 1)
    if isinstance(obj, np.ndarray):
        if obj.size > 256:
            return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        return to_jsonable(obj.tolist(), _depth + 1)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, deque)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, Measurement):
        return measurement_dict(obj)
    if hasattr(obj, "to_dict"):
        try:
            return to_jsonable(obj.to_dict(), _depth + 1)
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def measurement_dict(m: Optional[Measurement]) -> Optional[dict]:
    if m is None:
        return None
    return {
        "value": to_jsonable(m.value), "sqi": to_jsonable(float(m.sqi)), "ts": to_jsonable(m.ts),
        "source": m.source, "latency_ms": to_jsonable(float(m.latency_ms)),
        "meta": to_jsonable(m.meta or {}),
    }


# --------------------------------------------------------------------------- #
# Scene context passed in each tick
# --------------------------------------------------------------------------- #
@dataclass
class SceneInfo:
    has_patient: bool = False
    in_chair_frac: float = 0.0
    staff_count: int = 0
    lighting: str = "unknown"
    h_centroid: float = 0.0
    thermal_online: bool = False
    calibrated: bool = False
    image_size: Optional[List[int]] = None          # [h, w]
    keypoints: Optional[list] = None                # 17 x [x, y] (patient)
    keypoints_conf: Optional[list] = None
    person_bbox: Optional[list] = None
    others: List[list] = field(default_factory=list)  # bboxes of non-patient people
    chair: Optional[list] = None                    # 4 x [x, y]
    redaction: str = "faces"                        # faces | unavailable | off
    nostril_box: Optional[list] = None              # [x0, y0, x1, y1] RGB px (breathing region)
    thermal_roi: Optional[list] = None              # [x0, y0, x1, y1] thermal px
    thermal_registered: bool = False
    thermal_quality: Optional[str] = None           # good | fair | poor | unchecked


def patient_state(scene: SceneInfo, fall_state: Optional[int]) -> str:
    if not scene.has_patient:
        return "UNKNOWN"
    if fall_state == 2 and scene.in_chair_frac < 0.2:
        return "ON_FLOOR"
    if scene.in_chair_frac >= 0.6:
        return "IN_CHAIR"
    if scene.in_chair_frac >= 0.2:
        return "SITTING_EDGE"
    return "OUT_OF_CHAIR"


# --------------------------------------------------------------------------- #
# The per-patient pipeline
# --------------------------------------------------------------------------- #
class VitalsPipeline:
    RR_SOURCES = ("rr.depth", "rr.thermal")
    HR_SOURCES = ("hr.pos",)

    def __init__(self, patient_id: str, label: str, rules: dict, signal_cfg: Optional[dict] = None,
                 mode: str = "camera", simulated: bool = False, stale_s: float = 5.0,
                 history_s: float = 4 * 3600.0, sample_period_s: float = 1.0):
        sig = signal_cfg or {}
        self.patient_id, self.label, self.mode, self.simulated = patient_id, label, mode, simulated
        self.rr_fuse = FusionConfig(
            sqi_min=float(((sig.get("rr") or {}).get("quality") or {}).get("peak_power_frac_min", 0.30)),
            agreement_gate_bpm=float((sig.get("rr") or {}).get("agreement_gate_bpm", 4.0)))
        self.hr_fuse = FusionConfig(
            sqi_min=float(((sig.get("hr") or {}).get("quality") or {}).get("peak_power_frac_min", 0.25)),
            agreement_gate_bpm=float((sig.get("hr") or {}).get("agreement_gate_bpm", 10.0)))
        # module sqi is already gated inside the modules; fusion's gate is a second, lower bar
        self.rr_fuse.sqi_min = min(self.rr_fuse.sqi_min, 0.30)
        self.hr_fuse.sqi_min = min(self.hr_fuse.sqi_min, 0.25)
        self.rr_kf, self.hr_kf = VitalKalman(RR_KALMAN), VitalKalman(HR_KALMAN)
        self.rr_trend, self.hr_trend = TrendTracker(), TrendTracker()
        self.watch = StalenessWatchdog(stale_s)
        self.scorer = EWSScorer(rules)
        self.alarms = AlarmManager(rules)
        self.stale_s = float(stale_s)
        self.sample_period_s = float(sample_period_s)
        self._hist: Deque[dict] = deque(maxlen=int(history_s / sample_period_s) + 10)
        self._last_sample = 0.0
        self._latest: Dict[str, Measurement] = {}
        self._latest_at: Dict[str, float] = {}
        self._rr_out: Optional[Measurement] = None
        self._hr_out: Optional[Measurement] = None
        self._rr_k: Optional[float] = None
        self._hr_k: Optional[float] = None
        self._last_fall_confirmed: Optional[float] = None
        self._lock = threading.Lock()
        self._alarm_lock = threading.Lock()
        self._snapshot: dict = {}
        self.version = 0
        self.started = time.time()
        self.modules: Dict[str, dict] = {}
        self.status, self.status_detail = "starting", ""
        self.metrics: Dict[str, Any] = {"fps": 0.0, "latency_ms": 0.0}
        self.scene = SceneInfo()

    # ------------------------------------------------------------------ step --
    def step(self, now: float, raw: Dict[str, Measurement], scene: SceneInfo) -> None:
        self.scene = scene
        for src, m in raw.items():
            self._latest[src], self._latest_at[src] = m, now

        rr_new = any(s in raw for s in self.RR_SOURCES)
        hr_new = any(s in raw for s in self.HR_SOURCES)
        taa_m = self._fresh("taa.depth", now)
        taa_flag = bool(taa_m is not None and taa_m.meta.get("taa_flag"))

        if rr_new:
            self._rr_out = fuse([self._fresh(s, now) for s in self.RR_SOURCES], "rr.fused", self.rr_fuse)
            self._rr_k = self.rr_kf.update(now, self._rr_out.value, self._rr_out.sqi, taa_active=taa_flag)
            self.rr_trend.add(now, self._rr_k)
            self.watch.observe(Measurement(self._rr_k, 1.0, now, "rr", 0.0) if self._rr_k is not None
                               else None, now)
        if hr_new:
            self._hr_out = fuse([self._fresh(s, now) for s in self.HR_SOURCES], "hr.fused", self.hr_fuse)
            self._hr_k = self.hr_kf.update(now, self._hr_out.value, self._hr_out.sqi)
            self.hr_trend.add(now, self._hr_k)
            self.watch.observe(Measurement(self._hr_k, 1.0, now, "hr", 0.0) if self._hr_k is not None
                               else None, now)

        rr_val = None if self.watch.is_stale("rr", now) else self._rr_k
        hr_val = None if self.watch.is_stale("hr", now) else self._hr_k
        ews = self.scorer.score({"rr": rr_val, "hr": hr_val})
        rr_stats = self.rr_trend.window(now, 3600)
        hr_stats = self.hr_trend.window(now, 3600)
        explanation = explain({"rr": rr_stats, "hr": hr_stats}, ews, current={"rr": rr_val, "hr": hr_val})

        fall_m = self._fresh("fall.classifier", now)
        fall_state = int(fall_m.value) if (fall_m is not None and fall_m.value is not None) else None
        if fall_m is not None and fall_m.meta.get("confirmed_fall"):
            self._last_fall_confirmed = now
        rr_m = self._latest.get("rr.depth")
        apnea = bool(rr_m is not None and rr_m.meta.get("apnea")
                     and (now - self._latest_at.get("rr.depth", 0)) <= self.stale_s)

        conditions: Dict[str, Optional[Condition]] = {
            "ews": (Condition(ews.level, f"Early-warning score {ews.total}", explanation)
                    if ews.level in ("amber", "red") else None),
            "fall": (Condition("red", "Fall detected", "Confirmed fall: posture change followed by lying "
                               "still. Check the patient now.")
                     if (fall_m is not None and fall_m.meta.get("confirmed_fall")) else None),
            "apnea": (Condition("red", "Possible apnoea", "Chest movement amplitude dropped below 25 % "
                                "of baseline for >= 10 s.") if apnea else None),
            "taa": (Condition("amber", "Paradoxical breathing", "Chest and abdomen moving out of phase "
                              "for >= 10 s (thoraco-abdominal asynchrony).") if taa_flag else None),
        }
        with self._alarm_lock:   # API thread may acknowledge concurrently
            open_alarms = self.alarms.update(now, conditions, staff_count=scene.staff_count)
            alarm_history = list(self.alarms.history[:10])
            actionable = self.alarms.actionable_this_shift(now)

        if now - self._last_sample >= self.sample_period_s:
            self._last_sample = now
            self._hist.append({
                "t": now, "rr": rr_val, "hr": hr_val,
                "rr_sqi": self._rr_out.sqi if self._rr_out is not None else None,
                "hr_sqi": self._hr_out.sqi if self._hr_out is not None else None,
                "ews": ews.total, "fall": fall_state, "taa": taa_flag,
            })

        snap = {
            "patient_id": self.patient_id, "label": self.label, "mode": self.mode,
            "simulated": self.simulated, "ts": now, "status": self.status,
            "status_detail": self.status_detail,
            "vitals": {
                "rr": self._vital_dict("rr", rr_val, self._rr_out, self.RR_SOURCES, now, ews,
                                       extra={"apnea": apnea}),
                "hr": self._vital_dict("hr", hr_val, self._hr_out, self.HR_SOURCES, now, ews),
            },
            "ews": {**ews.to_dict(), "explanation": explanation},
            "fall": {
                "state": fall_state, "state_name": FALL_STATE_NAME.get(fall_state, "UNKNOWN"),
                "patient_state": patient_state(scene, fall_state),
                "sqi": float(fall_m.sqi) if fall_m is not None else 0.0,
                "stale": fall_m is None,
                "ml_probability": to_jsonable(fall_m.meta.get("ml_probability")) if fall_m else None,
                "provisional": bool(fall_m.meta.get("provisional_active")) if fall_m else False,
                "last_confirmed": self._last_fall_confirmed,
            },
            "taa": {
                "flag": taa_flag,
                "phase_rad": to_jsonable(taa_m.value) if taa_m is not None else None,
                "sustain_s": to_jsonable(taa_m.meta.get("sustain_s")) if taa_m is not None else None,
                "sqi": float(taa_m.sqi) if taa_m is not None else 0.0,
                "stale": taa_m is None,
                "reason": to_jsonable(taa_m.meta.get("reason")) if taa_m is not None else None,
            },
            "pain": {"available": False,
                     "reason": "Pain module pending: src/modules/pain.py is a scaffold "
                               "(needs SynPAIN weights + AU extraction)."},
            "scene": to_jsonable(scene.__dict__),
            "alarms": [a.to_dict() for a in open_alarms],
            "alarm_history": [a.to_dict() for a in alarm_history],
            "trends": {
                "rr": {str(w): s.to_dict() for w, s in self.rr_trend.stats(now).items()},
                "hr": {str(w): s.to_dict() for w, s in self.hr_trend.stats(now).items()},
            },
            "modules": self.modules,
            "metrics": {**self.metrics, "uptime_s": now - self.started,
                        "actionable_alerts_shift": actionable},
        }
        with self._lock:
            self.version += 1
            snap["version"] = self.version
            self._snapshot = to_jsonable(snap)

    def backfill(self, ts: float, rr: Optional[float], hr: Optional[float], rr_sqi: float,
                 hr_sqi: float, fall_state: Optional[int], taa: bool) -> None:
        """Seed history + trends with past samples (demo mode only).
        Touches nothing that alarms or the live vitals depend on."""
        self.rr_trend.add(ts, rr)
        self.hr_trend.add(ts, hr)
        ews = self.scorer.score({"rr": rr, "hr": hr})
        self._hist.append({"t": ts, "rr": rr, "hr": hr, "rr_sqi": rr_sqi if rr is not None else None,
                           "hr_sqi": hr_sqi if hr is not None else None, "ews": ews.total,
                           "fall": fall_state, "taa": taa})
        self._last_sample = ts

    # --------------------------------------------------------------- helpers --
    def _fresh(self, src: str, now: float) -> Optional[Measurement]:
        m = self._latest.get(src)
        if m is None or (now - self._latest_at.get(src, 0.0)) > self.stale_s:
            return None
        return m

    def _vital_dict(self, key: str, value: Optional[float], fused: Optional[Measurement],
                    sources, now: float, ews, extra: Optional[dict] = None) -> dict:
        age = self.watch.age(key, now)
        return {
            "value": value,
            "raw": fused.value if fused is not None else None,
            "sqi": fused.sqi if fused is not None else 0.0,
            "stale": self.watch.is_stale(key, now),
            "age_s": age,
            "unit": "bpm",
            "reason": (fused.meta.get("reason") if fused is not None and fused.value is None
                       else (None if fused is not None else "warming_up")),
            "subscore": ews.subscores.get(key),
            "sources": {s: measurement_dict(self._fresh(s, now)) for s in sources},
            **(extra or {}),
        }

    # ---------------------------------------------------------------- reads --
    def snapshot(self) -> dict:
        with self._lock:
            if self._snapshot:
                return dict(self._snapshot)
        return to_jsonable({
            "patient_id": self.patient_id, "label": self.label, "mode": self.mode,
            "simulated": self.simulated, "ts": time.time(), "status": self.status,
            "status_detail": self.status_detail, "vitals": None, "ews": None, "fall": None,
            "taa": None, "pain": {"available": False}, "scene": self.scene.__dict__, "alarms": [],
            "alarm_history": [], "trends": None, "modules": self.modules,
            "metrics": self.metrics, "version": self.version,
        })

    def touch_status(self, status: str, detail: str = "") -> None:
        """Update status without a full step (camera offline, awaiting calibration...)."""
        self.status, self.status_detail = status, detail
        with self._lock:
            if self._snapshot:
                self._snapshot["status"], self._snapshot["status_detail"] = status, detail
                self._snapshot["modules"] = to_jsonable(self.modules)
                self._snapshot["scene"] = to_jsonable(self.scene.__dict__)
            self.version += 1
            if self._snapshot:
                self._snapshot["version"] = self.version

    def history(self, minutes: float = 60.0, max_points: int = 720) -> dict:
        cutoff = time.time() - minutes * 60.0
        pts = [p for p in list(self._hist) if p["t"] >= cutoff]
        if len(pts) > max_points:
            step = len(pts) / float(max_points)
            pts = [pts[int(i * step)] for i in range(max_points)]
        keys = ("t", "rr", "hr", "rr_sqi", "hr_sqi", "ews", "fall", "taa")
        return to_jsonable({"patient_id": self.patient_id, "minutes": minutes,
                            **{k: [p.get(k) for p in pts] for k in keys}})

    def acknowledge(self, alarm_id: Optional[int] = None) -> int:
        with self._alarm_lock:
            n = self.alarms.acknowledge(alarm_id, now=time.time())
            open_now = [a.to_dict() for a in self.alarms.open_alarms()]
            hist = [a.to_dict() for a in self.alarms.history[:10]]
        with self._lock:   # reflect the ack immediately, not on the next tick
            if self._snapshot:
                self._snapshot["alarms"] = to_jsonable(open_now)
                self._snapshot["alarm_history"] = to_jsonable(hist)
                self.version += 1
                self._snapshot["version"] = self.version
        return n
