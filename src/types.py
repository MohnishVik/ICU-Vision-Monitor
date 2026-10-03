"""
types.py — shared data contracts for VisionICU
All modules import from here. Never import from a module back into types.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Measurement:
    """
    The only output type every module is allowed to return.
    value=None is legal and means "I cannot measure right now."
    Silently guessing a wrong value is the only real failure mode.
    """
    value: Optional[float]      # None = unavailable; never imputed as normal
    sqi: float                  # Signal Quality Index 0.0–1.0
    ts: float                   # Unix timestamp of window end
    source: str                 # e.g. "rr.depth", "hr.efficientphys", "fall.geometry"
    latency_ms: float           # wall-clock processing time for this estimate
    meta: dict = field(default_factory=dict)  # module-specific extras


@dataclass
class FrameBundle:
    """
    One synchronised tick of all camera streams.
    thermal may be None if ESP32 is offline — all consumers must handle this.
    """
    rgb: np.ndarray             # (H=480, W=640, 3) uint8
    depth: np.ndarray           # (H=480, W=640)   uint16, aligned to RGB space
    ir: Optional[np.ndarray]    # (H=480, W=848)   uint8,  IR left stream
    ts_mono: float
    thermal_ts: Optional[float] = None   # when the thermal frame was received (lets the RR thermal path skip repeats)


@dataclass
class ROIBundle:
    """
    Scene-layer output: crops and geometric features consumed by all 4 modules.
    None fields mean that ROI could not be extracted this tick (occlusion, out-of-frame).
    """
    # Visual crops (RGB unless noted)
    face_roi: Optional[np.ndarray]        # (H, W, 3) uint8
    chest_roi: Optional[np.ndarray]       # (H, W, 3) uint8
    chest_depth: Optional[np.ndarray]     # (H, W)    uint16, same crop coords as chest_roi
    abdomen_roi: Optional[np.ndarray]     # (H, W, 3) uint8
    abdomen_depth: Optional[np.ndarray]   # (H, W)    uint16
    limb_rois: list                        # list of np.ndarray crops

    # Geometric features (from plane_ransac + chair OBB)
    h_centroid: float           # height of body centroid above floor in metres
    in_chair_frac: float        # 0.0–1.0 fraction of person 3D cloud inside chair OBB

    # Scene context
    staff_count: int            # people detected in zone besides the patient
    lighting: str               # "bright" | "dim" | "dark"
    patient_id: str
    ts: float                   # Unix timestamp
