"""Assemble the ROIBundle each clinical module reads, from one patient track.

WHAT THIS IS
------------
person_tracker.py decides *who* the patient is. This module turns that
patient's skeleton into the anatomical windows each vital sign needs, and
packs them into the ROIBundle contract (src/types.py):

    face_roi      BGR colour crop of the face     -> HR (POS skin signal)
    chest_roi     BGR colour crop of the chest    -> (reserved)
    chest_depth   RAW depth crop of the chest     -> RR (PSD) + TAA (phase)
    abdomen_roi   BGR colour crop of the abdomen  -> (reserved)
    abdomen_depth RAW depth crop of the abdomen   -> TAA (phase vs chest)
    pose_*        the patient's keypoints/bbox    -> Fall
    in_chair_frac, h_centroid, staff_count        -> scoring / EWS context
    lighting, patient_id, ts

DEPTH UNITS - IMPORTANT
-----------------------
Depth crops are handed through RAW (uint16, as the camera delivers them).
The modules were built around raw units: TAA's amplitude gate is
`min_amplitude_units` in uint16 steps and it filters validity with
`min_valid_depth_units` / `min_valid_pixels`; RR takes the mean of the raw
crop and works in the frequency domain (scale-invariant). Converting to
metres here would silently break TAA's threshold, so we never convert.

COLOUR ORDER
------------
face_roi / chest_roi / abdomen_roi are cut straight from bundle.rgb, which is
BGR (bgr8 from the D435). HR expects BGR and swaps channels itself.

ANATOMY FROM COCO-17 KEYPOINTS
------------------------------
Shoulders (5, 6) and hips (11, 12) frame the torso. Chest = its upper half,
abdomen = its lower half, face = the head keypoints (0-4). When those points
are missing or low-confidence we fall back to fixed fractions of the person's
bounding box, so a module always receives a usable (if coarser) window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..types import FrameBundle, ROIBundle
from .person_tracker import PatientTrack

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.scene.roi")

# COCO indices
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SH, _R_SH, _L_HIP, _R_HIP = 5, 6, 11, 12
_FACE_PTS = (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)


@dataclass
class ROIGeometryConfig:
    conf_min: float = 0.30
    # padding as a fraction of the region's own width/height
    face_pad: float = 0.45
    chest_x_pad: float = 0.15
    abd_x_pad: float = 0.10
    # bbox-fraction fallbacks (top, bottom) along the person's height
    face_frac: tuple[float, float] = (0.00, 0.22)
    chest_frac: tuple[float, float] = (0.22, 0.52)
    abd_frac: tuple[float, float] = (0.52, 0.80)
    bbox_x_keep: float = 0.60       # central width fraction for bbox fallbacks
    # lighting thresholds on mean grey (0-255)
    dark_below: float = 40.0
    dim_below: float = 90.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ROIGeometryConfig":
        c = cls()
        if not d:
            return c
        for k in c.__dataclass_fields__:  # type: ignore[attr-defined]
            if k in d and d[k] is not None:
                cur = getattr(c, k)
                setattr(c, k, tuple(d[k]) if isinstance(cur, tuple) else type(cur)(d[k]))
        return c


def _clamp_box(x1: float, y1: float, x2: float, y2: float,
               w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _confident(kpts: np.ndarray, confs: np.ndarray, idxs, conf_min: float):
    pts = []
    for i in idxs:
        if confs[i] >= conf_min and kpts[i].sum() > 0:
            pts.append((float(kpts[i][0]), float(kpts[i][1])))
    return pts


class ROIBuilder:
    def __init__(
        self,
        patient_id: str = "patient",
        config: Optional[ROIGeometryConfig] = None,
    ) -> None:
        self.patient_id = patient_id
        self.cfg = config or ROIGeometryConfig()

    # -- region geometry --------------------------------------------------- #
    def _face_box(self, kpts, confs, bbox, w, h):
        pts = _confident(kpts, confs, _FACE_PTS, self.cfg.conf_min)
        if len(pts) >= 2:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = sum(ys) / len(ys)
            # A face is ~as tall as wide; eye/nose/ear points span width, not
            # height, so size the box from the horizontal spread and extend it
            # vertically - biased upward to include the forehead.
            width = max(30.0, max(xs) - min(xs))
            half_w = width / 2.0 * (1.0 + self.cfg.face_pad)
            half_h = half_w * 1.3
            return _clamp_box(cx - half_w, cy - half_h * 1.1,
                              cx + half_w, cy + half_h * 0.9, w, h)
        return self._bbox_frac_box(bbox, self.cfg.face_frac, w, h)

    def _torso_box(self, kpts, confs, bbox, top_pts, bot_pts, x_pad, frac, w, h):
        sh = _confident(kpts, confs, top_pts, self.cfg.conf_min)
        hp = _confident(kpts, confs, bot_pts, self.cfg.conf_min)
        if len(sh) >= 1 and len(hp) >= 1:
            sh_y = np.mean([p[1] for p in sh]); hp_y = np.mean([p[1] for p in hp])
            xs = [p[0] for p in sh + hp]
            x1, x2 = min(xs), max(xs)
            pad = max(8.0, (x2 - x1) * x_pad)
            return _clamp_box(x1 - pad, min(sh_y, hp_y), x2 + pad, max(sh_y, hp_y), w, h)
        return self._bbox_frac_box(bbox, frac, w, h)

    def _bbox_frac_box(self, bbox, frac, w, h):
        if bbox is None:
            return None
        bx1, by1, bx2, by2 = (float(v) for v in bbox[:4])
        bw, bh = bx2 - bx1, by2 - by1
        keep = self.cfg.bbox_x_keep
        cx = (bx1 + bx2) / 2.0
        x1 = cx - bw * keep / 2.0; x2 = cx + bw * keep / 2.0
        y1 = by1 + bh * frac[0]; y2 = by1 + bh * frac[1]
        return _clamp_box(x1, y1, x2, y2, w, h)

    @staticmethod
    def _crop(img: Optional[np.ndarray], box, tw: int, th: int):
        """Crop `img` at `box` (given in RGB pixel coords), scaling the box to
        the image's own size first - depth may differ in resolution if not
        perfectly aligned."""
        if img is None or box is None:
            return None
        ih, iw = img.shape[:2]
        sx, sy = iw / float(tw), ih / float(th)
        x1, y1, x2, y2 = box
        X1, Y1 = int(x1 * sx), int(y1 * sy)
        X2, Y2 = int(x2 * sx), int(y2 * sy)
        if X2 - X1 < 1 or Y2 - Y1 < 1:
            return None
        crop = img[Y1:Y2, X1:X2]
        return crop if crop.size > 0 else None

    def _lighting(self, rgb: np.ndarray) -> str:
        if cv2 is not None:
            grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            m = float(grey.mean())
        else:
            m = float(rgb.mean())
        if m < self.cfg.dark_below:
            return "dark"
        if m < self.cfg.dim_below:
            return "dim"
        return "normal"

    # -- main -------------------------------------------------------------- #
    def build(self, bundle: FrameBundle, track: PatientTrack) -> ROIBundle:
        rgb = bundle.rgb
        depth = bundle.depth
        h, w = rgb.shape[:2]
        lighting = self._lighting(rgb)

        if track.person is None:
            return ROIBundle(
                face_roi=None, chest_roi=None, chest_depth=None,
                abdomen_roi=None, abdomen_depth=None, limb_rois=[],
                h_centroid=track.h_centroid, in_chair_frac=track.in_chair_frac,
                staff_count=track.staff_count, lighting=lighting,
                patient_id=self.patient_id, ts=track.ts,
                pose_keypoints=None, pose_keypoints_conf=None, person_bbox=None,
            )

        p = track.person
        kpts, confs, bbox = p.keypoints, p.keypoints_conf, p.bbox

        face_box = self._face_box(kpts, confs, bbox, w, h)
        chest_box = self._torso_box(
            kpts, confs, bbox, (_L_SH, _R_SH),
            (_L_HIP, _R_HIP), self.cfg.chest_x_pad, self.cfg.chest_frac, w, h)
        # abdomen: between the chest's lower edge and the hips
        abd_box = self._abdomen_box(kpts, confs, bbox, chest_box, w, h)

        return ROIBundle(
            face_roi=self._crop(rgb, face_box, w, h),
            chest_roi=self._crop(rgb, chest_box, w, h),
            chest_depth=self._crop(depth, chest_box, w, h),      # RAW
            abdomen_roi=self._crop(rgb, abd_box, w, h),
            abdomen_depth=self._crop(depth, abd_box, w, h),      # RAW
            limb_rois=[],
            h_centroid=track.h_centroid,
            in_chair_frac=track.in_chair_frac,
            staff_count=track.staff_count,
            lighting=lighting,
            patient_id=self.patient_id,
            ts=track.ts,
            pose_keypoints=kpts,
            pose_keypoints_conf=confs,
            person_bbox=bbox,
        )

    def _abdomen_box(self, kpts, confs, bbox, chest_box, w, h):
        sh = _confident(kpts, confs, (_L_SH, _R_SH), self.cfg.conf_min)
        hp = _confident(kpts, confs, (_L_HIP, _R_HIP), self.cfg.conf_min)
        if len(sh) >= 1 and len(hp) >= 1:
            mid_y = (np.mean([p[1] for p in sh]) + np.mean([p[1] for p in hp])) / 2.0
            hp_y = np.mean([p[1] for p in hp])
            xs = [p[0] for p in hp]
            x1, x2 = min(xs), max(xs)
            pad = max(8.0, (x2 - x1) * self.cfg.abd_x_pad)
            # extend a little below the hips to capture belly rise
            span = max(10.0, hp_y - mid_y)
            return _clamp_box(x1 - pad, mid_y, x2 + pad, hp_y + 0.3 * span, w, h)
        return self._bbox_frac_box(bbox, self.cfg.abd_frac, w, h)


def load_roi_builder(
    config_path: str | Path = "config/signal.yaml",
    patient_id: str = "patient",
) -> ROIBuilder:
    """Build an ROIBuilder, reading an optional `scene.roi` block.

    Falls back to geometry defaults when the file/PyYAML/block is absent.
    """
    cfg = None
    path = Path(config_path)
    if yaml is not None and path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
            cfg = ROIGeometryConfig.from_dict((raw.get("scene") or {}).get("roi"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read ROI config from %s (%s); using defaults.", path, exc)
    return ROIBuilder(patient_id=patient_id, config=cfg)