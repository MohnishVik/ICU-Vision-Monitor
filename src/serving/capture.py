"""
capture.py — one row of training data per camera frame, for sessions started with capture on.

Each row holds the exact numbers the learned models are fed live, so a network fine-tuned on
these rows sees the same input it will see at the bedside:

  chest_depth        plain mean of valid chest depth  (learned D435 CNN, src/modules/learned.py)
  chest_depth_body   body-band mean of chest depth     (rr.depth, src/modules/rr.py)
  abdomen_depth      plain mean of valid abdomen depth (TAA)
  green              whole-frame mean green            (learned LBRD-IC CNN)
  keypoints + box    the tracked patient's pose, in image pixels (fall GRU, replayed offline)

No pixels are stored. A value the frame cannot provide (no patient, empty crop) is left empty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.modules.rr import _mean_from_depth_roi

_MIN_VALID, _MIN_PIXELS, _BODY_BAND = 5, 30, 150.0


def capture_row(bundle, roi, person, t_wall: float) -> dict:
    """Build a FRAME_COLUMNS dict from one frame, its ROIBundle and the tracked person."""
    h, w = bundle.rgb.shape[:2]
    chest = getattr(roi, "chest_depth", None) if roi is not None else None
    abd = getattr(roi, "abdomen_depth", None) if roi is not None else None
    row = {
        "t": t_wall, "t_mono": bundle.ts_mono, "img_h": h, "img_w": w,
        "chest_depth": _depth(chest, None),
        "chest_depth_body": _depth(chest, _BODY_BAND),
        "abdomen_depth": _depth(abd, None),
        "green": float(np.mean(bundle.rgb[:, :, 1])),
        "has_person": person is not None,
    }
    if person is not None:
        add_pose(row, person.keypoints, person.keypoints_conf, person.bbox)
    return row


def add_pose(row: dict, kpts, confs, bbox) -> None:
    """Write 17 keypoints (x, y, conf) and the person box into a capture row."""
    k = np.asarray(kpts, dtype=np.float64).reshape(17, 2)
    c = np.asarray(confs if confs is not None else np.ones(17), dtype=np.float64).reshape(17)
    for i in range(17):
        row[f"kp{i}_x"], row[f"kp{i}_y"], row[f"kp{i}_c"] = float(k[i, 0]), float(k[i, 1]), float(c[i])
    if bbox is not None:
        row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"] = (float(v) for v in bbox[:4])


def _depth(crop: Optional[np.ndarray], band: Optional[float]) -> Optional[float]:
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return _mean_from_depth_roi(crop, _MIN_VALID, _MIN_PIXELS, band)
