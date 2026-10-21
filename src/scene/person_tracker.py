"""Single-patient selection for the VisionICU scene layer.

WHAT THIS IS
------------
pose.py returns every person in the frame. This module decides which one is
*the patient* - exactly one per camera view - and holds that choice stably
over time so the clinical modules (RR, HR, Fall, TAA) all read the same body.

THE ACQUIRE / MAINTAIN ASYMMETRY (the whole idea)
-------------------------------------------------
Acquiring a new patient is STRICT: the candidate must be in the chair/bed
(chair.fraction_inside above a floor). That is why calibration runs first -
it prevents the system from locking onto a nurse who is in frame when
monitoring starts.

Maintaining a locked patient is LOOSE: once locked we follow that blob by
nearest-centroid, ignoring chair membership. A patient who slides out of the
chair (a fall) has low chair-fraction exactly when we must keep tracking
them, so chair membership cannot gate maintenance. Staff cannot steal the
lock while the patient blob persists near its last position, because a
nurse's centroid is a different, farther point.

If the patient blob disappears (full occlusion) for longer than
`max_lost_frames`, the lock drops and re-acquisition uses the strict chair
rule again.

WHAT IT FILLS IN ROIBundle
--------------------------
pose_keypoints (17,2), pose_keypoints_conf (17,), person_bbox [x1,y1,x2,y2],
in_chair_frac (chosen patient), h_centroid (normalised vertical centroid,
0=top .. 1=bottom), staff_count (other people in view). roi_bundle.py reads
the PatientTrack this module returns and assembles the full ROIBundle.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .chair_calibration import ChairCalibration
from .pose import PosePerson

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.scene.tracker")

# COCO torso keypoints used for the vertical centroid.
_TORSO = (5, 6, 11, 12)  # shoulders + hips


@dataclass
class TrackerConfig:
    conf_min: float = 0.30            # keypoint conf floor for chair/centroid
    acquire_chair_min: float = 0.30   # min chair-fraction to LOCK a new patient
    acquire_accept: float = 0.30      # min acquire score to accept
    w_chair: float = 0.75             # acquire score: chair weight
    w_size: float = 0.25              # acquire score: relative-size weight
    match_gate_frac: float = 0.20     # max centroid jump (fraction of diag) to keep lock
    maintain_box_min: float = 0.15    # min box score to keep a matched patient
    max_lost_frames: int = 15         # frames the lock survives with no match (~0.5 s @30fps)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TrackerConfig":
        c = cls()
        if not d:
            return c
        for k in c.__dataclass_fields__:  # type: ignore[attr-defined]
            if k in d and d[k] is not None:
                setattr(c, k, type(getattr(c, k))(d[k]))
        return c


@dataclass
class PatientTrack:
    """The patient selection for one frame."""

    person: Optional[PosePerson]   # None when no patient this frame
    in_chair_frac: float
    h_centroid: float              # normalised vertical centroid 0..1 (0=top)
    staff_count: int
    score: float
    is_locked: bool
    ts: float


class PersonTracker:
    def __init__(
        self,
        chair: ChairCalibration,
        image_size: Optional[tuple[int, int]] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.chair = chair
        self.cfg = config or TrackerConfig()
        # (h, w); prefer explicit, else the chair's, else filled on first update
        self.image_size = image_size or (
            chair.image_size if chair.image_size != (0, 0) else None
        )
        self._locked = False
        self._last_centroid: Optional[tuple[float, float]] = None
        self._frames_since_seen = 0

    # -- helpers ----------------------------------------------------------- #
    def _diag(self) -> float:
        if not self.image_size:
            return 1.0
        h, w = self.image_size
        return math.hypot(float(h), float(w)) or 1.0

    def _rel_sizes(self, people: list[PosePerson]) -> list[float]:
        areas = [p.bbox_area for p in people]
        mx = max(areas) if areas else 0.0
        return [(a / mx) if mx > 0 else 0.0 for a in areas]

    def _h_centroid(self, p: PosePerson) -> float:
        if not self.image_size:
            return 0.0
        h = float(self.image_size[0]) or 1.0
        ys = [
            float(p.keypoints[i][1])
            for i in _TORSO
            if p.keypoints_conf[i] >= self.cfg.conf_min and p.keypoints[i].sum() > 0
        ]
        y = (sum(ys) / len(ys)) if ys else p.bbox_center[1]
        return float(np.clip(y / h, 0.0, 1.0))

    def _nearest_to_lock(
        self, people: list[PosePerson]
    ) -> tuple[Optional[PosePerson], float]:
        if self._last_centroid is None:
            return None, math.inf
        lx, ly = self._last_centroid
        best, best_d = None, math.inf
        for p in people:
            cx, cy = p.bbox_center
            d = math.hypot(cx - lx, cy - ly)
            if d < best_d:
                best, best_d = p, d
        return best, best_d

    # -- main -------------------------------------------------------------- #
    def update(
        self,
        people: list[PosePerson],
        ts: float,
        image_size: Optional[tuple[int, int]] = None,
    ) -> PatientTrack:
        if image_size is not None:
            self.image_size = image_size
        if self.image_size is None and people:
            # last-resort inference from the widest bbox seen
            mx = max(float(p.bbox[2]) for p in people)
            my = max(float(p.bbox[3]) for p in people)
            self.image_size = (int(my), int(mx))

        selected: Optional[PosePerson] = None
        sel_score = 0.0

        # ---- MAINTAIN: follow the locked blob by proximity ---------------- #
        if self._locked:
            match, d = self._nearest_to_lock(people)
            gate = self.cfg.match_gate_frac * self._diag()
            if (
                match is not None
                and d <= gate
                and match.score >= self.cfg.maintain_box_min
            ):
                selected, sel_score = match, match.score
                self._last_centroid = match.bbox_center
                self._frames_since_seen = 0
            else:
                self._frames_since_seen += 1
                if self._frames_since_seen > self.cfg.max_lost_frames:
                    self._locked = False
                    self._last_centroid = None
                    log.info("Patient lock lost after %d frames.", self._frames_since_seen)

        # ---- ACQUIRE: strict, chair-gated ------------------------------- #
        if not self._locked and people:
            rel = self._rel_sizes(people)
            best, best_score, best_frac = None, -1.0, 0.0
            for p, size in zip(people, rel):
                frac = self.chair.fraction_inside(
                    p.keypoints, p.keypoints_conf, self.cfg.conf_min
                )
                s = self.cfg.w_chair * frac + self.cfg.w_size * size
                if s > best_score:
                    best, best_score, best_frac = p, s, frac
            if (
                best is not None
                and best_frac >= self.cfg.acquire_chair_min
                and best_score >= self.cfg.acquire_accept
            ):
                selected, sel_score = best, best_score
                self._locked = True
                self._last_centroid = best.bbox_center
                self._frames_since_seen = 0
                log.info("Patient acquired (chair_frac=%.2f score=%.2f).", best_frac, best_score)

        # ---- build result ------------------------------------------------ #
        if selected is not None:
            frac = self.chair.fraction_inside(
                selected.keypoints, selected.keypoints_conf, self.cfg.conf_min
            )
            return PatientTrack(
                person=selected,
                in_chair_frac=frac,
                h_centroid=self._h_centroid(selected),
                staff_count=max(0, len(people) - 1),
                score=sel_score,
                is_locked=self._locked,
                ts=ts,
            )
        return PatientTrack(
            person=None,
            in_chair_frac=0.0,
            h_centroid=0.0,
            staff_count=len(people),
            score=0.0,
            is_locked=self._locked,
            ts=ts,
        )

    # -- housekeeping ------------------------------------------------------ #
    def reset(self) -> None:
        """Drop the current lock (e.g. operator re-calibrated)."""
        self._locked = False
        self._last_centroid = None
        self._frames_since_seen = 0


def load_person_tracker(
    chair: ChairCalibration,
    config_path: str | Path = "config/signal.yaml",
    image_size: Optional[tuple[int, int]] = None,
) -> PersonTracker:
    """Build a PersonTracker, reading an optional `scene.tracker` block.

    Falls back to TrackerConfig defaults when the file, PyYAML, or the block
    is absent - so it is always safe to call.
    """
    tcfg = None
    path = Path(config_path)
    if yaml is not None and path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
            tcfg = TrackerConfig.from_dict((raw.get("scene") or {}).get("tracker"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read tracker config from %s (%s); using defaults.", path, exc)
    return PersonTracker(chair, image_size=image_size, config=tcfg)