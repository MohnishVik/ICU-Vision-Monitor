"""Scene layer: one-time geometric grounding of the camera view.

The scene layer answers *where things are* so the clinical modules can
reason about *what is happening*. Its first component is the chair/bed
calibration, an operator-driven, one-time-per-session click that pins the
patient's furniture into image coordinates.
"""
from __future__ import annotations

from .chair_calibration import ChairCalibration, load_chair_calibration
from .pose import (
    COCO_KEYPOINTS,
    PoseEstimator,
    PosePerson,
    load_pose_estimator,
)
from .person_tracker import (
    PatientTrack,
    PersonTracker,
    TrackerConfig,
    load_person_tracker,
)
from .roi_bundle import (
    ROIBuilder,
    ROIGeometryConfig,
    load_roi_builder,
)
from .privacy import (
    PrivacyGuard,
    PrivacyPolicy,
    load_privacy_guard,
)

__all__ = [
    "ChairCalibration",
    "load_chair_calibration",
    "PoseEstimator",
    "PosePerson",
    "load_pose_estimator",
    "COCO_KEYPOINTS",
    "PersonTracker",
    "PatientTrack",
    "TrackerConfig",
    "load_person_tracker",
    "ROIBuilder",
    "ROIGeometryConfig",
    "load_roi_builder",
    "PrivacyGuard",
    "PrivacyPolicy",
    "load_privacy_guard",
]