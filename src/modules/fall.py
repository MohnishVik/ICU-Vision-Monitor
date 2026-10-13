"""
Fall detection adapter for ICU-Vision-Monitor.

Wraps the existing RealtimeHybridDetector and exposes the repository's
Measurement interface. The detector's temporal state is preserved between
frames; reset only at a new monitoring session.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from src.types import Measurement
except ImportError:
    from types import SimpleNamespace as _Unused  # fallback handled below
    try:
        from ..types import Measurement
    except ImportError as exc:
        raise ImportError(
            "Cannot import Measurement from the repository's src.types module."
        ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class FallModule:
    """Adapter exposing frame-by-frame fall detection as Measurements."""

    STATE_CODES = {
        "NORMAL": 0,
        "SITTING": 1,
        "FALLING": 2,
        "LYING_DOWN": 3,
    }

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.detector: Any = None
        self._frame_number = 0
        self._session_id = "default_session"

    def start_session(self, session_id: str) -> None:
        """Start a new session and clear the detector's temporal history."""
        detector = self._get_detector()
        detector.reset(session_id=session_id)
        self._session_id = session_id
        self._frame_number = 0

    def _get_detector(self) -> Any:
        """Load the existing detector only when first needed."""
        if self.detector is None:
            try:
                module = importlib.import_module("hybrid_detector")
                detector_class = module.RealtimeHybridDetector
                self.detector = detector_class(config_path=self.config_path)
            except Exception as exc:
                raise RuntimeError(
                    "Could not initialize RealtimeHybridDetector. Check that "
                    "scripts/model.py, the GRU checkpoint, YOLO pose weights, "
                    "configuration files, and Python dependencies are available."
                ) from exc
        return self.detector

    def process_frame(
        self,
        frame: np.ndarray,
        ts: float,
        patient_id: Optional[str] = None,
    ) -> Measurement:
        """
        Process one RGB frame.

        Args:
            frame: RGB image as a NumPy array.
            ts: Frame timestamp in seconds.
            patient_id: Optional patient identifier for output metadata.

        Returns:
            Measurement containing the current state and detection metadata.
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        detector = self._get_detector()
        self._frame_number += 1

        result = detector.process_frame(
            frame=frame,
            timestamp_s=float(ts),
            frame_number=self._frame_number,
        )

        state = str(result.get("fsm_state") or "NORMAL").upper()
        confirmed_event = result.get("confirmed_fall")
        if confirmed_event:
            state = "LYING_DOWN"

        state_code = self.STATE_CODES.get(state)
        has_person = bool(result.get("has_person", False))
        raw_score = result.get("raw_kinematic_score")
        ml_probability = result.get("ml_probability")

        metadata = {
            "patient_state": state,
            "patient_id": patient_id,
            "session_id": result.get("session_id", self._session_id),
            "frame_number": result.get("frame_number", self._frame_number),
            "has_person": has_person,
            "raw_kinematic_score": raw_score,
            "ml_probability": ml_probability,
            "provisional_active": bool(result.get("provisional_active", False)),
            "confirmed_fall": confirmed_event,
            "rejected_alarm": result.get("rejected_alarm"),
            "track_id": result.get("track_id"),
            "box": result.get("box"),
        }

        # SQI here reflects person/keypoint availability, not a clinical measure.
        keypoints = result.get("keypoints")
        keypoint_quality = 0.0
        if keypoints is not None:
            try:
                keypoint_array = np.asarray(keypoints, dtype=float)
                if keypoint_array.size:
                    keypoint_quality = float(
                        np.isfinite(keypoint_array).all()
                    )
            except (TypeError, ValueError):
                keypoint_quality = 0.0

        sqi = 1.0 if has_person and keypoint_quality else (0.5 if has_person else 0.0)

        return Measurement(
            value=state_code,
            sqi=sqi,
            ts=float(ts),
            source="fall.classifier",
            latency_ms=0.0,
            meta=metadata,
        )

    def reset(self, session_id: Optional[str] = None) -> None:
        """Reset temporal state at a session boundary, not between frames."""
        detector = self._get_detector()
        self._session_id = session_id or self._session_id
        detector.reset(session_id=self._session_id)
        self._frame_number = 0
