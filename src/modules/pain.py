"""
pain.py — Pain Assessment module (EXPERIMENTAL)

Binary pain-expression classifier on the patient's face crop.

Model: MobileNetV3-Small (ImageNet init), one sigmoid output = P(pain expression),
trained on SynPAIN in notebooks/SynPain.ipynb and exported as
pain_synpain_mobilenetv3.pth. Expected at models/public/pain_mobilenetv3_synpain.pth
(override with signal.yaml -> pain.weights).

Training metadata (pain_synpain_meta.json): input 224, ImageNet mean/std, sigmoid,
decision_threshold 0.2962, val AUROC 0.688, test AUROC 0.816 (only 18 no-pain test
images), include_neutral_as_negative=True.

WHY "EXPERIMENTAL": the training run found 2,717 of SynPAIN's 10,710 images
(2,667 pain vs 50 no-pain, no neutral), so the classifier is strongly biased toward
"pain" (a blank image scores ~0.44, above the threshold). Its output is shown as a
probability for research use and never drives alarms. Retrain on the full dataset
before relying on it.

PER-PATIENT BASELINE. Measured on ordinary, neutral real faces the model returns
0.73-0.91, far above its 0.2962 threshold: it was trained on synthetic faces and its
absolute output does not transfer to real ones. (Earlier builds looked better only
because the camera ran underexposed; darkening the same faces pulls the output to
0.32-0.46. Underexposing on purpose would ruin heart rate, so that is not the fix.)
So the score is read RELATIVE to the patient's own calm face: for the first
baseline_s seconds of visible face (or after "Set calm baseline" on the bed page) the
module learns the patient's resting score and its spread; afterwards "pain_detected"
means the 5-s mean rose above baseline + max(min_rise, 3 x robust spread). The model's
fixed bias is in the baseline too, so it cancels. Still experimental, never alarms.

Preprocessing reproduces the notebook's eval transform exactly:
    BGR face crop -> RGB PIL -> Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize

Output (at most once per update_every_s):
    {"pain.mobilenetv3": Measurement(value = mean P(pain) over the last `window`
     scores | None, sqi, ts, source, latency_ms,
     meta = {probability_raw, threshold, pain_detected, face_px, window_n,
             experimental, model, trained_on, reason})}
value is None when no usable face is in view (reason: no_face | face_too_small).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.types import FrameBundle, Measurement, ROIBundle

SOURCE = "pain.mobilenetv3"


@dataclass
class PainConfig:
    weights: Path = Path("models/public/pain_mobilenetv3_synpain.pth")
    threshold: float = 0.2962          # F1-optimal on SynPAIN validation (training metadata)
    update_every_s: float = 1.0
    window: int = 5                    # scores averaged into `value`
    min_face_px: int = 40              # shorter side of the face crop
    device: str = "auto"
    baseline_s: float = 60.0           # seconds of visible face used to learn the calm baseline
    min_rise: float = 0.10             # smallest rise above baseline that counts
    forget_after_s: float = 120.0      # no face this long -> relearn the baseline (new patient?)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PainConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k) and v is not None:
                setattr(c, k, Path(v) if k == "weights" else type(getattr(c, k))(v))
        return c


class PainModule:
    def __init__(self, cfg: PainConfig):
        from torchvision import models, transforms   # imported here: torchvision is optional elsewhere

        self.cfg = cfg
        if not Path(cfg.weights).exists():
            raise FileNotFoundError(f"Pain weights not found: {cfg.weights}")
        dev = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        net = models.mobilenet_v3_small(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
        net.load_state_dict(torch.load(cfg.weights, map_location="cpu", weights_only=True))
        self.net = net.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._scores: Deque[float] = deque(maxlen=max(1, cfg.window))
        self._last_ts: Optional[float] = None
        self._last_face_ts: Optional[float] = None
        self.reset_baseline()

    def reset_baseline(self) -> None:
        """Start learning the patient's calm baseline again (patient is settled and comfortable)."""
        self._calib: list = []
        self.baseline: Optional[float] = None
        self.baseline_spread: Optional[float] = None

    def personal_threshold(self) -> Optional[float]:
        if self.baseline is None:
            return None
        return min(0.99, self.baseline + max(self.cfg.min_rise, 3.0 * (self.baseline_spread or 0.0)))

    def _learn(self, p: float) -> None:
        if self.baseline is not None:
            return
        self._calib.append(p)
        n_needed = max(5, int(self.cfg.baseline_s / max(self.cfg.update_every_s, 1e-3)))
        if len(self._calib) >= n_needed:
            a = np.asarray(self._calib)
            self.baseline = float(np.median(a))
            self.baseline_spread = float(1.4826 * np.median(np.abs(a - self.baseline)))

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> Dict[str, Measurement]:
        now = bundle.ts_mono
        if self._last_ts is not None and now - self._last_ts < self.cfg.update_every_s:
            return {}
        self._last_ts = now
        t0 = time.perf_counter()

        face = roi.face_roi if roi is not None else None
        if face is None or face.size == 0:
            return self._none("no_face", t0, 0)
        # No face for a long time usually means the patient left (or a new one arrived):
        # never judge a face against someone else's calm baseline.
        if self._last_face_ts is not None and now - self._last_face_ts > self.cfg.forget_after_s:
            self.reset_baseline()
        self._last_face_ts = now
        face_px = int(min(face.shape[:2]))
        if face_px < self.cfg.min_face_px:
            return self._none("face_too_small", t0, face_px)

        from PIL import Image
        x = self.tf(Image.fromarray(np.ascontiguousarray(face[:, :, ::-1]))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = float(torch.sigmoid(self.net(x)).item())
        self._scores.append(p)
        self._learn(p)
        value = float(np.mean(self._scores))
        # quality: how full the averaging window is x how large the face is (96 px = full credit)
        sqi = (len(self._scores) / self._scores.maxlen) * min(1.0, face_px / 96.0)
        thr = self.personal_threshold()
        n_needed = max(5, int(self.cfg.baseline_s / max(self.cfg.update_every_s, 1e-3)))
        return {SOURCE: Measurement(
            value=value, sqi=float(sqi), ts=time.time(), source=SOURCE,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={"probability_raw": p, "threshold": thr,
                  "pain_detected": bool(thr is not None and value >= thr), "face_px": face_px,
                  "baseline": self.baseline, "baseline_spread": self.baseline_spread,
                  "baseline_state": "ready" if self.baseline is not None else "learning",
                  "baseline_progress": 1.0 if self.baseline is not None else len(self._calib) / n_needed,
                  "rise": (value - self.baseline) if self.baseline is not None else None,
                  "model_threshold": self.cfg.threshold,
                  "window_n": len(self._scores), "experimental": True,
                  "model": "mobilenet_v3_small", "trained_on": "SynPAIN", "reason": None})}

    def _none(self, reason: str, t0: float, face_px: int) -> Dict[str, Measurement]:
        self._scores.clear()                       # never average across a gap in the face track
        return {SOURCE: Measurement(
            value=None, sqi=0.0, ts=time.time(), source=SOURCE,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={"probability_raw": None, "threshold": self.cfg.threshold, "pain_detected": False,
                  "face_px": face_px, "window_n": 0, "experimental": True,
                  "model": "mobilenet_v3_small", "trained_on": "SynPAIN", "reason": reason})}


def load_pain_module(signal_yaml_path: str = "config/signal.yaml") -> PainModule:
    """Build a PainModule from the optional `pain:` block of signal.yaml (defaults otherwise)."""
    import yaml
    cfg: dict = {}
    try:
        with open(signal_yaml_path) as f:
            cfg = (yaml.safe_load(f) or {}).get("pain") or {}
    except FileNotFoundError:
        pass
    return PainModule(PainConfig.from_dict(cfg))
