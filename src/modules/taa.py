"""
taa.py — Thoraco-Abdominal Asynchrony (TAA) detector

Why this module exists
  Healthy breathing: chest and abdomen rise and fall together — the diaphragm pushes
  belly out as the ribs expand, so the two surfaces are in phase.
  Impending respiratory distress: the diaphragm fatigues or the airway narrows and
  the two surfaces start moving out of step. The chest can rise while the belly
  draws in (paradoxical breathing). TAA is a clinically-recognised early warning
  that the spectral RR estimate alone will miss, because the rate itself can stay
  normal while the mechanics deteriorate.

Signal (per tick)
  1. Sample chest-ROI mean depth and abdomen-ROI mean depth from bundle.depth
  2. Rolling 30 s buffer at ~30 Hz for each surface
  3. Bandpass each buffer to the respiratory band (0.067–0.70 Hz)
  4. Instantaneous phase of each via Hilbert transform
  5. Wrapped phase difference, folded to [0, pi] (direction doesn't matter)
  6. SUSTAINED |phase_diff| > threshold over sustain_window_s raises the flag

Output
  Measurement(value=mean |phase_diff| in radians over the recent sub-window,
              sqi=signal quality from amplitude ratio,
              source="taa.depth",
              meta={ "taa_flag": bool, "sustain_s": float, "chest_fs_hz": float, ... })

Nothing here reads cameras or writes files.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import scipy.signal

from src.types import FrameBundle, Measurement, ROIBundle


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ROICfg:
    """Fallback fractional ROI boxes when the scene layer doesn't supply crops."""
    x: list = field(default_factory=lambda: [0.44, 0.65])
    y: list = field(default_factory=lambda: [0.54, 0.82])


@dataclass
class TAAConfig:
    enabled: bool                  = True
    window_s: float                = 30.0           # rolling buffer length
    update_every_s: float          = 1.0
    min_fill_s: float              = 20.0
    band_hz: Tuple[float, float]   = (0.067, 0.70)  # same band as rr.py depth path

    # Threshold on wrapped |phase diff| in radians.
    #   0        = perfectly in phase (normal)
    #   pi / 2   = 90 deg out of phase (chest leads/lags belly by a quarter cycle)
    #   pi       = fully anti-phase (paradoxical breathing)
    phase_diff_threshold_rad: float = 1.2
    sustain_window_s: float         = 10.0           # how long |phase| must stay high

    # Minimum breathing amplitude in the recent window. Below this, phase is noise
    # and we withhold the measurement rather than report a bogus TAA flag.
    min_amplitude_units: float      = 0.5           # depth-raw units (uint16 steps)
    min_valid_depth_units: int      = 5
    min_valid_pixels: int           = 30

    chest_fallback_roi:   _ROICfg   = field(default_factory=lambda: _ROICfg([0.44, 0.65], [0.54, 0.68]))
    abdomen_fallback_roi: _ROICfg   = field(default_factory=lambda: _ROICfg([0.42, 0.68], [0.70, 0.90]))

    @classmethod
    def from_dict(cls, cfg: dict) -> "TAAConfig":
        out = cls()
        rr   = (cfg or {}).get("rr", {})
        taa  = (cfg or {}).get("taa", {})
        ews  = taa.get("ews_rules", {})
        for k in ("enabled", "window_s", "update_every_s", "min_fill_s",
                  "phase_diff_threshold_rad", "sustain_window_s",
                  "min_amplitude_units", "min_valid_depth_units", "min_valid_pixels"):
            if k in taa:
                setattr(out, k, type(getattr(out, k))(taa[k]))
        if "band_hz" in taa:
            out.band_hz = tuple(taa["band_hz"])
        if "chest_fallback_roi" in taa:
            out.chest_fallback_roi = _ROICfg(**taa["chest_fallback_roi"])
        if "abdomen_fallback_roi" in taa:
            out.abdomen_fallback_roi = _ROICfg(**taa["abdomen_fallback_roi"])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Rolling signal buffer, one per surface (chest or abdomen)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Surface:
    name: str
    cfg: TAAConfig
    fs_nominal: float
    sig: Deque[float] = field(init=False)
    ts:  Deque[float] = field(init=False)

    def __post_init__(self):
        maxlen = int(round(self.cfg.window_s * self.fs_nominal)) + 10
        self.sig = deque(maxlen=maxlen)
        self.ts  = deque(maxlen=maxlen)

    def append(self, value: float, ts: float) -> None:
        self.sig.append(float(value))
        self.ts.append(float(ts))

    def ready(self) -> bool:
        if len(self.sig) < 4:                               return False
        if self.ts[-1] - self.ts[0] < self.cfg.min_fill_s:  return False
        return True

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.sig, dtype=np.float32), np.asarray(self.ts, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _effective_fs(ts: np.ndarray, fallback: float) -> float:
    if len(ts) < 2: return fallback
    dt = ts[-1] - ts[0]
    return (len(ts) - 1) / dt if dt > 0 else fallback


def _bandpass_hilbert(sig: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """Linear detrend -> Butterworth bandpass -> Hilbert analytic signal."""
    detr = scipy.signal.detrend(sig, type="linear")
    nyq  = 0.5 * fs
    lo   = max(1e-4, band[0] / nyq)
    hi   = min(0.99, band[1] / nyq)
    if lo >= hi:
        return scipy.signal.hilbert(detr)
    b, a = scipy.signal.butter(2, [lo, hi], btype="bandpass")
    filt = scipy.signal.filtfilt(b, a, detr)
    return scipy.signal.hilbert(filt)


def _wrapped_abs_phase_diff(ac: np.ndarray, aa: np.ndarray) -> np.ndarray:
    """
    |phase_diff| folded to [0, pi]. Direction (chest leading vs lagging) does not
    matter clinically — asynchrony is asynchrony.
    """
    diff = np.angle(ac) - np.angle(aa)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi    # wrap to (-pi, pi]
    return np.abs(diff)


def _amplitude(sig: np.ndarray) -> float:
    """Peak-to-peak amplitude via 10-90 percentile range (robust to outliers)."""
    return float(np.percentile(sig, 90) - np.percentile(sig, 10))


# ─────────────────────────────────────────────────────────────────────────────
# ROI sampling
# ─────────────────────────────────────────────────────────────────────────────

def _crop_depth(depth: np.ndarray, roi: _ROICfg) -> np.ndarray:
    h, w = depth.shape[:2]
    x1, x2 = int(w * roi.x[0]), int(w * roi.x[1])
    y1, y2 = int(h * roi.y[0]), int(h * roi.y[1])
    return depth[y1:y2, x1:x2]


def _mean_valid(crop: Optional[np.ndarray], min_valid: int, min_pixels: int) -> Optional[float]:
    if crop is None or crop.size == 0:
        return None
    valid = crop[crop > min_valid]
    if valid.size < min_pixels:
        return None
    return float(np.mean(valid))


# ─────────────────────────────────────────────────────────────────────────────
# Sustain tracker — timebar for how long the threshold has been continuously exceeded
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SustainState:
    first_high_ts: Optional[float] = None

    def update(self, high: bool, ts: float, min_duration_s: float) -> Tuple[bool, float]:
        if high:
            if self.first_high_ts is None:
                self.first_high_ts = ts
            dur = ts - self.first_high_ts
            return (dur >= min_duration_s, dur)
        self.first_high_ts = None
        return (False, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public module
# ─────────────────────────────────────────────────────────────────────────────

class TAAModule:
    """
    Per-tick entry: process(bundle, roi) -> {"taa.depth": Measurement}
    The dict stays empty while the buffer is filling or TAA is disabled.
    """

    def __init__(self, cfg: TAAConfig, fs: float = 30.0):
        self.cfg = cfg
        self._chest   = _Surface("chest",   cfg, fs)
        self._abdomen = _Surface("abdomen", cfg, fs)
        self._sustain = _SustainState()
        self._last_estimate_ts: float = 0.0

    def reset(self):
        self._chest   = _Surface("chest",   self.cfg, self._chest.fs_nominal)
        self._abdomen = _Surface("abdomen", self.cfg, self._abdomen.fs_nominal)
        self._sustain = _SustainState()
        self._last_estimate_ts = 0.0

    # ---- per-tick -----------------------------------------------------------

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> Dict[str, Measurement]:
        if not self.cfg.enabled:
            return {}

        now = bundle.ts_mono
        if roi is not None and roi.chest_depth is None:
            # pose runs but nobody is tracked: fixed boxes would compare chair and wall
            if self._chest.sig or self._abdomen.sig:
                self.reset()
            return {}
        self._sample(bundle, roi, now)

        if not (self._chest.ready() and self._abdomen.ready()):
            return {}
        if now - self._last_estimate_ts < self.cfg.update_every_s:
            return {}
        self._last_estimate_ts = now

        m = self._estimate()
        return {"taa.depth": m} if m is not None else {}

    # ---- sampling -----------------------------------------------------------

    def _sample(self, bundle: FrameBundle, roi: Optional[ROIBundle], now: float) -> None:
        chest_crop = roi.chest_depth if (roi is not None and roi.chest_depth is not None
                                         and roi.chest_depth.size > 0) \
                                     else _crop_depth(bundle.depth, self.cfg.chest_fallback_roi)
        abd_crop   = roi.abdomen_depth if (roi is not None and roi.abdomen_depth is not None
                                           and roi.abdomen_depth.size > 0) \
                                       else _crop_depth(bundle.depth, self.cfg.abdomen_fallback_roi)

        c = _mean_valid(chest_crop, self.cfg.min_valid_depth_units, self.cfg.min_valid_pixels)
        a = _mean_valid(abd_crop,   self.cfg.min_valid_depth_units, self.cfg.min_valid_pixels)

        # one-sample hold on brief dropouts, mirrors rr.py
        if c is not None:                       self._chest.append(c, now)
        elif self._chest.sig:                   self._chest.append(self._chest.sig[-1], now)
        if a is not None:                       self._abdomen.append(a, now)
        elif self._abdomen.sig:                 self._abdomen.append(self._abdomen.sig[-1], now)

    # ---- estimation ---------------------------------------------------------

    def _estimate(self) -> Optional[Measurement]:
        t0 = time.perf_counter()
        c_sig, c_ts = self._chest.snapshot()
        a_sig, a_ts = self._abdomen.snapshot()

        fs_c = _effective_fs(c_ts, self._chest.fs_nominal)
        fs_a = _effective_fs(a_ts, self._abdomen.fs_nominal)
        fs   = min(fs_c, fs_a)                 # share one fs across both analytic signals

        # amplitude gate: if there is no breathing to speak of, phase is noise
        c_amp = _amplitude(c_sig)
        a_amp = _amplitude(a_sig)
        if c_amp < self.cfg.min_amplitude_units or a_amp < self.cfg.min_amplitude_units:
            # Not measurable: the asynchrony streak is broken, so a later high reading
            # must sustain for the full window again rather than resume the old count.
            self._sustain = _SustainState()
            # Return a Measurement with value=None so the UI can show "insufficient signal"
            # instead of a confident 0 radians.
            return Measurement(
                value=None, sqi=0.0, ts=time.time(), source="taa.depth",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                meta={"reason": "amplitude_below_min",
                      "chest_amp": round(c_amp, 3), "abdomen_amp": round(a_amp, 3),
                      "fs_hz": round(fs, 3)},
            )

        # analytic signals + wrapped |phase diff|
        ac = _bandpass_hilbert(c_sig, fs, self.cfg.band_hz)
        aa = _bandpass_hilbert(a_sig, fs, self.cfg.band_hz)
        n = min(len(ac), len(aa))
        phase_diff = _wrapped_abs_phase_diff(ac[-n:], aa[-n:])

        # recent sub-window = last update_every_s, but at least 1 s worth of samples
        recent_n = max(int(fs * max(self.cfg.update_every_s, 1.0)), 4)
        recent = phase_diff[-recent_n:]
        mean_recent = float(np.mean(recent))

        high = mean_recent >= self.cfg.phase_diff_threshold_rad
        taa_flag, sustain_s = self._sustain.update(high, max(c_ts[-1], a_ts[-1]),
                                                   self.cfg.sustain_window_s)

        # SQI = how clearly chest + abdomen amplitudes are above the min floor.
        # Clamp into [0, 1] via log scaling: amp == min -> 0, amp == 10x min -> 1.
        min_amp = max(self.cfg.min_amplitude_units, 1e-3)
        sqi_c = float(np.clip(np.log1p((c_amp - min_amp) / min_amp) / np.log(11), 0, 1))
        sqi_a = float(np.clip(np.log1p((a_amp - min_amp) / min_amp) / np.log(11), 0, 1))
        sqi   = float(min(sqi_c, sqi_a))

        return Measurement(
            value=mean_recent,                        # radians, folded to [0, pi]
            sqi=sqi,
            ts=time.time(),
            source="taa.depth",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={
                "taa_flag":        bool(taa_flag),
                "sustain_s":       round(sustain_s, 2),
                "phase_diff_rad":  round(mean_recent, 3),
                "phase_diff_deg":  round(float(np.degrees(mean_recent)), 1),
                "threshold_rad":   self.cfg.phase_diff_threshold_rad,
                "chest_amp":       round(c_amp, 3),
                "abdomen_amp":     round(a_amp, 3),
                "fs_hz":           round(fs, 3),
                "window_s":        round(c_ts[-1] - c_ts[0], 2),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_taa_module(signal_yaml_path: str = "config/signal.yaml",
                    fs: float = 30.0) -> TAAModule:
    import yaml
    cfg = {}
    try:
        with open(signal_yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    return TAAModule(TAAConfig.from_dict(cfg), fs=fs)