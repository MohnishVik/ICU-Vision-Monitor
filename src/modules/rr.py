"""
rr.py — Respiratory Rate module

Two signal-processing paths run in parallel. The learned CNNs live in src/modules/learned.py
and are shown alongside (fused only when signal.yaml learned.fuse is true). Outputs are fused by inverse-variance weighting
in src/fusion/inv_variance.py.

Paths implemented here
  depth    — chest ROI mean depth -> detrend -> smooth -> Welch PSD -> peak in band
             Works in complete darkness. Also runs the apnea detector.
  thermal  — nostril/mouth ROI mean temperature -> same DSP
             Skipped cleanly when FrameBundle.thermal is None or the thermal frame
             has not changed since the last tick (thermal_ts repeats).

Quality gate (shared)
  Signal power is concentrated in a narrow peak when breathing is steady and spreads
  across the band when it is not. peak_power_frac = power within +-peak_half_width_hz
  of the strongest peak / total power in the band. Below peak_power_frac_min the
  Measurement is returned with value=None; above peak_power_frac_good sqi reaches 1.0.

Apnea gate (depth path only, computed before the spectral estimate)
  Compares the amplitude of the recent sub-window against the full window. A sustained
  drop below amplitude_ratio_below sets meta["apnea"]=True and forces value=None, so a
  spectral peak can never report "normal breathing" during a breath-hold.

Nothing in here reads cameras or writes files.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
import scipy.signal

from src.types import FrameBundle, Measurement, ROIBundle


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass (built from config/signal.yaml at load time)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RRPathCfg:
    """Settings shared by both depth and thermal RR paths."""
    window_s: float
    update_every_s: float
    min_fill_s: float
    band_hz: tuple[float, float]
    detrend: str = "linear"               # "linear" (live) or "quadratic" (legacy offline)
    smooth_s: float = 0.35
    nfft: int = 16384


@dataclass
class _QualityCfg:
    peak_power_frac_min:  float = 0.30
    peak_power_frac_good: float = 0.60
    peak_half_width_hz:   float = 0.03


@dataclass
class _ApneaCfg:
    window_s: float            = 10.0
    amplitude_ratio_below: float = 0.25
    min_duration_s: float      = 10.0


@dataclass
class RRConfig:
    """Built once at startup from the parsed signal.yaml['rr'] dict."""
    reported_range_bpm: tuple[float, float]
    depth: _RRPathCfg
    thermal: _RRPathCfg
    quality: _QualityCfg
    apnea:   _ApneaCfg
    fallback_roi_frac: dict         # {"x":[0.44,0.65], "y":[0.54,0.82]} for depth
    min_valid_depth_units: int = 5
    min_valid_pixels: int      = 30

    @classmethod
    def from_dict(cls, cfg: dict) -> "RRConfig":
        rr = cfg["rr"]
        d, t = rr["depth"], rr["thermal"]
        return cls(
            reported_range_bpm=tuple(rr["reported_range_bpm"]),
            depth=_RRPathCfg(
                window_s=float(d["window_s"]),
                update_every_s=float(d["update_every_s"]),
                min_fill_s=float(d["min_fill_s"]),
                band_hz=tuple(d["band_hz"]),
                detrend=d.get("detrend", "linear"),
                smooth_s=float(d.get("smooth_s", 0.35)),
                nfft=int(d.get("psd", {}).get("nfft", 16384)),
            ),
            thermal=_RRPathCfg(
                window_s=float(t["window_s"]),
                update_every_s=float(t["update_every_s"]),
                min_fill_s=float(t["min_fill_s"]),
                band_hz=tuple(t["band_hz"]),
            ),
            quality=_QualityCfg(**{k: rr["quality"][k] for k in
                                   ("peak_power_frac_min", "peak_power_frac_good",
                                    "peak_half_width_hz")}),
            apnea=_ApneaCfg(**{k: rr["apnea"][k] for k in
                               ("window_s", "amplitude_ratio_below", "min_duration_s")}),
            fallback_roi_frac=d["fallback_roi_frac"],
            min_valid_depth_units=int(d.get("min_valid_depth_units", 5)),
            min_valid_pixels=int(d.get("min_valid_pixels", 30)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal rolling-buffer path (one per signal source)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Path:
    """A sampled 1-D signal + the timestamps of its samples.
    Appends run at the camera tick rate; estimate() runs on a slower schedule."""
    cfg: _RRPathCfg
    fs_nominal: float                  # 30 Hz for depth (D435), ~8 Hz for thermal (ESP32)
    source: str                        # "rr.depth" | "rr.thermal"
    sig: Deque[float]  = field(init=False)
    ts:  Deque[float]  = field(init=False)
    last_estimate_ts: float = 0.0

    def __post_init__(self):
        maxlen = int(round(self.cfg.window_s * self.fs_nominal)) + 10
        self.sig = deque(maxlen=maxlen)
        self.ts  = deque(maxlen=maxlen)

    def append(self, value: float, ts: float) -> None:
        self.sig.append(float(value))
        self.ts.append(float(ts))

    def reset(self) -> None:
        """Drop the buffer, e.g. when the signal source changes (two different
        signals spliced into one window would corrupt the spectrum)."""
        self.sig.clear()
        self.ts.clear()
        self.last_estimate_ts = 0.0

    def ready(self, now: float) -> bool:
        """True when enough signal exists and the update interval has elapsed."""
        if len(self.sig) < 4:
            return False
        if self.ts[-1] - self.ts[0] < self.cfg.min_fill_s:
            return False
        if now - self.last_estimate_ts < self.cfg.update_every_s:
            return False
        return True

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy the current buffer out so DSP doesn't fight new samples."""
        return np.asarray(self.sig, dtype=np.float32), np.asarray(self.ts, dtype=np.float64)


def _effective_fs(ts: np.ndarray, fallback: float) -> float:
    """Measured sample rate across the window; falls back to the nominal value."""
    if len(ts) < 2:
        return fallback
    dt = ts[-1] - ts[0]
    return (len(ts) - 1) / dt if dt > 0 else fallback


def _detrend(sig: np.ndarray, kind: str) -> np.ndarray:
    """Remove a linear or quadratic trend before spectral estimation."""
    if kind == "quadratic":
        t = np.arange(len(sig), dtype=np.float32)
        return sig - np.polyval(np.polyfit(t, sig, 2), t)
    return scipy.signal.detrend(sig, type="linear")


def _smooth(sig: np.ndarray, win_samples: int) -> np.ndarray:
    """Moving-average smoother used to attenuate cardiac micro-vibrations."""
    win_samples = max(3, int(win_samples))
    return np.convolve(sig, np.ones(win_samples, dtype=np.float32) / win_samples, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# Spectral estimate + quality score
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_rate_bpm(
    sig: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    quality: _QualityCfg,
    nfft: int,
) -> tuple[Optional[float], float, dict]:
    """
    Welch PSD -> strongest peak inside band_hz.
    Returns (rate_bpm, peak_power_frac, debug). rate_bpm is None if the band is empty
    after filtering.
    """
    nperseg = min(len(sig), max(32, int(fs * 10)))
    freqs, psd = scipy.signal.welch(
        sig, fs=fs, window="hann", nperseg=nperseg, nfft=nfft
    )

    idx = np.where((freqs >= band_hz[0]) & (freqs <= band_hz[1]))[0]
    if len(idx) < 3:
        return None, 0.0, {}

    sub_freqs = freqs[idx]
    sub_psd   = psd[idx]
    total_power = float(np.sum(sub_psd))
    if total_power <= 0:
        return None, 0.0, {}

    peak_i = int(np.argmax(sub_psd))
    peak_freq = float(sub_freqs[peak_i])

    half = quality.peak_half_width_hz
    peak_mask = (sub_freqs >= peak_freq - half) & (sub_freqs <= peak_freq + half)
    peak_power = float(np.sum(sub_psd[peak_mask]))
    frac = peak_power / total_power

    return peak_freq * 60.0, frac, {"peak_freq_hz": peak_freq, "band_power": total_power}


def _sqi_from_frac(frac: float, q: _QualityCfg) -> float:
    """Linearly map peak_power_frac in [min, good] -> sqi in [0, 1]."""
    if frac <= q.peak_power_frac_min:
        return 0.0
    if frac >= q.peak_power_frac_good:
        return 1.0
    return (frac - q.peak_power_frac_min) / (q.peak_power_frac_good - q.peak_power_frac_min)


# ─────────────────────────────────────────────────────────────────────────────
# Apnea detector (depth only; thermal cools too slowly on a breath-hold to be reliable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ApneaState:
    """Tracks how long the amplitude-drop condition has been continuously true."""
    first_low_ts: Optional[float] = None

    def update(self, triggered: bool, ts: float, min_duration_s: float) -> tuple[bool, float]:
        if triggered:
            if self.first_low_ts is None:
                self.first_low_ts = ts
            dur = ts - self.first_low_ts
            return (dur >= min_duration_s, dur)
        self.first_low_ts = None
        return (False, 0.0)


def _is_apnea_now(
    sig: np.ndarray,
    ts:  np.ndarray,
    fs:  float,
    apnea_cfg: _ApneaCfg,
) -> bool:
    """
    Compare the recent sub-window amplitude to the full-window amplitude.
    Uses the 10th-90th percentile range so a single outlying sample can't dominate.
    """
    recent_n = int(apnea_cfg.window_s * fs)
    if len(sig) < 2 * recent_n:
        return False
    recent_amp = float(np.percentile(sig[-recent_n:], 90) - np.percentile(sig[-recent_n:], 10))
    window_amp = float(np.percentile(sig,            90) - np.percentile(sig,            10))
    if window_amp <= 1e-6:
        return False
    return (recent_amp / window_amp) < apnea_cfg.amplitude_ratio_below


# ─────────────────────────────────────────────────────────────────────────────
# ROI sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_from_depth_roi(depth_crop: np.ndarray, min_valid: int, min_pixels: int) -> Optional[float]:
    """Mean of valid depth pixels in a crop. Returns None when the crop is unusable."""
    if depth_crop is None or depth_crop.size == 0:
        return None
    valid = depth_crop[depth_crop > min_valid]
    if valid.size < min_pixels:
        return None
    return float(np.mean(valid))


def _fallback_depth_roi(depth: np.ndarray, roi_frac: dict) -> np.ndarray:
    """Fixed sternal box used when the scene layer provides no pose-based ROI."""
    h, w = depth.shape[:2]
    x1, x2 = int(w * roi_frac["x"][0]), int(w * roi_frac["x"][1])
    y1, y2 = int(h * roi_frac["y"][0]), int(h * roi_frac["y"][1])
    return depth[y1:y2, x1:x2]


def _nostril_mean_c(thermal: np.ndarray) -> Optional[float]:
    """
    Mean temperature over the warmest patch inside the thermal frame.
    Placeholder until a thermal<->RGB homography provides the real nostril ROI.
    The warmest patch tracks the face reliably at mount distance, and nostril
    temperature is the warmest point there.
    """
    if thermal is None or thermal.size == 0:
        return None
    flat = thermal.ravel()
    top = np.partition(flat, -max(1, flat.size // 50))[-max(1, flat.size // 50):]
    return float(np.mean(top))


# ─────────────────────────────────────────────────────────────────────────────
# Public module
# ─────────────────────────────────────────────────────────────────────────────

class RRModule:
    """
    Receives (FrameBundle, ROIBundle) each tick. Returns a dict of Measurement objects:
      {"rr.depth":   Measurement(...), "rr.thermal": Measurement(...)}
    Keys appear only when the path produced at least one estimate this tick.
    Fusion of the two paths lives in src/fusion/inv_variance.py.
    """

    def __init__(self, cfg: RRConfig, depth_fs: float = 30.0, thermal_fs: float = 8.0):
        self.cfg = cfg
        self._depth   = _Path(cfg=cfg.depth,   fs_nominal=depth_fs,   source="rr.depth")
        self._thermal = _Path(cfg=cfg.thermal, fs_nominal=thermal_fs, source="rr.thermal")
        self._apnea_state = _ApneaState()
        self._last_thermal_ts: Optional[float] = None
        self._thermal_src: Optional[str] = None     # "nostril" | "warmest_patch"

    # ---- per-tick entry point ------------------------------------------------

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> dict:
        """Append samples, then compute each path on its own schedule."""
        now = bundle.ts_mono
        self._sample_depth(bundle, roi, now)
        self._sample_thermal(bundle, roi, now)

        out: dict[str, Measurement] = {}
        if self._depth.ready(now):
            m = self._estimate(self._depth, source="rr.depth", with_apnea=True)
            if m is not None:
                out[m.source] = m
                self._depth.last_estimate_ts = now
        if self._thermal.ready(now) and self._thermal.fs_nominal > 0:
            m = self._estimate(self._thermal, source="rr.thermal", with_apnea=False)
            if m is not None:
                out[m.source] = m
                self._thermal.last_estimate_ts = now
        return out

    # ---- sampling ------------------------------------------------------------

    def _sample_depth(self, bundle: FrameBundle, roi: Optional[ROIBundle], now: float) -> None:
        # Prefer the pose-based crop from the scene layer; fall back to a fixed box
        if roi is not None and roi.chest_depth is not None and roi.chest_depth.size > 0:
            crop = roi.chest_depth
        else:
            crop = _fallback_depth_roi(bundle.depth, self.cfg.fallback_roi_frac)

        mean_z = _mean_from_depth_roi(crop, self.cfg.min_valid_depth_units, self.cfg.min_valid_pixels)
        if mean_z is not None:
            self._depth.append(mean_z, now)
        elif self._depth.sig:
            # One-sample hold prevents the buffer from going stale during a brief dropout
            self._depth.append(self._depth.sig[-1], now)

    def _sample_thermal(self, bundle: FrameBundle, roi: Optional[ROIBundle], now: float) -> None:
        if bundle.thermal is None or bundle.thermal_ts is None:
            return
        if bundle.thermal_ts == self._last_thermal_ts:
            return                                    # same frame as the previous tick
        self._last_thermal_ts = bundle.thermal_ts

        # Prefer the nostril ROI (needs thermal<->RGB registration + a visible nose);
        # otherwise fall back to the warmest patch.
        temp, src = None, "warmest_patch"
        t_roi = getattr(roi, "thermal_roi", None) if roi is not None else None
        if t_roi is not None:
            x0, y0, x1, y1 = (int(v) for v in t_roi)
            patch = bundle.thermal[y0:y1, x0:x1]
            if patch.size > 0 and np.isfinite(patch).all():
                temp, src = float(np.mean(patch)), "nostril"
        if temp is None:
            temp = _nostril_mean_c(bundle.thermal)
        if temp is None:
            return
        if src != self._thermal_src:
            if self._thermal_src is not None:
                self._thermal.reset()                 # never splice two different signals
            self._thermal_src = src
        self._thermal.append(temp, bundle.thermal_ts)

    # ---- estimation ----------------------------------------------------------

    def _estimate(self, path: _Path, source: str, with_apnea: bool) -> Optional[Measurement]:
        t0 = time.perf_counter()
        sig, ts = path.snapshot()
        fs = _effective_fs(ts, path.fs_nominal)

        # Apnea gate first: never report a spectral peak as "normal" during a breath-hold
        apnea_now = False
        apnea_dur = 0.0
        if with_apnea:
            trig = _is_apnea_now(sig, ts, fs, self.cfg.apnea)
            apnea_now, apnea_dur = self._apnea_state.update(trig, ts[-1], self.cfg.apnea.min_duration_s)

        detrended = _detrend(sig, path.cfg.detrend)
        smoothed  = _smooth(detrended, int(path.cfg.smooth_s * fs))

        rate_bpm, frac, dbg = _spectral_rate_bpm(
            smoothed, fs, path.cfg.band_hz, self.cfg.quality, path.cfg.nfft
        )
        sqi = _sqi_from_frac(frac, self.cfg.quality)

        # value=None whenever the quality gate fails or an apnea is confirmed
        value: Optional[float] = rate_bpm
        if rate_bpm is None or frac < self.cfg.quality.peak_power_frac_min:
            value = None
        if apnea_now:
            value = None
            sqi   = 1.0            # we are confident NO breathing is occurring

        latency_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "fs_hz": round(fs, 3),
            "peak_power_frac": round(frac, 3),
            "window_s": round(ts[-1] - ts[0], 2),
            **dbg,
        }
        if with_apnea:
            meta["apnea"] = bool(apnea_now)
            meta["apnea_duration_s"] = round(apnea_dur, 2)
        if source == "rr.thermal":
            meta["roi"] = self._thermal_src or "warmest_patch"

        return Measurement(
            value=value,
            sqi=float(np.clip(sqi, 0.0, 1.0)),
            ts=time.time(),
            source=source,
            latency_ms=latency_ms,
            meta=meta,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_rr_module(signal_yaml_path: str = "config/signal.yaml",
                   depth_fs: float = 30.0,
                   thermal_fs: float = 8.0) -> RRModule:
    """Build an RRModule straight from the YAML, for scripts/run_monitor.py."""
    import yaml
    with open(signal_yaml_path) as f:
        cfg = yaml.safe_load(f)
    return RRModule(RRConfig.from_dict(cfg), depth_fs=depth_fs, thermal_fs=thermal_fs)