"""
hr.py — Heart Rate module (rPPG)

Primary path: POS (Plane-Orthogonal-to-Skin). No weights. Always on.
             Lifted from your evaluate_hr_pos.py, wrapped in the Measurement contract
             and the same peak-power-fraction quality gate used by rr.py.
Secondary:   EfficientPhys lives in src/modules/learned.py and is shown alongside, fused only when
             signal.yaml learned.fuse is true; on 7 held-out UBFC subjects it gave ~100 BPM for everyone.

ROI
  Prefers roi.face_roi from the scene layer. Falls back to a fixed upper-centre box
  from signal.yaml -> hr.pos.fallback_roi_frac. From that crop we take the mean R, G, B
  of the YCrCb skin-masked pixels (your own fallback — robust across skin tones).

Quality
  POS can latch onto brightness oscillations when the signal is thin. The spectral
  gate reports value=None whenever peak_power_frac < hr.quality.peak_power_frac_min,
  so a quiet-skin or low-light tick returns "unavailable" instead of a confident
  wrong number.

Motion and light
  A head turn or a hand across the face shifts the skin brightness by several percent in a
  fraction of a second, far more than a heartbeat (well under 1 %). When the half-second
  mean skin brightness steps by more than motion_rel the last good colour is held and,
  once still, the channels are rescaled to continue where they were (src/modules/motion.py).
  Only movement longer than motion_max_hold_s restarts the window; while it refills the
  module reports value=None with meta["reason"] = "motion". When a reading is
  rejected and the face averages below low_light_luma (0-255), the reason is "low_light":
  sensor noise is then larger than the pulse and more light on the face is the fix.

Nothing in here reads cameras or writes files.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import cv2
import numpy as np
import scipy.signal

from src.modules.motion import MotionBridge, MotionGate
from src.types import FrameBundle, Measurement, ROIBundle


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _POSCfg:
    inner_window_s: float               # 1.6 s POS sliding window (Wang et al.)
    window_s: float                     # rolling buffer length used for one estimate
    update_every_s: float
    min_fill_s: float
    band_hz: tuple[float, float]
    bandpass_order: int = 2
    nfft: int           = 16384
    fallback_roi_frac: dict = field(default_factory=lambda: {
        "x": [0.25, 0.75], "y": [0.08, 0.50]
    })
    segment_s: float    = 10.0          # Welch segment; 20 s = one finer-resolution spectrum
    resample_hz: float  = 30.0          # put samples on an even grid before POS (0 = off)


@dataclass
class _SteadyCfg:
    """Accept a weaker peak when it has stayed put: `count` estimates in a row with
    peak_power_frac >= frac_min, all within +-tol_bpm of each other. Noise wanders from
    window to window; a pulse does not."""
    frac_min: float = 0.15
    count: int      = 6
    tol_bpm: float  = 4.0
    sqi: float      = 0.30              # quality reported for such a reading (low fusion weight)


@dataclass
class _QualityCfg:
    peak_power_frac_min:  float = 0.25
    peak_power_frac_good: float = 0.50
    peak_half_width_hz:   float = 0.05


@dataclass
class HRConfig:
    reported_range_bpm: tuple[float, float]
    pos: _POSCfg
    quality: _QualityCfg
    motion_rel: float = 0.05            # half-second skin-brightness step that counts as movement
    motion_max_hold_s: float = 3.0      # bridge movements up to this long; restart after
    low_light_luma: float = 60.0        # face darker than this -> "low_light" when rejected
    steady: Optional[_SteadyCfg] = None  # None: only the peak_power_frac gate

    @classmethod
    def from_dict(cls, cfg: dict) -> "HRConfig":
        hr = cfg["hr"]
        p = hr["pos"]
        st = hr.get("steady")
        return cls(
            steady=None if not st else _SteadyCfg(**{k: st[k] for k in ("frac_min", "count", "tol_bpm", "sqi")
                                                    if k in st}),
            reported_range_bpm=tuple(hr["reported_range_bpm"]),
            pos=_POSCfg(
                inner_window_s=float(p["inner_window_s"]),
                window_s=float(p["window_s"]),
                update_every_s=float(p["update_every_s"]),
                min_fill_s=float(p["min_fill_s"]),
                band_hz=tuple(p["band_hz"]),
                bandpass_order=int(p.get("bandpass_order", 2)),
                nfft=int(p.get("psd", {}).get("nfft", 16384)),
                fallback_roi_frac=p.get("fallback_roi_frac", {
                    "x": [0.25, 0.75], "y": [0.08, 0.50]
                }),
                segment_s=float(p.get("segment_s", 10.0)),
                resample_hz=float(p.get("resample_hz", 30.0)),
            ),
            quality=_QualityCfg(**{k: hr["quality"][k] for k in
                                   ("peak_power_frac_min", "peak_power_frac_good",
                                    "peak_half_width_hz")}),
            motion_rel=float(p.get("motion_rel", 0.05)),
            motion_max_hold_s=float(p.get("motion_max_hold_s", 3.0)),
            low_light_luma=float(p.get("low_light_luma", 60.0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Skin-masked RGB extractor (your evaluate_hr_pos.py algorithm)
# ─────────────────────────────────────────────────────────────────────────────

# Standard YCrCb thresholds that cover most skin tones under varied lighting.
_YCRCB_LO = np.array([0, 133, 77],  dtype=np.uint8)
_YCRCB_HI = np.array([255, 173, 127], dtype=np.uint8)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _mean_skin_rgb(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns mean [R, G, B] over skin pixels inside face_bgr, or None if the crop
    is unusable (empty, too dark, or no skin found).

    Order matters: OpenCV reads BGR, POS expects RGB. We swap channels at the end.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb, _YCRCB_LO, _YCRCB_HI)
    mask  = cv2.erode (mask, _MORPH_KERNEL, iterations=1)
    mask  = cv2.dilate(mask, _MORPH_KERNEL, iterations=2)

    skin_pixels = face_bgr[mask > 0]
    if len(skin_pixels) >= 100:
        mean_bgr = np.mean(skin_pixels, axis=0)
    else:
        # Fallback to centre forehead region (same behaviour as your original)
        mean_bgr = np.mean(face_bgr.reshape(-1, 3), axis=0)

    # BGR -> RGB
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)


def _luma(rgb_mean: np.ndarray) -> float:
    return float(0.299 * rgb_mean[0] + 0.587 * rgb_mean[1] + 0.114 * rgb_mean[2])


def _fallback_face_roi(rgb: np.ndarray, roi_frac: dict) -> np.ndarray:
    """Fixed upper-centre box used when scene layer doesn't supply a face crop."""
    h, w = rgb.shape[:2]
    x1, x2 = int(w * roi_frac["x"][0]), int(w * roi_frac["x"][1])
    y1, y2 = int(h * roi_frac["y"][0]), int(h * roi_frac["y"][1])
    return rgb[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# POS (Wang et al. 2016) — exactly the maths from your evaluate_hr_pos.py
# ─────────────────────────────────────────────────────────────────────────────

def _pos_pulse_wave(rgb_trace: np.ndarray, fs: float, inner_window_s: float) -> Optional[np.ndarray]:
    """
    Compute the POS blood-volume-pulse wave from a (3, T) RGB trace.
    Returns a 1-D wave of length T, or None if the trace is too short.
    """
    T = rgb_trace.shape[1]
    inner = int(fs * inner_window_s)
    if T <= inner:
        return None

    h = np.zeros(T, dtype=np.float32)
    for t in range(T - inner):
        chunk = rgb_trace[:, t:t + inner]
        mean  = np.mean(chunk, axis=1, keepdims=True)
        mean[mean == 0] = 1e-6
        norm  = chunk / mean
        # S1 = G - B,  S2 = G + B - 2R
        s1 = norm[1] - norm[2]
        s2 = norm[1] + norm[2] - 2.0 * norm[0]
        std_s2 = float(np.std(s2))
        alpha  = (float(np.std(s1)) / std_s2) if std_s2 > 1e-6 else 0.0
        p = s1 + alpha * s2
        h[t:t + inner] += (p - np.mean(p))
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Spectral estimate + quality score (same shape as rr.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resample_uniform(rgb: np.ndarray, ts: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation of a (3, T) trace onto an even time grid.
    Frames reach the modules at uneven intervals (pose inference and the browser stream
    take 30-200 ms per frame), but POS and the spectrum treat sample n as time n/fs. That
    mismatch smears the pulse peak across neighbouring frequencies; on the paced recording
    re-gridding cut the spread of consecutive HR estimates from 12.8 to 4.2 bpm.
    """
    grid = np.arange(ts[0], ts[-1], 1.0 / fs)
    if len(grid) < 4:
        return rgb, ts
    return np.vstack([np.interp(grid, ts, c) for c in rgb]).astype(np.float32), grid


def _spectral_rate_bpm(
    sig: np.ndarray, fs: float, band_hz: tuple[float, float],
    quality: _QualityCfg, nfft: int, segment_s: float = 10.0,
) -> tuple[Optional[float], float, dict]:
    nperseg = min(len(sig), max(32, int(fs * segment_s)))
    freqs, psd = scipy.signal.welch(sig, fs=fs, window="hann", nperseg=nperseg, nfft=nfft)
    idx = np.where((freqs >= band_hz[0]) & (freqs <= band_hz[1]))[0]
    if len(idx) < 3:
        return None, 0.0, {}
    sub_freqs, sub_psd = freqs[idx], psd[idx]
    total_power = float(np.sum(sub_psd))
    if total_power <= 0:
        return None, 0.0, {}
    peak_i    = int(np.argmax(sub_psd))
    peak_freq = float(sub_freqs[peak_i])
    half      = quality.peak_half_width_hz
    peak_power = float(np.sum(sub_psd[(sub_freqs >= peak_freq - half) & (sub_freqs <= peak_freq + half)]))
    frac = peak_power / total_power
    return peak_freq * 60.0, frac, {"peak_freq_hz": peak_freq, "band_power": total_power}


def _sqi_from_frac(frac: float, q: _QualityCfg) -> float:
    if frac <= q.peak_power_frac_min:  return 0.0
    if frac >= q.peak_power_frac_good: return 1.0
    return (frac - q.peak_power_frac_min) / (q.peak_power_frac_good - q.peak_power_frac_min)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-buffer state for POS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _POSBuffer:
    """Holds the per-frame mean [R,G,B] samples and their timestamps."""
    cfg: _POSCfg
    fs_nominal: float
    r: Deque[float] = field(init=False)
    g: Deque[float] = field(init=False)
    b: Deque[float] = field(init=False)
    ts: Deque[float] = field(init=False)
    last_estimate_ts: float = 0.0

    def __post_init__(self):
        maxlen = int(round(self.cfg.window_s * self.fs_nominal)) + 10
        self.r, self.g, self.b, self.ts = (deque(maxlen=maxlen) for _ in range(4))

    def append(self, rgb_mean: np.ndarray, ts: float) -> None:
        self.r.append(float(rgb_mean[0]))
        self.g.append(float(rgb_mean[1]))
        self.b.append(float(rgb_mean[2]))
        self.ts.append(float(ts))

    def ready(self, now: float) -> bool:
        if len(self.ts) < 4:                                        return False
        if self.ts[-1] - self.ts[0] < self.cfg.min_fill_s:          return False
        if now - self.last_estimate_ts < self.cfg.update_every_s:   return False
        return True

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.stack([np.asarray(self.r), np.asarray(self.g), np.asarray(self.b)],
                       axis=0).astype(np.float32)
        return rgb, np.asarray(self.ts, dtype=np.float64)


def _effective_fs(ts: np.ndarray, fallback: float) -> float:
    if len(ts) < 2: return fallback
    dt = ts[-1] - ts[0]
    return (len(ts) - 1) / dt if dt > 0 else fallback


# ─────────────────────────────────────────────────────────────────────────────
# Public module
# ─────────────────────────────────────────────────────────────────────────────

class HRModule:
    """
    Per-tick entry: process(bundle, roi) -> {"hr.pos": Measurement(...)}
    The dict stays empty on ticks when POS has not reaccumulated enough signal.
    """

    def __init__(self, cfg: HRConfig, fs: float = 30.0):
        self.cfg = cfg
        self._fs = fs
        self._buf = _POSBuffer(cfg=cfg.pos, fs_nominal=fs)
        self._motion = MotionGate(fs, rel_thresh=cfg.motion_rel)
        self._bridge = MotionBridge(self._motion, mode="mul", max_hold_s=cfg.motion_max_hold_s, trigger=_luma)
        self._last_rgb_raw: Optional[np.ndarray] = None
        self._last_motion_report = 0.0
        n = cfg.steady.count if cfg.steady else 1
        self._recent: Deque[Optional[float]] = deque(maxlen=n)   # last estimates (None = too weak)
        self._face_lost_since: Optional[float] = None

    DROPOUT_HOLD_S = 1.0                # repeat the last colour through face dropouts this long

    def reset(self) -> None:
        """Drop all signal (patient left, or a new patient): windows refill from scratch."""
        self._buf = _POSBuffer(cfg=self.cfg.pos, fs_nominal=self._fs)
        self._bridge.reset()
        self._last_rgb_raw = None
        self._recent.clear()

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> dict:
        out = self._process(bundle, roi)
        now = bundle.ts_mono
        buf = self._buf
        filled = len(buf.ts) >= 4 and buf.ts[-1] - buf.ts[0] >= self.cfg.pos.min_fill_s
        if "hr.pos" not in out and self._motion.recent(now, self.cfg.pos.min_fill_s) \
                and not filled and now - self._last_motion_report >= self.cfg.pos.update_every_s:
            # refilling after movement: say so instead of going quiet
            self._last_motion_report = now
            out["hr.pos"] = Measurement(None, 0.0, time.time(), "hr.pos", 0.0, {"reason": "motion"})
        return out

    def _process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> dict:
        # 1. Pick the face crop. roi=None means no pose system: then a fixed box is used.
        # With pose running but no face found, nothing is measured (a fixed box over an
        # empty chair turns wall colour noise into a "pulse").
        now = bundle.ts_mono
        if roi is None:
            face = _fallback_face_roi(bundle.rgb, self.cfg.pos.fallback_roi_frac)
        elif roi.face_roi is not None and roi.face_roi.size > 0:
            face = roi.face_roi
        else:
            face = None

        # 2. Mean skin RGB -> buffer
        rgb_mean = _mean_skin_rgb(face) if face is not None else None
        if rgb_mean is None:
            if self._face_lost_since is None:
                self._face_lost_since = now
            if now - self._face_lost_since > self.DROPOUT_HOLD_S:
                # face gone for longer than a blink of the pose model: start over, and say why
                if self._buf.ts:
                    self.reset()
                    self._face_lost_since = now
                if now - self._last_motion_report >= self.cfg.pos.update_every_s:
                    self._last_motion_report = now
                    return {"hr.pos": Measurement(None, 0.0, time.time(), "hr.pos", 0.0, {"reason": "no_face"})}
                return {}
            rgb_mean = self._last_rgb_raw                # brief ROI dropout: repeat the last sample
        else:
            self._face_lost_since = None
        if rgb_mean is not None:
            self._last_rgb_raw = rgb_mean
            items, restart = self._bridge.feed(rgb_mean, bundle.ts_mono)
            if restart:                                  # moving longer than motion_max_hold_s
                self._buf = _POSBuffer(cfg=self.cfg.pos, fs_nominal=self._fs)
                self._recent.clear()
                return {}
            for t_item, value in items:                  # emitted ~0.5 s late (see MotionBridge)
                self._buf.append(np.asarray(value, dtype=np.float32), t_item)

        # 3. Estimate on the configured interval
        if not self._buf.ready(bundle.ts_mono):
            return {}

        m = self._estimate()
        self._buf.last_estimate_ts = bundle.ts_mono
        return {"hr.pos": m} if m is not None else {}

    # ---- estimation ----------------------------------------------------------

    def _estimate(self) -> Optional[Measurement]:
        t0 = time.perf_counter()
        rgb, ts = self._buf.snapshot()
        fs_in = _effective_fs(ts, self._buf.fs_nominal)
        fs = fs_in
        if self.cfg.pos.resample_hz > 0:
            rgb, ts = _resample_uniform(rgb, ts, self.cfg.pos.resample_hz)
            fs = self.cfg.pos.resample_hz

        # Build the POS pulse wave
        pulse = _pos_pulse_wave(rgb, fs, self.cfg.pos.inner_window_s)
        if pulse is None:
            return None

        # Butterworth bandpass + Welch peak
        nyq = 0.5 * fs
        lo  = self.cfg.pos.band_hz[0] / nyq
        hi  = min(0.99, self.cfg.pos.band_hz[1] / nyq)
        b, a = scipy.signal.butter(self.cfg.pos.bandpass_order, [lo, hi], btype="bandpass")
        filtered = scipy.signal.filtfilt(b, a, scipy.signal.detrend(pulse))

        rate_bpm, frac, dbg = _spectral_rate_bpm(
            filtered, fs, self.cfg.pos.band_hz, self.cfg.quality, self.cfg.pos.nfft, self.cfg.pos.segment_s
        )
        sqi   = _sqi_from_frac(frac, self.cfg.quality)
        value = rate_bpm if (rate_bpm is not None and frac >= self.cfg.quality.peak_power_frac_min) else None
        st = self.cfg.steady
        if st is not None:
            self._recent.append(rate_bpm if rate_bpm is not None and frac >= st.frac_min else None)
            recent = list(self._recent)
            if value is None and len(recent) == st.count and all(r is not None for r in recent) \
                    and max(recent) - min(recent) <= 2 * st.tol_bpm:
                value, sqi = rate_bpm, max(sqi, st.sqi)
                dbg["accepted"] = "steady"          # weak peak, but the same rate for count seconds
        face_luma = float(np.mean(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]))
        if value is None:
            # "weak_pulse": the face is lit well enough but no rhythm stands out (moving,
            # talking, or skin too dim for the camera); "low_light": more light is the fix
            dbg["reason"] = "low_light" if face_luma < self.cfg.low_light_luma else "weak_pulse"

        return Measurement(
            value=value,
            sqi=float(np.clip(sqi, 0.0, 1.0)),
            ts=time.time(),
            source="hr.pos",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={
                "fs_hz": round(fs_in, 3),
                "peak_power_frac": round(frac, 3),
                "window_s": round(ts[-1] - ts[0], 2),
                "face_luma": round(face_luma, 1),
                **dbg,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader (mirrors rr.load_rr_module)
# ─────────────────────────────────────────────────────────────────────────────

def load_hr_module(signal_yaml_path: str = "config/signal.yaml",
                   fs: float = 30.0) -> HRModule:
    import yaml
    with open(signal_yaml_path) as f:
        cfg = yaml.safe_load(f)
    return HRModule(HRConfig.from_dict(cfg), fs=fs)