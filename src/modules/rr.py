"""
rr.py — Respiratory Rate module

Two signal-processing paths run in parallel. The learned CNNs live in src/modules/learned.py
and are shown alongside (fused only when signal.yaml learned.fuse is true). Outputs are fused by inverse-variance weighting
in src/fusion/inv_variance.py.

Paths implemented here
  depth    — chest and abdomen ROI mean depth, each -> high-pass -> smooth -> Welch PSD;
             the two spectra are merged (each normalised, weighted by how clean it is)
             and the rate read off the merged peak. A chest breather and a belly breather
             both give a clear signal, and noise that only one surface has averages out.
             Works in complete darkness. Also runs the apnea detector.
  thermal  — nostril/mouth ROI mean temperature -> same DSP
             Skipped cleanly when FrameBundle.thermal is None or the thermal frame
             has not changed since the last tick (thermal_ts repeats).

Quality gate (shared)
  Signal power is concentrated in a narrow peak when breathing is steady and spreads
  across the band when it is not. peak_power_frac = power within +-peak_half_width_hz
  of the strongest peak / total power in the band. Below peak_power_frac_min the
  Measurement is returned with value=None; above peak_power_frac_good sqi reaches 1.0.

Motion gate (depth path)
  A posture change moves the chest box by tens of millimetres in a fraction of a second,
  more than all the breathing in the window put together. When the half-second mean
  depth jumps by more than motion_mm (src/modules/motion.py) the last good value is held
  and, once still, the trace is re-baselined to continue where it was, so the window and
  the reading survive a brief shift. Only movement longer than motion_max_hold_s drops
  the window; while it refills the path reports value=None with meta["reason"] = "motion",
  so the screen says "Patient moving", not "signal too weak".

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

from src.modules.motion import MotionBridge, MotionGate
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
    highpass_hz: Optional[float] = None   # remove posture drift slower than this (replaces detrend)
    segment_s: float = 10.0               # Welch segment length; longer resolves slow breathing


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
    motion_mm: float           = 20.0      # half-second mean depth step that counts as movement
    motion_max_hold_s: float   = 4.0       # bridge movements up to this long; restart after
    dropout_hold_s: float      = 1.0       # repeat the last sample through depth holes this long
    body_band_mm: Optional[float] = 150.0  # keep chest pixels within this of the median depth
    regions: tuple = ("chest", "abdomen")  # depth surfaces whose spectra rr.depth combines

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
                highpass_hz=(float(d["highpass_hz"]) if d.get("highpass_hz") else None),
                segment_s=float(d.get("segment_s", 10.0)),
            ),
            thermal=_RRPathCfg(
                window_s=float(t["window_s"]),
                update_every_s=float(t["update_every_s"]),
                min_fill_s=float(t["min_fill_s"]),
                band_hz=tuple(t["band_hz"]),
                highpass_hz=(float(t["highpass_hz"]) if t.get("highpass_hz") else None),
                segment_s=float(t.get("segment_s", 10.0)),
            ),
            quality=_QualityCfg(**{k: rr["quality"][k] for k in
                                   ("peak_power_frac_min", "peak_power_frac_good",
                                    "peak_half_width_hz")}),
            apnea=_ApneaCfg(**{k: rr["apnea"][k] for k in
                               ("window_s", "amplitude_ratio_below", "min_duration_s")}),
            fallback_roi_frac=d["fallback_roi_frac"],
            min_valid_depth_units=int(d.get("min_valid_depth_units", 5)),
            min_valid_pixels=int(d.get("min_valid_pixels", 30)),
            motion_mm=float(d.get("motion_mm", 20.0)),
            motion_max_hold_s=float(d.get("motion_max_hold_s", 4.0)),
            body_band_mm=(None if d.get("body_band_mm", 150.0) is None else float(d.get("body_band_mm", 150.0))),
            # the chest is always measured; the abdomen is added unless regions leaves it out
            regions=("chest", "abdomen") if "abdomen" in d.get("regions", ["chest", "abdomen"]) else ("chest",),
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

    def filled(self) -> bool:
        """True when the window holds at least min_fill_s of signal."""
        return len(self.sig) >= 4 and self.ts[-1] - self.ts[0] >= self.cfg.min_fill_s

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


def _highpass(sig: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """
    Remove slow posture drift (settling into the chair, sliding down) before the spectrum.
    A straight-line detrend cannot follow a curved drift, and what it leaves behind sits at
    the bottom of the breathing band, where it out-powers the breaths and pulls the
    estimate towards 4-6 /min. A zero-phase 2nd-order high-pass removes it while passing
    breathing at the cutoff and above (0.08 Hz = 4.8 /min).
    """
    x = np.asarray(sig, dtype=np.float64)
    if len(x) < 16 or not 0 < cutoff_hz < fs / 2:
        return scipy.signal.detrend(x, type="linear").astype(np.float32)
    b, a = scipy.signal.butter(2, cutoff_hz / (fs / 2), btype="highpass")
    return scipy.signal.filtfilt(b, a, x - x.mean()).astype(np.float32)


def _smooth(sig: np.ndarray, win_samples: int) -> np.ndarray:
    """Moving-average smoother used to attenuate cardiac micro-vibrations."""
    win_samples = max(3, int(win_samples))
    return np.convolve(sig, np.ones(win_samples, dtype=np.float32) / win_samples, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# Spectral estimate + quality score
# ─────────────────────────────────────────────────────────────────────────────

def _band_psd(sig: np.ndarray, fs: float, band_hz: tuple[float, float], nfft: int,
              segment_s: float = 10.0) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Welch PSD restricted to band_hz: (freqs, psd), or (None, None) when the band is empty."""
    nperseg = min(len(sig), max(32, int(fs * segment_s)))
    freqs, psd = scipy.signal.welch(sig, fs=fs, window="hann", nperseg=nperseg, nfft=nfft)
    idx = np.where((freqs >= band_hz[0]) & (freqs <= band_hz[1]))[0]
    if len(idx) < 3 or float(np.sum(psd[idx])) <= 0:
        return None, None
    return freqs[idx], psd[idx]


def _spectral_rate_bpm(
    sig: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    quality: _QualityCfg,
    nfft: int,
    segment_s: float = 10.0,
) -> tuple[Optional[float], float, dict]:
    """
    Welch PSD -> strongest peak inside band_hz.
    Returns (rate_bpm, peak_power_frac, debug). rate_bpm is None if the band is empty
    after filtering. A 10 s segment holds under two breaths at 10 /min, which blurs the
    peak; breathing uses 20 s (signal.yaml rr.depth.segment_s).
    """
    sub_freqs, sub_psd = _band_psd(sig, fs, band_hz, nfft, segment_s)
    if sub_freqs is None:
        return None, 0.0, {}
    return _peak_stats(sub_freqs, sub_psd, quality)


def _peak_stats(sub_freqs: np.ndarray, sub_psd: np.ndarray,
                quality: _QualityCfg) -> tuple[Optional[float], float, dict]:
    """Strongest peak of an in-band spectrum: (rate_bpm, peak_power_frac, debug)."""
    total_power = float(np.sum(sub_psd))
    if total_power <= 0:
        return None, 0.0, {}

    peak_i = int(np.argmax(sub_psd))
    peak_freq = float(sub_freqs[peak_i])

    half = quality.peak_half_width_hz
    peak_mask = (sub_freqs >= peak_freq - half) & (sub_freqs <= peak_freq + half)
    peak_power = float(np.sum(sub_psd[peak_mask]))
    frac = peak_power / total_power

    # Second-tallest separate peak, relative to the tallest. Near 1 means two rhythms of
    # similar strength (shallow breathing mixed with talking or typing): the "rate" would be
    # a coin toss between them.
    pk, _ = scipy.signal.find_peaks(sub_psd)
    far = [i for i in pk if abs(sub_freqs[i] - peak_freq) > 2 * half]
    second = float(max((sub_psd[i] for i in far), default=0.0) / sub_psd[peak_i])

    return peak_freq * 60.0, frac, {"peak_freq_hz": peak_freq, "band_power": total_power,
                                    "second_peak_ratio": round(second, 3)}


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

def _mean_from_depth_roi(depth_crop: np.ndarray, min_valid: int, min_pixels: int,
                         body_band: Optional[float] = None) -> Optional[float]:
    """
    Mean of valid depth pixels in a crop. Returns None when the crop is unusable.

    With body_band set, only pixels within +-body_band of the crop's median depth count.
    The chest box usually catches some wall or chair behind the shoulders, metres
    further away; when the box shifts by a pixel the share of those pixels changes and
    drags the mean by far more than a breath moves the chest. The median sits on the
    body, so the band keeps the chest and drops the background.
    """
    if depth_crop is None or depth_crop.size == 0:
        return None
    valid = depth_crop[depth_crop > min_valid]
    if valid.size < min_pixels:
        return None
    if body_band is not None:
        med = float(np.median(valid))
        body = valid[np.abs(valid.astype(np.float32) - med) <= body_band]
        if body.size >= min_pixels:
            valid = body
    return float(np.mean(valid))


def _fallback_depth_roi(depth: np.ndarray, roi_frac: dict) -> np.ndarray:
    """Fixed sternal box used when the scene layer provides no pose-based ROI."""
    h, w = depth.shape[:2]
    x1, x2 = int(w * roi_frac["x"][0]), int(w * roi_frac["x"][1])
    y1, y2 = int(h * roi_frac["y"][0]), int(h * roi_frac["y"][1])
    return depth[y1:y2, x1:x2]


class _DepthRegion:
    """
    One breathing surface seen by the depth camera (chest or abdomen): its own rolling
    window, motion bridge and dropout handling, so a hand resting on the belly or a
    blanket shadow over the chest only affects that region.
    """

    def __init__(self, name: str, cfg: RRConfig, fs: float):
        self.name, self.cfg = name, cfg
        self.path = _Path(cfg=cfg.depth, fs_nominal=fs, source=f"rr.depth.{name}")
        self.motion = MotionGate(fs, abs_thresh=cfg.motion_mm)
        self.bridge = MotionBridge(self.motion, mode="add", max_hold_s=cfg.motion_max_hold_s)
        self.last_raw: Optional[float] = None
        self.dropout_since: Optional[float] = None

    def sample(self, crop: Optional[np.ndarray], now: float) -> bool:
        """Add one frame. Returns True when the window was dropped (long dropout or long
        movement), so the caller can reset state tied to it."""
        mean_z = None if crop is None else _mean_from_depth_roi(
            crop, self.cfg.min_valid_depth_units, self.cfg.min_valid_pixels, self.cfg.body_band_mm)
        if mean_z is None:
            # brief dropout (a few frames of depth holes): repeat the last raw sample. A long
            # one (chest covered, out of view) must not become a flat line, which the apnoea
            # detector would read as a breath-hold: drop the window and wait for signal.
            if self.dropout_since is None:
                self.dropout_since = now
            if now - self.dropout_since > self.cfg.dropout_hold_s:
                if self.path.sig:
                    self.path.reset()
                    self.bridge.reset()
                    return True
                return False
            mean_z = self.last_raw
            if mean_z is None:
                return False
        else:
            self.dropout_since = None
        self.last_raw = mean_z
        items, restart = self.bridge.feed(mean_z, now)
        if restart:                                    # moving for longer than motion_max_hold_s
            self.path.reset()
            return True
        for t_item, value in items:                    # emitted ~0.5 s late (see MotionBridge)
            self.path.append(value, t_item)
        return False


def _fuse_spectra(spectra: list[tuple[np.ndarray, np.ndarray, float]],
                  q: _QualityCfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine the in-band spectra of several regions into one.

    Each spectrum is normalised to unit band power (a deep-breathing belly must not drown
    a cleaner chest just by moving more millimetres) and weighted by its peak_power_frac
    squared, so a clean region counts several times more than a noisy one. Breathing is
    the one rhythm all regions share, so its peak adds up while each region's own noise
    (blanket folds, a hand, depth speckle) does not. Regions below peak_power_frac_min
    are left out unless none passes, then the cleanest one is used alone.
    Spectra are interpolated onto the first (chest) frequency grid.
    """
    good = [s for s in spectra if s[2] >= q.peak_power_frac_min] or [max(spectra, key=lambda s: s[2])]
    ref_f = good[0][0]
    acc = np.zeros_like(ref_f, dtype=np.float64)
    for f, p, frac in good:
        p = np.interp(ref_f, f, p / float(np.sum(p)), left=0.0, right=0.0)
        acc += (frac ** 2) * p
    return ref_f, acc


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
        self._regions = {name: _DepthRegion(name, cfg, depth_fs) for name in cfg.regions}
        self._chest = self._regions["chest"]
        self._thermal = _Path(cfg=cfg.thermal, fs_nominal=thermal_fs, source="rr.thermal")
        self._apnea_state = _ApneaState()
        self._depth_last_estimate = 0.0
        self._last_motion_report = 0.0
        self._last_thermal_ts: Optional[float] = None
        self._thermal_src: Optional[str] = None     # "nostril" | "warmest_patch"

    # ---- per-tick entry point ------------------------------------------------

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> dict:
        """Append samples, then compute each path on its own schedule."""
        now = bundle.ts_mono
        self._sample_depth(bundle, roi, now)
        self._sample_thermal(bundle, roi, now)

        out: dict[str, Measurement] = {}
        filled = [r for r in self._regions.values() if r.path.filled()]
        if roi is not None and roi.chest_depth is None and not filled \
                and now - self._last_motion_report >= self.cfg.depth.update_every_s:
            self._last_motion_report = now          # nobody tracked: say so once a second
            out["rr.depth"] = Measurement(None, 0.0, time.time(), "rr.depth", 0.0,
                                          {"reason": "no_patient", "apnea": False})
            return out
        if self._chest.motion.recent(now, self.cfg.depth.min_fill_s) and not filled \
                and now - self._last_motion_report >= self.cfg.depth.update_every_s:
            # refilling after movement: say so instead of going quiet
            self._last_motion_report = now
            out["rr.depth"] = Measurement(None, 0.0, time.time(), "rr.depth", 0.0,
                                          {"reason": "motion", "apnea": False})
        if filled and now - self._depth_last_estimate >= self.cfg.depth.update_every_s:
            m = self._estimate_depth(filled)
            if m is not None:
                out[m.source] = m
                self._depth_last_estimate = now
        if self._thermal.ready(now) and self._thermal.fs_nominal > 0:
            m = self._estimate(self._thermal, source="rr.thermal", with_apnea=False)
            if m is not None:
                out[m.source] = m
                self._thermal.last_estimate_ts = now
        return out

    # ---- sampling ------------------------------------------------------------

    def _sample_depth(self, bundle: FrameBundle, roi: Optional[ROIBundle], now: float) -> None:
        # Pose-based crops from the scene layer. roi=None means there is no pose system at
        # all, and only then is a fixed chest box used. When the scene layer runs but finds
        # nobody (crops are None), nothing is measured: a fixed box over an empty chair
        # reads fans, curtains and depth noise as "breathing". The abdomen has no fixed box.
        if roi is None:
            chest = _fallback_depth_roi(bundle.depth, self.cfg.fallback_roi_frac)
        elif roi.chest_depth is not None and roi.chest_depth.size > 0:
            chest = roi.chest_depth
        else:
            chest = None
        if self._chest.sample(chest, now):
            self._apnea_state = _ApneaState()          # the window the apnoea timer refers to is gone
        abd = self._regions.get("abdomen")
        if abd is not None:
            crop = getattr(roi, "abdomen_depth", None) if roi is not None else None
            abd.sample(crop if crop is not None and crop.size > 0 else None, now)

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

    def _estimate_depth(self, regions: list) -> Optional[Measurement]:
        """
        rr.depth from every region with a full window. Each region's trace is filtered and
        turned into a spectrum exactly like the single-chest path; the spectra are merged by
        _fuse_spectra and the rate, quality and second-peak check read off the merged one.
        With the abdomen out of view this is the chest-only estimate, unchanged.
        Apnoea needs every measured region to have gone still, so a hand resting on one
        surface cannot fake a breath-hold.
        """
        t0 = time.perf_counter()
        c, q = self.cfg.depth, self.cfg.quality
        spectra, per = [], {}
        apnea_trig, window_s, fs_out, last_ts = True, 0.0, None, 0.0
        for r in regions:
            sig, ts = r.path.snapshot()
            fs = _effective_fs(ts, r.path.fs_nominal)
            apnea_trig = apnea_trig and _is_apnea_now(sig, ts, fs, self.cfg.apnea)
            x = _highpass(sig, fs, c.highpass_hz) if c.highpass_hz else _detrend(sig, c.detrend)
            x = _smooth(x, int(c.smooth_s * fs))
            f, p = _band_psd(x, fs, c.band_hz, c.nfft, c.segment_s)
            window_s, last_ts = max(window_s, float(ts[-1] - ts[0])), max(last_ts, float(ts[-1]))
            fs_out = fs if fs_out is None else fs_out
            if f is None:
                continue
            rate_r, frac_r, _ = _peak_stats(f, p, q)
            spectra.append((f, p, frac_r))
            per[r.name] = {"rate_bpm": None if rate_r is None else round(rate_r, 2),
                           "peak_power_frac": round(frac_r, 3)}
        apnea_now, apnea_dur = self._apnea_state.update(apnea_trig, last_ts, self.cfg.apnea.min_duration_s)

        rate_bpm, frac, dbg, used = None, 0.0, {}, []
        if spectra:
            f, p = _fuse_spectra(spectra, q)
            rate_bpm, frac, dbg = _peak_stats(f, p, q)
            used = [name for name, s in zip(per, spectra)
                    if s[2] >= q.peak_power_frac_min] or [max(per, key=lambda n: per[n]["peak_power_frac"])]
        sqi = _sqi_from_frac(frac, q)
        second = dbg.get("second_peak_ratio")
        if second is not None:              # two rhythms of similar strength: see _estimate
            sqi *= float(np.clip((1.0 - second) / 0.5, 0.3, 1.0))

        value: Optional[float] = rate_bpm
        if rate_bpm is None or frac < q.peak_power_frac_min:
            value = None
        if apnea_now:
            value, sqi = None, 1.0          # confident that NO breathing is occurring

        meta = {"fs_hz": round(fs_out or 0.0, 3), "peak_power_frac": round(frac, 3),
                "window_s": round(window_s, 2), **dbg,
                "apnea": bool(apnea_now), "apnea_duration_s": round(apnea_dur, 2),
                "regions": per, "regions_used": used}
        return Measurement(value=value, sqi=float(np.clip(sqi, 0.0, 1.0)), ts=time.time(),
                           source="rr.depth", latency_ms=(time.perf_counter() - t0) * 1000.0, meta=meta)

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

        detrended = (_highpass(sig, fs, path.cfg.highpass_hz) if path.cfg.highpass_hz
                     else _detrend(sig, path.cfg.detrend))
        smoothed  = _smooth(detrended, int(path.cfg.smooth_s * fs))

        rate_bpm, frac, dbg = _spectral_rate_bpm(
            smoothed, fs, path.cfg.band_hz, self.cfg.quality, path.cfg.nfft, path.cfg.segment_s
        )
        sqi = _sqi_from_frac(frac, self.cfg.quality)
        # A second rhythm almost as strong as the chosen one (shallow breathing mixed with
        # talking or typing) makes the rate uncertain: quality falls from full at a ratio of
        # 0.5 to 30 % at 0.9+, so fusion and the signal bars reflect the doubt. On the paced
        # recording this kept every reading and cut the error of the ones that stay fused.
        second = dbg.get("second_peak_ratio")
        if second is not None:
            sqi *= float(np.clip((1.0 - second) / 0.5, 0.3, 1.0))

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