"""
learned.py — the learned RR / HR estimators (EXPERIMENTAL, display + logging only by default)

Three networks, each fed exactly the input it was trained on:

  rr.cnn_lbrd       Resp1DCNN, models/public/rr_cnn1d_lbrdic.pth   (baseline_rr.ipynb)
                    Trained on LBRD-IC: mean GREEN of the WHOLE frame, first 250 frames of a
                    25 fps clip (= 10 s), z-scored. Live: last 10 s of whole-frame mean green,
                    resampled onto a 25 Hz grid (250 samples), z-scored.
  rr.cnn_d435       Resp1DCNN, models/finetuned/rr_cnn1d_d435_finetuned.pth
                    Fine-tuned on your D435 clips. The input recipe is taken from
                    train_finetune_rr_realsense.py: 300-frame (10 s @ 30 fps) chest-depth window,
                    linear detrend, z-score. NOTE: that script defines RespiratoryNet1D, but the
                    weight file has Resp1DCNN's layers, so it came from another version of the
                    script; the input recipe is an assumption until the comparison confirms it.
  hr.efficientphys  EfficientPhys (TSM), models/public/hr_efficientphys_ubfc.pth (baseline_hr.ipynb)
                    Trained on UBFC-rPPG: 72x72 BGR frames, normalised frame difference
                    (f[t+1]-f[t]) / (f[t+1]+f[t]), 72-frame chunks -> BVP. Live: the face crop,
                    chunks every 36 frames (50 % overlap, as in the notebook's evaluation), BVP
                    overlap-added, rate = Welch peak (nperseg 256, nfft 4096) in the HR band.
                    On your 7 held-out UBFC subjects it estimated 99-100 BPM for everyone.

Every Measurement carries meta["experimental"] = True and the network's unfiltered output
(raw_output / rate_unfiltered), so the comparison tools can score it even when the quality
gate hides the value. They join the fused RR/HR (and therefore alarms) only when
signal.yaml -> learned.fuse is true.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from src.modules.rr import _fallback_depth_roi, _mean_from_depth_roi
from src.types import FrameBundle, Measurement, ROIBundle


# ─────────────────────────────────────────────────────────────────────────────
# Architectures (verbatim from the training notebooks)
# ─────────────────────────────────────────────────────────────────────────────

class Resp1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.regressor(self.features(x).squeeze(-1))


class _AttentionMask(nn.Module):
    def forward(self, x):
        return 2.0 * x / (torch.sum(x, dim=(2, 3), keepdim=True) + 1e-7)


class _TSM(nn.Module):
    def __init__(self, n_segment=72, fold_div=3):
        super().__init__()
        self.n_segment, self.fold_div = n_segment, fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    def __init__(self, in_channels=3, frame_depth=72):
        super().__init__()
        self.tsm1 = _TSM(frame_depth)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.attn_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.attn_mask = _AttentionMask()
        self.tsm2 = _TSM(frame_depth)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(), nn.AvgPool2d(2))
        self.tsm3 = _TSM(frame_depth)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh())
        self.tsm4 = _TSM(frame_depth)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh(), nn.AvgPool2d(2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x = self.conv1(self.tsm1(x))
        x = x * self.attn_mask(self.attn_conv(x))
        x = self.conv2(self.tsm2(x))
        x = self.conv3(self.tsm3(x))
        x = self.conv4(self.tsm4(x))
        return self.head(x).view(b, t)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RRNet:
    weights: Path
    n_samples: int                 # network input length
    window_s: float = 10.0         # seconds of signal resampled onto n_samples
    update_every_s: float = 1.0


@dataclass
class _HRNet:
    weights: Path = Path("models/public/hr_efficientphys_ubfc.pth")
    window_s: float = 20.0         # BVP seconds used for one rate estimate
    min_fill_s: float = 10.0
    update_every_s: float = 1.0
    chunk_len: int = 72
    stride: int = 36
    img_size: int = 72


@dataclass
class LearnedConfig:
    fuse: bool = False             # True -> learned estimates join rr/hr fusion (and alarms)
    device: str = "auto"
    rr_lbrd: _RRNet = field(default_factory=lambda: _RRNet(Path("models/public/rr_cnn1d_lbrdic.pth"), 250))
    rr_d435: _RRNet = field(default_factory=lambda: _RRNet(
        Path("models/finetuned/rr_cnn1d_d435_finetuned.pth"), 300))
    hr: _HRNet = field(default_factory=_HRNet)
    rr_band_hz: Tuple[float, float] = (0.067, 0.70)
    rr_range_bpm: Tuple[float, float] = (4.0, 42.0)
    hr_band_hz: Tuple[float, float] = (0.67, 3.0)
    hr_range_bpm: Tuple[float, float] = (40.0, 180.0)
    rr_quality: Tuple[float, float, float] = (0.30, 0.60, 0.03)   # min, good, half-width Hz
    hr_quality: Tuple[float, float, float] = (0.25, 0.50, 0.05)
    depth_fallback_roi_frac: dict = field(default_factory=lambda: {"x": [0.44, 0.65], "y": [0.54, 0.82]})
    min_valid_depth_units: int = 5
    min_valid_pixels: int = 30

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "LearnedConfig":
        """Build from the whole parsed signal.yaml (reads learned:, plus rr/hr bands)."""
        cfg = cfg or {}
        c = cls()
        ln = cfg.get("learned") or {}
        c.fuse = bool(ln.get("fuse", c.fuse))
        c.device = str(ln.get("device", c.device))
        for key, net in (("rr_lbrd", c.rr_lbrd), ("rr_d435", c.rr_d435), ("hr_efficientphys", c.hr)):
            for k, v in (ln.get(key) or {}).items():
                if hasattr(net, k) and v is not None:
                    setattr(net, k, Path(v) if k == "weights" else type(getattr(net, k))(v))
        rr, hr = cfg.get("rr") or {}, cfg.get("hr") or {}
        d = rr.get("depth") or {}
        if d.get("band_hz"):
            c.rr_band_hz = tuple(d["band_hz"])
        if rr.get("reported_range_bpm"):
            c.rr_range_bpm = tuple(rr["reported_range_bpm"])
        if (hr.get("pos") or {}).get("band_hz"):
            c.hr_band_hz = tuple(hr["pos"]["band_hz"])
        if hr.get("reported_range_bpm"):
            c.hr_range_bpm = tuple(hr["reported_range_bpm"])
        for attr, q in (("rr_quality", rr.get("quality")), ("hr_quality", hr.get("quality"))):
            if q:
                lo, good, hw = getattr(c, attr)
                setattr(c, attr, (float(q.get("peak_power_frac_min", lo)),
                                  float(q.get("peak_power_frac_good", good)),
                                  float(q.get("peak_half_width_hz", hw))))
        if d.get("fallback_roi_frac"):
            c.depth_fallback_roi_frac = d["fallback_roi_frac"]
        return c


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _device(name: str) -> torch.device:
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def _load(net: nn.Module, path: Path, device: torch.device) -> nn.Module:
    if not Path(path).exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    return net.to(device).eval()


def _peak(sig: np.ndarray, fs: float, band: Tuple[float, float], half_width: float,
          nperseg: int, nfft: int) -> Tuple[Optional[float], float]:
    """Strongest Welch peak in band -> (rate_bpm, fraction of band power around the peak)."""
    freqs, psd = scipy.signal.welch(sig, fs=fs, window="hann", nperseg=min(len(sig), nperseg), nfft=nfft)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if len(idx) < 3 or float(np.sum(psd[idx])) <= 0:
        return None, 0.0
    f, p = freqs[idx], psd[idx]
    pk = float(f[int(np.argmax(p))])
    frac = float(np.sum(p[(f >= pk - half_width) & (f <= pk + half_width)]) / np.sum(p))
    return pk * 60.0, frac


def _sqi(frac: float, lo: float, good: float) -> float:
    return float(np.clip((frac - lo) / max(1e-6, good - lo), 0.0, 1.0))


def _meas(source: str, value: Optional[float], sqi: float, t0: float, meta: dict) -> Measurement:
    return Measurement(value=value, sqi=float(sqi), ts=time.time(), source=source,
                       latency_ms=(time.perf_counter() - t0) * 1000.0,
                       meta={"experimental": True, **meta})


class _Series:
    """Timestamped 1-D buffer resampled onto a fixed grid ending at the newest sample."""

    def __init__(self, net: _RRNet, fs_hint: float = 30.0):
        self.net = net
        self.v: Deque[float] = deque(maxlen=int(net.window_s * fs_hint * 3) + 32)
        self.t: Deque[float] = deque(maxlen=self.v.maxlen)
        self.last_emit: Optional[float] = None

    def add(self, value: float, ts: float) -> None:
        self.v.append(float(value))
        self.t.append(float(ts))

    def ready(self, now: float) -> bool:
        if len(self.t) < 4 or self.t[-1] - self.t[0] < self.net.window_s - 1e-6:
            return False
        return self.last_emit is None or now - self.last_emit >= self.net.update_every_s - 1e-6

    def window(self) -> Tuple[np.ndarray, float]:
        """Last window resampled to n_samples points spaced window_s / n_samples apart."""
        t, v = np.asarray(self.t), np.asarray(self.v, dtype=np.float64)
        n, fs = self.net.n_samples, self.net.n_samples / self.net.window_s
        grid = t[-1] - (n - 1 - np.arange(n)) / fs
        return np.interp(grid, t, v), fs


# ─────────────────────────────────────────────────────────────────────────────
# Learned RR
# ─────────────────────────────────────────────────────────────────────────────

class LearnedRRModule:
    """Emits rr.cnn_lbrd (whole-frame green) and/or rr.cnn_d435 (chest depth)."""

    def __init__(self, cfg: LearnedConfig, use_lbrd: bool = True, use_d435: bool = True):
        self.cfg = cfg
        self.device = _device(cfg.device)
        self._paths: Dict[str, Tuple[nn.Module, _Series, bool]] = {}   # source -> (net, series, detrend)
        if use_lbrd:
            self._paths["rr.cnn_lbrd"] = (_load(Resp1DCNN(), cfg.rr_lbrd.weights, self.device),
                                          _Series(cfg.rr_lbrd), False)
        if use_d435:
            self._paths["rr.cnn_d435"] = (_load(Resp1DCNN(), cfg.rr_d435.weights, self.device),
                                          _Series(cfg.rr_d435), True)

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(self._paths)

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> Dict[str, Measurement]:
        now = bundle.ts_mono
        if "rr.cnn_lbrd" in self._paths:
            self._paths["rr.cnn_lbrd"][1].add(float(np.mean(bundle.rgb[:, :, 1])), now)
        if "rr.cnn_d435" in self._paths:
            series = self._paths["rr.cnn_d435"][1]
            crop = (roi.chest_depth if roi is not None and roi.chest_depth is not None
                    and roi.chest_depth.size > 0
                    else _fallback_depth_roi(bundle.depth, self.cfg.depth_fallback_roi_frac))
            z = _mean_from_depth_roi(crop, self.cfg.min_valid_depth_units, self.cfg.min_valid_pixels)
            if z is not None:
                series.add(z, now)
            elif series.v:
                series.add(series.v[-1], now)          # one-sample hold, as in rr.depth

        out: Dict[str, Measurement] = {}
        for source, (net, series, detrend) in self._paths.items():
            if series.ready(now):
                series.last_emit = now
                out[source] = self._estimate(source, net, series, detrend)
        return out

    def _estimate(self, source: str, net: nn.Module, series: _Series, detrend: bool) -> Measurement:
        t0 = time.perf_counter()
        x, fs = series.window()                                   # float64
        scale = max(1.0, float(np.max(np.abs(x))))
        if detrend:
            x = scipy.signal.detrend(x, type="linear")
        lo, good, hw = self.cfg.rr_quality
        std = float(np.std(x))
        if std < 1e-6 * scale:                                    # constant input (relative to its size)
            return _meas(source, None, 0.0, t0, {"raw_output": None, "reason": "flat_signal",
                                                 "input_peak_bpm": None, "fs_hz": fs})
        z = (x - np.mean(x)) / std
        with torch.no_grad():
            raw = float(net(torch.from_numpy(z.astype(np.float32))[None, None].to(self.device)).item())
        peak_bpm, frac = _peak(z, fs, self.cfg.rr_band_hz, hw, nperseg=len(z), nfft=16384)
        rng = self.cfg.rr_range_bpm
        in_range = rng[0] <= raw <= rng[1]
        return _meas(source, raw if in_range else None, _sqi(frac, lo, good), t0, {
            "raw_output": raw, "reason": None if in_range else "out_of_range",
            "input_peak_bpm": peak_bpm, "input_peak_frac": round(frac, 3), "fs_hz": fs,
            "model": "Resp1DCNN"})


# ─────────────────────────────────────────────────────────────────────────────
# Learned HR
# ─────────────────────────────────────────────────────────────────────────────

class LearnedHRModule:
    """Emits hr.efficientphys from the face crop.

    background=True runs each 72-frame chunk on a worker thread (one chunk takes ~0.7 s on a
    laptop CPU), so the camera frame loop never waits; a chunk that comes due while the
    previous one is still running is skipped rather than queued."""
    SOURCE = "hr.efficientphys"

    def __init__(self, cfg: LearnedConfig, background: bool = False):
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="efficientphys") if background else None
        self._pending: Optional[Tuple[Future, int, list, int]] = None   # future, first idx, ts, generation
        self._gen = 0
        self.device = _device(cfg.device)
        h = cfg.hr
        self.net = _load(EfficientPhys(3, h.chunk_len), h.weights, self.device)
        self._frames: Deque[np.ndarray] = deque(maxlen=h.chunk_len + 1)
        self._frame_ts: Deque[float] = deque(maxlen=h.chunk_len + 1)
        self._n = 0                       # frames appended since the last reset
        self._last_chunk_end = 0          # self._n at the last chunk
        self._bvp: Dict[int, list] = {}   # diff index -> [sum, count, ts]
        self._last_emit: Optional[float] = None
        self.last_chunk_bvp: Optional[np.ndarray] = None

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    def _reset(self) -> None:
        self._gen += 1                    # results of an in-flight chunk are dropped
        self._pending = None
        self._frames.clear()
        self._frame_ts.clear()
        self._n = self._last_chunk_end = 0
        self._bvp.clear()

    def process(self, bundle: FrameBundle, roi: Optional[ROIBundle] = None) -> Dict[str, Measurement]:
        now = bundle.ts_mono
        h = self.cfg.hr
        face = roi.face_roi if roi is not None else None
        if face is None or face.size == 0 or min(face.shape[:2]) < 8:
            had = self._n > 0
            self._reset()
            if had or self._last_emit is None or now - self._last_emit >= h.update_every_s:
                self._last_emit = now
                return {self.SOURCE: _meas(self.SOURCE, None, 0.0, time.perf_counter(),
                                           {"reason": "no_face", "rate_unfiltered": None})}
            return {}

        self._collect()
        self._frames.append(cv2.resize(face, (h.img_size, h.img_size)).astype(np.float32))
        self._frame_ts.append(now)
        self._n += 1
        if len(self._frames) == h.chunk_len + 1 and (
                self._last_chunk_end == 0 or self._n - self._last_chunk_end >= h.stride):
            self._run_chunk()

        span = self._bvp_span()
        if span < h.min_fill_s or (self._last_emit is not None and now - self._last_emit < h.update_every_s):
            return {}
        self._last_emit = now
        return {self.SOURCE: self._estimate()}

    def _infer(self, fr: np.ndarray) -> np.ndarray:
        diff = (fr[1:] - fr[:-1]) / (fr[1:] + fr[:-1] + 1e-6)
        x = torch.from_numpy(np.transpose(diff, (3, 0, 1, 2)).copy()).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            return self.net(x).squeeze(0).cpu().numpy()

    def _run_chunk(self) -> None:
        fr = np.stack(self._frames)                                 # (73, 72, 72, 3) BGR
        first = self._n - len(self._frames)                         # diff index of the chunk start
        ts = list(self._frame_ts)[1:]                               # diff j stamped at frame j+1
        self._last_chunk_end = self._n
        if self._pool is None:
            self._merge(self._infer(fr), first, ts)
        elif self._pending is None:
            self._pending = (self._pool.submit(self._infer, fr), first, ts, self._gen)

    def _collect(self) -> None:
        """Merge a finished background chunk (no-op inline or while it is still running)."""
        if self._pending is None or not self._pending[0].done():
            return
        fut, first, ts, gen = self._pending
        self._pending = None
        if gen == self._gen and fut.exception() is None:
            self._merge(fut.result(), first, ts)

    def drain(self) -> None:
        """Wait for an in-flight chunk and merge it (tests, shutdown)."""
        if self._pending is not None:
            self._pending[0].result()
            self._collect()

    def _merge(self, bvp: np.ndarray, first: int, ts: list) -> None:
        self.last_chunk_bvp = bvp
        for k, (v, t) in enumerate(zip(bvp, ts)):
            cell = self._bvp.setdefault(first + k, [0.0, 0, t])
            cell[0] += float(v)
            cell[1] += 1
        keep = int(self.cfg.hr.window_s * 40) + self.cfg.hr.chunk_len   # generous at <= 40 fps
        for j in sorted(self._bvp)[:-keep]:
            del self._bvp[j]

    def _bvp_span(self) -> float:
        if len(self._bvp) < 2:
            return 0.0
        ks = sorted(self._bvp)
        return self._bvp[ks[-1]][2] - self._bvp[ks[0]][2]

    def _estimate(self) -> Measurement:
        t0 = time.perf_counter()
        h = self.cfg.hr
        ks = sorted(self._bvp)
        ts = np.array([self._bvp[k][2] for k in ks])
        sig = np.array([self._bvp[k][0] / self._bvp[k][1] for k in ks], dtype=np.float64)
        keep = ts >= ts[-1] - h.window_s
        ts, sig = ts[keep], sig[keep]
        fs = (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])
        lo_hz, hi_hz = self.cfg.hr_band_hz
        b, a = scipy.signal.butter(2, [lo_hz / (fs / 2), min(0.99, hi_hz / (fs / 2))], btype="bandpass")
        filt = scipy.signal.filtfilt(b, a, scipy.signal.detrend(sig))
        lo, good, hw = self.cfg.hr_quality
        rate, frac = _peak(filt, fs, self.cfg.hr_band_hz, hw, nperseg=256, nfft=4096)
        sqi = _sqi(frac, lo, good)
        rng = self.cfg.hr_range_bpm
        ok = rate is not None and frac >= lo and rng[0] <= rate <= rng[1]
        reason = None if ok else ("low_quality" if rate is not None and frac < lo else "out_of_range")
        return _meas(self.SOURCE, rate if ok else None, sqi, t0, {
            "rate_unfiltered": rate, "peak_power_frac": round(frac, 3), "reason": reason,
            "fs_hz": round(fs, 2), "window_s": round(float(ts[-1] - ts[0]), 2), "model": "EfficientPhys"})


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_learned_modules(signal_yaml_path: str = "config/signal.yaml"):
    """Returns ({"rr_learned": ..., "hr_learned": ...}, {path_key: error}) and the config.
    Each network loads independently: a missing file disables only that path."""
    import yaml
    try:
        with open(signal_yaml_path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raw = {}
    cfg = LearnedConfig.from_dict(raw)
    mods: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    ok = {}
    for key, net in (("rr_lbrd", cfg.rr_lbrd), ("rr_d435", cfg.rr_d435)):
        ok[key] = Path(net.weights).exists()
        if not ok[key]:
            errors[key] = f"Weights not found: {net.weights}"
    if ok["rr_lbrd"] or ok["rr_d435"]:
        try:
            mods["rr_learned"] = LearnedRRModule(cfg, use_lbrd=ok["rr_lbrd"], use_d435=ok["rr_d435"])
        except Exception as exc:  # noqa: BLE001
            errors["rr_lbrd" if ok["rr_lbrd"] else "rr_d435"] = f"{type(exc).__name__}: {exc}"
    try:
        mods["hr_learned"] = LearnedHRModule(cfg, background=True)
    except Exception as exc:  # noqa: BLE001
        errors["hr_efficientphys"] = f"{type(exc).__name__}: {exc}"
    return mods, errors
