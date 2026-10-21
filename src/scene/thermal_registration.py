"""
thermal_registration.py — align the ESP32 thermal image (62x80) with the D435 RGB image.

WHY
---
The two cameras sit a few centimetres apart with different lenses, so thermal
pixel (x, y) does not look at the same spot as RGB pixel (x, y). A homography H
(a 3x3 matrix) maps every thermal pixel to the RGB pixel that sees the same
point. With H, the nose found by pose tracking in RGB can be translated into
thermal pixels, and respiratory rate is read from the temperature right at the
nostrils (warm on exhale, cool on inhale) instead of from the warmest spot.

HOW IT IS FITTED
----------------
The operator marks the same physical point in both images at least 4 times
(e.g. a mug of hot water held at the patient's distance, one snapshot per spot).
H is solved by the normalised Direct Linear Transform (Hartley & Zisserman,
alg. 4.2) with numpy only. A homography is exact for one plane, so points should
be marked at roughly the patient's distance from the camera.

Quality is reported as the RMS reprojection error in RGB pixels. With exactly 4
pairs the fit is exact (error 0), so 6+ pairs are needed for a meaningful check.

STORAGE
-------
config/sessions/thermal_<patient_id>.json (machine-specific, git-ignored), or
camera.yaml -> waveshare_thermal.homography_matrix (3x3) as a fixed fallback.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

THERMAL_SHAPE = (62, 80)          # (h, w) of the Waveshare ESP32-S3 sensor
MAX_ERROR_PX = 40.0               # reject fits worse than this (RGB pixels)

Box = Tuple[float, float, float, float]          # x0, y0, x1, y1
IntBox = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Homography fitting
# --------------------------------------------------------------------------- #
def _normaliser(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2.0) / d if d > 1e-9 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _polygon_area(pts: np.ndarray) -> float:
    """Area of the convex hull (monotone chain); 0 for collinear points."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    return 0.5 * abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1]
                         - hull[(i + 1) % len(hull)][0] * hull[i][1] for i in range(len(hull))))


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares H with dst ~ H @ src (normalised DLT). Raises ValueError if degenerate."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("At least 4 matching point pairs are needed.")
    Ts, Td = _normaliser(src), _normaliser(dst)
    s = _apply(Ts, src)
    d = _apply(Td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows)
    _, sv, vt = np.linalg.svd(A)
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError("The points do not pin down a unique alignment - spread them out.")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Alignment is numerically unstable - spread the points out.")
    return H / H[2, 2]


# --------------------------------------------------------------------------- #
# Registration object
# --------------------------------------------------------------------------- #
@dataclass
class ThermalRegistration:
    H: np.ndarray                                   # thermal (x, y) -> rgb (x, y)
    error_px: Optional[float] = None                # RMS reprojection error, RGB px
    n_pairs: int = 0
    pairs: List[dict] = field(default_factory=list)
    source: str = "session"                         # session | camera.yaml
    ts: float = field(default_factory=time.time)
    thermal_shape: Tuple[int, int] = THERMAL_SHAPE
    rgb_shape: Tuple[int, int] = (480, 640)

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=np.float64).reshape(3, 3)
        self._Hinv = np.linalg.inv(self.H)

    # mapping -------------------------------------------------------------- #
    def to_rgb(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        return _apply(self.H, np.asarray(pts, dtype=np.float64).reshape(-1, 2))

    def to_thermal(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        return _apply(self._Hinv, np.asarray(pts, dtype=np.float64).reshape(-1, 2))

    @property
    def quality(self) -> str:
        if self.error_px is None:
            return "unchecked"
        if self.n_pairs <= 4:
            return "unchecked"          # 4 pairs always fit exactly
        if self.error_px < 10:
            return "good"
        if self.error_px < 20:
            return "fair"
        return "poor"

    # fitting -------------------------------------------------------------- #
    @classmethod
    def fit(cls, pairs: Sequence[dict], thermal_shape: Tuple[int, int] = THERMAL_SHAPE,
            rgb_shape: Tuple[int, int] = (480, 640),
            max_error_px: float = MAX_ERROR_PX) -> "ThermalRegistration":
        """pairs: [{"rgb": [x, y], "thermal": [x, y]}, ...] (>= 4)."""
        if len(pairs) < 4:
            raise ValueError(f"Mark at least 4 point pairs (have {len(pairs)}).")
        try:
            rgb = np.array([[float(p["rgb"][0]), float(p["rgb"][1])] for p in pairs])
            th = np.array([[float(p["thermal"][0]), float(p["thermal"][1])] for p in pairs])
        except (KeyError, TypeError, IndexError, ValueError):
            raise ValueError('Each pair needs "rgb": [x, y] and "thermal": [x, y].') from None
        if not (np.isfinite(rgb).all() and np.isfinite(th).all()):
            raise ValueError("Point coordinates must be finite numbers.")
        th_h, th_w = thermal_shape
        rgb_h, rgb_w = rgb_shape
        if (th[:, 0] < 0).any() or (th[:, 0] > th_w).any() or (th[:, 1] < 0).any() or (th[:, 1] > th_h).any():
            raise ValueError(f"A thermal point lies outside the {th_w}x{th_h} thermal image.")
        if (rgb[:, 0] < 0).any() or (rgb[:, 0] > rgb_w).any() or (rgb[:, 1] < 0).any() or (rgb[:, 1] > rgb_h).any():
            raise ValueError(f"A colour point lies outside the {rgb_w}x{rgb_h} image.")
        if _polygon_area(th) < 0.02 * th_w * th_h or _polygon_area(rgb) < 0.02 * rgb_w * rgb_h:
            raise ValueError("The points are bunched together or in a line - spread them across the view.")

        H = fit_homography(th, rgb)
        # every thermal pixel must land in front of the camera (positive w)
        corners = np.array([[0, 0], [th_w, 0], [th_w, th_h], [0, th_h]], dtype=np.float64)
        w = (np.c_[corners, np.ones(4)] @ H.T)[:, 2]
        if not ((w > 0).all() or (w < 0).all()):
            raise ValueError("These points fold the image over itself - check that each pair marks "
                             "the same spot in both images.")
        err = float(np.sqrt(((_apply(H, th) - rgb) ** 2).sum(axis=1).mean()))
        if err > max_error_px:
            raise ValueError(f"The points disagree by {err:.0f} px on average - at least one pair "
                             f"probably marks different spots. Undo it and try again.")
        clean = [{"rgb": [float(a), float(b)], "thermal": [float(c), float(d)]}
                 for (a, b), (c, d) in zip(rgb, th)]
        return cls(H=H, error_px=err, n_pairs=len(pairs), pairs=clean, source="session",
                   thermal_shape=tuple(thermal_shape), rgb_shape=tuple(rgb_shape))

    # persistence ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"H": self.H.tolist(), "error_px": self.error_px, "n_pairs": self.n_pairs,
                "pairs": self.pairs, "source": self.source, "ts": self.ts,
                "thermal_shape": list(self.thermal_shape), "rgb_shape": list(self.rgb_shape),
                "quality": self.quality}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalRegistration":
        return cls(H=np.asarray(d["H"]), error_px=d.get("error_px"), n_pairs=int(d.get("n_pairs", 0)),
                   pairs=list(d.get("pairs", [])), source=d.get("source", "session"),
                   ts=float(d.get("ts", 0.0)),
                   thermal_shape=tuple(d.get("thermal_shape", THERMAL_SHAPE)),  # type: ignore[arg-type]
                   rgb_shape=tuple(d.get("rgb_shape", (480, 640))))             # type: ignore[arg-type]


def registration_path(config_dir: str | Path, patient_id: str) -> Path:
    return Path(config_dir) / "sessions" / f"thermal_{patient_id}.json"


def load_thermal_registration(config_dir: str | Path, patient_id: str) -> Optional[ThermalRegistration]:
    """Session file first, then camera.yaml homography_matrix. None if neither is usable."""
    p = registration_path(config_dir, patient_id)
    if p.exists():
        try:
            return ThermalRegistration.from_dict(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt file must not stop the monitor
            pass
    cam = Path(config_dir) / "camera.yaml"
    if yaml is not None and cam.exists():
        try:
            raw = yaml.safe_load(cam.read_text()) or {}
            m = (raw.get("waveshare_thermal") or {}).get("homography_matrix")
            if m is not None:
                return ThermalRegistration(H=np.asarray(m, dtype=np.float64), source="camera.yaml")
        except Exception:  # noqa: BLE001
            pass
    return None


# --------------------------------------------------------------------------- #
# Nostril region
# --------------------------------------------------------------------------- #
def nostril_box_rgb(kpts: Optional[np.ndarray], confs: Optional[np.ndarray],
                    conf_min: float = 0.3) -> Optional[Box]:
    """Box (RGB px) just below the nose tip covering nostrils, upper lip and mouth.

    Scaled by the eye distance (or half the ear distance) so it follows the face size.
    Returns None when the nose or a face scale cannot be found.
    """
    if kpts is None or confs is None or len(kpts) < 5:
        return None

    def ok(i: int) -> bool:
        return confs[i] >= conf_min and float(np.asarray(kpts[i]).sum()) > 0

    if not ok(0):
        return None
    nx, ny = float(kpts[0][0]), float(kpts[0][1])
    if ok(1) and ok(2):
        scale = float(np.hypot(*(np.asarray(kpts[1]) - np.asarray(kpts[2]))))
    elif ok(3) and ok(4):
        scale = 0.5 * float(np.hypot(*(np.asarray(kpts[3]) - np.asarray(kpts[4]))))
    else:
        return None
    scale = max(scale, 6.0)
    half_w = 0.45 * scale
    return (nx - half_w, ny + 0.05 * scale, nx + half_w, ny + 0.75 * scale)


def thermal_roi_from_rgb_box(reg: ThermalRegistration, box: Box,
                             min_px: int = 3) -> Optional[IntBox]:
    """Map an RGB box into thermal pixels: integer (x0, y0, x1, y1), end-exclusive,
    at least min_px x min_px, clamped to the sensor. None if it falls off the sensor."""
    th_h, th_w = reg.thermal_shape
    x0, y0, x1, y1 = box
    corners = reg.to_thermal([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if not np.isfinite(corners).all():
        return None
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
    if not (0 <= cx < th_w and 0 <= cy < th_h):
        return None
    # size in whole pixels (at least min_px), centred on the mapped centre; rounding the
    # start and adding the size keeps a 3-px ROI at exactly 3 px (floor/ceil would give 4
    # and dilute the small breathing signal with steady face temperature)
    nw = max(min_px, int(round(corners[:, 0].max() - corners[:, 0].min())))
    nh = max(min_px, int(round(corners[:, 1].max() - corners[:, 1].min())))
    ix0 = int(round(cx - nw / 2))
    iy0 = int(round(cy - nh / 2))
    ix0 = min(max(0, ix0), th_w - nw)
    iy0 = min(max(0, iy0), th_h - nh)
    ix1, iy1 = min(th_w, ix0 + nw), min(th_h, iy0 + nh)
    if ix1 - ix0 < 1 or iy1 - iy0 < 1 or ix0 < 0 or iy0 < 0:
        return None
    return (ix0, iy0, ix1, iy1)


class RoiStabilizer:
    """Keep the thermal ROI steady on the nostrils despite noisy pose keypoints.

    At 80x62 a 3x3 ROI sits on a breathing hot spot only ~1-2 px wide. Pose keypoints
    wobble by a few RGB px every frame (~0.3-0.7 thermal px), and the face-size
    estimate wobbles too. Following each raw estimate slides the window on and off
    the hot spot (and flips its size between 3 and 4 px), injecting step noise far
    larger than the breathing signal.

    So the centre and size are smoothed with an exponential moving average, which
    converges to sub-pixel accuracy because detector noise averages out, and the
    integer window moves only when the smoothed centre drifts >= move_px away from
    it (hysteresis). A jump > snap_px (the patient really moved) re-centres at once.
    """

    def __init__(self, alpha: float = 0.12, move_px: float = 0.75, snap_px: float = 3.0,
                 thermal_shape: Tuple[int, int] = THERMAL_SHAPE):
        self.alpha, self.move_px, self.snap_px = float(alpha), float(move_px), float(snap_px)
        self.th_h, self.th_w = thermal_shape
        self.reset()

    def reset(self) -> None:
        self._c: Optional[np.ndarray] = None       # smoothed centre (x, y)
        self._s: Optional[np.ndarray] = None       # smoothed size (w, h)
        self._roi: Optional[IntBox] = None

    def _place(self, c: np.ndarray, s: np.ndarray) -> IntBox:
        nw, nh = max(1, int(round(s[0]))), max(1, int(round(s[1])))
        x0 = min(max(0, int(round(c[0] - nw / 2))), self.th_w - nw)
        y0 = min(max(0, int(round(c[1] - nh / 2))), self.th_h - nh)
        return (x0, y0, x0 + nw, y0 + nh)

    def update(self, roi: Optional[IntBox]) -> Optional[IntBox]:
        if roi is None:
            self.reset()
            return None
        c = np.array([(roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0])
        s = np.array([float(roi[2] - roi[0]), float(roi[3] - roi[1])])
        if self._c is None or np.abs(c - self._c).max() > self.snap_px:
            self._c, self._s = c, s                 # first sight, or the patient moved
            self._roi = self._place(c, s)
            return self._roi
        self._c = self._c + self.alpha * (c - self._c)
        self._s = self._s + self.alpha * (s - self._s)
        held = self._roi
        target = self._place(self._c, self._s)
        hc = np.array([(held[0] + held[2]) / 2.0, (held[1] + held[3]) / 2.0])
        size_changed = (target[2] - target[0], target[3] - target[1]) != (held[2] - held[0], held[3] - held[1])
        if size_changed or np.abs(self._c - hc).max() >= self.move_px:
            self._roi = target
        return self._roi


def mean_in_roi(thermal: Optional[np.ndarray], roi: Optional[IntBox]) -> Optional[float]:
    if thermal is None or roi is None:
        return None
    x0, y0, x1, y1 = roi
    patch = thermal[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    v = float(np.mean(patch))
    return v if math.isfinite(v) else None


def warmest_patch_c(thermal: Optional[np.ndarray]) -> Optional[float]:
    """Fallback signal: mean of the warmest 2 % of pixels (face, usually)."""
    if thermal is None or thermal.size == 0:
        return None
    flat = thermal.ravel()
    k = max(1, flat.size // 50)
    return float(np.mean(np.partition(flat, -k)[-k:]))
