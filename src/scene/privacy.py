"""Privacy guard for the VisionICU scene layer.

THE CONTRACT
------------
VisionICU stores vital-sign *numbers*, never subject imagery, and never
retrains on subject video. This module is the single sanctioned chokepoint
for the rare case where a frame must touch disk (e.g. a fall-alarm thumbnail
so a nurse can verify the posture). Everything that could persist an image
goes through PrivacyGuard, which:

  1. refuses to write any frame unless debug capture is EXPLICITLY enabled
     in config (default: off);
  2. de-identifies faces before writing - blurs the face (identity) while
     leaving the body (the clinically useful part) visible;
  3. writes only into a dedicated debug directory, and prunes old captures
     by age and count so nothing accumulates silently.

HONEST LIMITATION
-----------------
This is enforcement by discipline, not by the operating system: Python
cannot stop other code from calling cv2.imwrite directly. The guarantee is
that the *pipeline's* sanctioned path never persists a clear face and never
writes at all unless debug capture was turned on. Keep all image writes
behind this class and that property holds.

FACE LOCATION
-------------
Faces are found from COCO face keypoints (nose/eyes/ears, 0-4). If those are
missing we fall back to the top of the person bbox, and if there is no bbox
either we blur the whole frame - the guard never saves an un-redacted face.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pose import PosePerson

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.scene.privacy")

_FACE_PTS = (0, 1, 2, 3, 4)  # nose, eyes, ears


@dataclass
class PrivacyPolicy:
    allow_debug_capture: bool = False          # master switch; OFF by default
    debug_dir: str = "debug_captures"
    blur_faces: bool = True                    # de-identify before any write
    blur_method: str = "pixelate"              # "pixelate" | "gaussian"
    pixelate_blocks: int = 12                  # coarser = more private
    face_pad: float = 0.6                      # grow the face box before blurring
    max_debug_files: int = 50                  # keep at most this many captures
    retain_hours: float = 24.0                 # delete captures older than this

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PrivacyPolicy":
        c = cls()
        if not d:
            return c
        for k in c.__dataclass_fields__:  # type: ignore[attr-defined]
            if k in d and d[k] is not None:
                setattr(c, k, type(getattr(c, k))(d[k]))
        return c


class PrivacyGuard:
    def __init__(self, policy: Optional[PrivacyPolicy] = None) -> None:
        self.p = policy or PrivacyPolicy()

    # -- de-identification ------------------------------------------------- #
    def _face_box(
        self,
        shape: tuple[int, int],
        keypoints: Optional[np.ndarray],
        confs: Optional[np.ndarray],
        bbox: Optional[np.ndarray],
        conf_min: float = 0.25,
    ) -> Optional[tuple[int, int, int, int]]:
        h, w = shape
        pts = []
        if keypoints is not None and confs is not None:
            for i in _FACE_PTS:
                if confs[i] >= conf_min and keypoints[i].sum() > 0:
                    pts.append((float(keypoints[i][0]), float(keypoints[i][1])))
        if len(pts) >= 2:
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = sum(ys) / len(ys)
            width = max(30.0, max(xs) - min(xs))
            half = width / 2.0 * (1.0 + self.p.face_pad)
            half_h = half * 1.3
            return _clamp(cx - half, cy - half_h * 1.1, cx + half, cy + half_h * 0.9, w, h)
        if bbox is not None:  # fallback: top of the person box (head region)
            bx1, by1, bx2, by2 = (float(v) for v in bbox[:4])
            return _clamp(bx1, by1, bx2, by1 + 0.32 * (by2 - by1), w, h)
        return None  # caller blurs the whole frame

    def _blur_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = box
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return
        if self.p.blur_method == "gaussian" and cv2 is not None:
            k = max(3, (min(region.shape[:2]) // 2) * 2 + 1)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
        else:
            frame[y1:y2, x1:x2] = _pixelate(region, self.p.pixelate_blocks)

    def deidentify(
        self,
        frame: np.ndarray,
        person: Optional[PosePerson] = None,
        keypoints: Optional[np.ndarray] = None,
        confs: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a copy of `frame` with the face blurred. Never mutates input.

        Accepts either a PosePerson or raw keypoints/confs/bbox. If the face
        cannot be located and blur_faces is on, the whole frame is blurred -
        the guard never returns an un-redacted face.
        """
        if not self.p.blur_faces:
            return frame
        if person is not None:
            keypoints, confs, bbox = person.keypoints, person.keypoints_conf, person.bbox
        out = frame.copy()
        box = self._face_box(out.shape[:2], keypoints, confs, bbox)
        if box is None:
            h, w = out.shape[:2]
            self._blur_region(out, (0, 0, w, h))
            log.debug("No face located; blurred whole frame as a precaution.")
        else:
            self._blur_region(out, box)
        return out

    # -- sanctioned write -------------------------------------------------- #
    def save_frame(
        self,
        frame: np.ndarray,
        name: str,
        person: Optional[PosePerson] = None,
        keypoints: Optional[np.ndarray] = None,
        confs: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """The ONLY sanctioned way to persist a frame.

        Returns the written path, or None if capture is disabled or the write
        could not be made. The frame is de-identified first.
        """
        if not self.p.allow_debug_capture:
            log.info("Debug capture disabled by policy; refusing to save %r.", name)
            return None
        if cv2 is None:
            log.warning("cv2 unavailable; cannot write %r.", name)
            return None

        safe = self.deidentify(frame, person=person, keypoints=keypoints,
                               confs=confs, bbox=bbox)
        out_dir = Path(self.p.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{stamp}_{name}.jpg"
        ok = cv2.imwrite(str(path), safe)
        if not ok:
            log.warning("cv2.imwrite failed for %s.", path)
            return None
        self._prune(out_dir)
        log.info("Saved de-identified capture -> %s", path)
        return path

    def _prune(self, out_dir: Path) -> None:
        now = time.time()
        files = sorted(out_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        # by age
        for f in list(files):
            if (now - f.stat().st_mtime) > self.p.retain_hours * 3600.0:
                _safe_unlink(f)
                files.remove(f)
        # by count (drop oldest)
        while len(files) > self.p.max_debug_files:
            _safe_unlink(files.pop(0))


def _clamp(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1))); y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w))); y2 = int(max(0, min(y2, h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _pixelate(region: np.ndarray, blocks: int) -> np.ndarray:
    """Mosaic a region by averaging into `blocks` x `blocks` cells. cv2-free."""
    h, w = region.shape[:2]
    bx = max(1, blocks); by = max(1, blocks)
    ys = np.linspace(0, h, by + 1, dtype=int)
    xs = np.linspace(0, w, bx + 1, dtype=int)
    out = region.copy()
    for j in range(by):
        for i in range(bx):
            y0, y1 = ys[j], ys[j + 1]
            x0, x1 = xs[i], xs[i + 1]
            if y1 > y0 and x1 > x0:
                cell = region[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = cell.mean(axis=(0, 1)).astype(region.dtype)
    return out


def _safe_unlink(f: Path) -> None:
    try:
        f.unlink()
    except OSError as exc:  # noqa: BLE001
        log.debug("Could not delete %s: %s", f, exc)


def load_privacy_guard(config_path: str | Path = "config/signal.yaml") -> PrivacyGuard:
    """Build a PrivacyGuard, reading an optional top-level `privacy:` block.

    Falls back to the safe defaults (capture OFF, blur ON) when the file,
    PyYAML, or the block is absent.
    """
    policy = None
    path = Path(config_path)
    if yaml is not None and path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
            policy = PrivacyPolicy.from_dict(raw.get("privacy"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read privacy config from %s (%s); using safe defaults.", path, exc)
    return PrivacyGuard(policy)