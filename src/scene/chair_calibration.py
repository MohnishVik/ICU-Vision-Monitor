"""Interactive chair/bed calibration for the VisionICU scene layer.

WHAT THIS IS
------------
At the start of every session the operator clicks the four corners of the
patient's chair or bed seat in the live camera view. Those four points
define a quadrilateral region in image coordinates. From it we derive:

  * a polygon      -> point-in-region tests feed ROIBundle.in_chair_frac
  * a seat depth   -> median depth (metres) over the region; the reference
                      distance that separates "seated at chair depth" from
                      "collapsed toward the floor"
  * image_size/ts  -> so a saved calibration can be validated on reload

WHY A QUADRILATERAL, NOT A BOX
------------------------------
A chair viewed from an angle is not an axis-aligned rectangle in the image;
perspective turns its right angles into a trapezoid. An axis-aligned box
would swallow floor and wall. Four clicked corners trace the seat exactly
as the camera sees it.

INTERACTION (OpenCV HighGUI window)
-----------------------------------
  * left click      add a corner (max 4; further clicks ignored)
  * r               reset all corners
  * space           grab a fresh frame (useful if the scene moved)
  * Enter / Return   confirm (only once 4 corners are placed)
  * Esc             cancel -> returns None

The window shows a live depth read-out under the cursor and a translucent
fill once four corners exist, so the operator can see exactly what the
system will treat as "the chair".

HEADLESS / NO DISPLAY
---------------------
On a box with no X display (SSH to the Jetson without a monitor) the GUI
cannot open. Use `ChairCalibration.from_corners(...)` to supply corners
programmatically, or `load_chair_calibration(path)` to reload a saved one.

DEPTH UNITS
-----------
RealSense raw depth is uint16 in millimetres (depth_scale ~= 0.001 m/unit).
If your D435Stream already converts frames to float metres, pass
`depth_already_meters=True`. The default assumes raw mm.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:  # cv2 is a hard dependency for the interactive path, soft for the rest
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only where cv2 is absent
    cv2 = None  # type: ignore

log = logging.getLogger("visionicu.scene.chair")

# Default RealSense raw-depth scale: millimetres -> metres.
_DEFAULT_DEPTH_SCALE = 0.001
_WINDOW = "VisionICU - click the 4 corners of the chair/bed (r reset, Enter ok, Esc cancel)"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered top-left, top-right, bottom-right, bottom-left.

    Clicks arrive in whatever order the operator likes; downstream geometry
    wants a stable winding. Top-left has the smallest x+y, bottom-right the
    largest. Top-right has the smallest (y-x), bottom-left the largest.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def _median_depth_m(
    depth: np.ndarray,
    polygon: np.ndarray,
    depth_already_meters: bool,
    depth_scale: float,
) -> Optional[float]:
    """Median valid depth (metres) inside the polygon, or None if unavailable."""
    if depth is None or cv2 is None:
        return None
    h, w = depth.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    vals = depth[mask > 0]
    vals = vals[vals > 0]  # RealSense returns 0 where it has no depth
    if vals.size == 0:
        return None
    med = float(np.median(vals.astype(np.float64)))
    if not depth_already_meters:
        med *= depth_scale
    return med


# --------------------------------------------------------------------------- #
# The calibration object
# --------------------------------------------------------------------------- #
@dataclass
class ChairCalibration:
    """A session's chair/bed region, in image coordinates.

    Attributes
    ----------
    corners : (4, 2) int array, ordered TL, TR, BR, BL.
    seat_depth_m : reference depth of the seat surface in metres, or None.
    image_size : (height, width) the calibration was made against.
    ts : monotonic/epoch timestamp of calibration.
    label : "chair" or "bed" (free text; informational only).
    """

    corners: np.ndarray
    seat_depth_m: Optional[float]
    image_size: tuple[int, int]
    ts: float
    label: str = "chair"
    _poly: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.corners = _order_corners(self.corners).astype(np.int32)
        self._poly = self.corners.reshape(-1, 1, 2).astype(np.int32)

    # -- region queries ---------------------------------------------------- #
    def contains(self, x: float, y: float) -> bool:
        """True if pixel (x, y) is inside (or on) the chair polygon."""
        if cv2 is None:
            return _contains_numpy(self.corners, x, y)
        return cv2.pointPolygonTest(self._poly, (float(x), float(y)), False) >= 0

    def fraction_inside(
        self,
        points: np.ndarray,
        confs: Optional[np.ndarray] = None,
        conf_min: float = 0.3,
    ) -> float:
        """Fraction of the given points that fall inside the chair region.

        Feeds ROIBundle.in_chair_frac. `points` is (N, 2) pixel coords
        (e.g. pose keypoints). Points below `conf_min` are ignored when
        `confs` is supplied. Returns 0.0 if no point qualifies.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if confs is not None:
            keep = np.asarray(confs, dtype=np.float64).reshape(-1) >= conf_min
            pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        inside = sum(1 for p in pts if self.contains(p[0], p[1]))
        return inside / float(pts.shape[0])

    def centroid(self) -> tuple[float, float]:
        """Pixel centroid of the four corners."""
        c = self.corners.mean(axis=0)
        return float(c[0]), float(c[1])

    # -- display ----------------------------------------------------------- #
    def draw(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (0, 200, 255),
        alpha: float = 0.25,
        thickness: int = 2,
    ) -> np.ndarray:
        """Return a copy of `frame` with the chair region overlaid."""
        if cv2 is None:
            return frame
        out = frame.copy()
        overlay = out.copy()
        cv2.fillPoly(overlay, [self._poly], color)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.polylines(out, [self._poly], True, color, thickness, cv2.LINE_AA)
        for (x, y) in self.corners:
            cv2.circle(out, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
        cx, cy = self.centroid()
        tag = self.label
        if self.seat_depth_m is not None:
            tag += f" {self.seat_depth_m:.2f} m"
        cv2.putText(
            out, tag, (int(cx) - 30, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        return out

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": self.corners.tolist(),
            "seat_depth_m": self.seat_depth_m,
            "image_size": list(self.image_size),
            "ts": self.ts,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChairCalibration":
        return cls(
            corners=np.asarray(d["corners"], dtype=np.int32),
            seat_depth_m=d.get("seat_depth_m"),
            image_size=tuple(d["image_size"]),  # type: ignore[arg-type]
            ts=float(d.get("ts", 0.0)),
            label=d.get("label", "chair"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        log.info("Chair calibration saved to %s", path)
        return path

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_corners(
        cls,
        corners: Sequence[Sequence[float]],
        depth: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
        label: str = "chair",
        depth_already_meters: bool = False,
        depth_scale: float = _DEFAULT_DEPTH_SCALE,
    ) -> "ChairCalibration":
        """Build a calibration from known corners (no GUI).

        If `depth` (aligned to the RGB frame) is given, the seat depth is
        computed from it; otherwise seat_depth_m is None.
        """
        corners_arr = _order_corners(np.asarray(corners, dtype=np.float64))
        seat_depth = (
            _median_depth_m(depth, corners_arr, depth_already_meters, depth_scale)
            if depth is not None
            else None
        )
        if image_size is None and depth is not None:
            image_size = (int(depth.shape[0]), int(depth.shape[1]))
        if image_size is None:
            image_size = (0, 0)
        return cls(
            corners=corners_arr.astype(np.int32),
            seat_depth_m=seat_depth,
            image_size=image_size,
            ts=time.time(),
            label=label,
        )

    @classmethod
    def run_interactive(
        cls,
        source: Any,
        label: str = "chair",
        depth_already_meters: bool = False,
        depth_scale: float = _DEFAULT_DEPTH_SCALE,
        save_path: Optional[str | Path] = None,
        bgr: bool = True,
    ) -> Optional["ChairCalibration"]:
        """Open the click-to-calibrate window and return a ChairCalibration.

        Parameters
        ----------
        source : one of
            * a FrameBundle (has `.rgb` and `.depth`)
            * an object with a `frames()` generator yielding FrameBundles
            * a tuple (rgb, depth)
            * an RGB ndarray on its own (depth unavailable)
        bgr : True if `rgb` is already in OpenCV's BGR order; set False to
            colour-correct the display only (clicks are unaffected either way).
        save_path : if given, the confirmed calibration is written there.

        Returns the calibration, or None if the operator pressed Esc.
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required for interactive calibration. "
                "Use ChairCalibration.from_corners(...) in headless mode."
            )

        rgb, depth = _grab(source)
        if rgb is None:
            raise RuntimeError("Calibration source produced no RGB frame.")

        clicks: list[tuple[int, int]] = []
        cursor: list[int] = [0, 0]

        def on_mouse(event: int, x: int, y: int, flags: int, _param: Any) -> None:
            cursor[0], cursor[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y))

        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WINDOW, on_mouse)

        try:
            while True:
                disp = rgb if bgr else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                disp = disp.copy()
                _render(disp, clicks, cursor, depth, depth_already_meters, depth_scale)
                cv2.imshow(_WINDOW, disp)
                key = cv2.waitKey(20) & 0xFF

                if key == 27:  # Esc
                    log.info("Chair calibration cancelled by operator.")
                    return None
                if key in (ord("r"), ord("R")):
                    clicks.clear()
                elif key == 32:  # space -> re-grab a frame
                    new_rgb, new_depth = _grab(source)
                    if new_rgb is not None:
                        rgb, depth = new_rgb, new_depth
                elif key in (13, 10):  # Enter / Return
                    if len(clicks) == 4:
                        break
                    log.info("Place all 4 corners before confirming (have %d).", len(clicks))
        finally:
            cv2.destroyWindow(_WINDOW)

        h, w = rgb.shape[:2]
        calib = cls.from_corners(
            corners=clicks,
            depth=depth,
            image_size=(h, w),
            label=label,
            depth_already_meters=depth_already_meters,
            depth_scale=depth_scale,
        )
        log.info(
            "Chair calibration confirmed: corners=%s seat_depth=%s",
            calib.corners.tolist(),
            f"{calib.seat_depth_m:.3f} m" if calib.seat_depth_m is not None else "n/a",
        )
        if save_path is not None:
            calib.save(save_path)
        return calib


# --------------------------------------------------------------------------- #
# Interactive rendering + frame grabbing
# --------------------------------------------------------------------------- #
def _render(
    disp: np.ndarray,
    clicks: list[tuple[int, int]],
    cursor: list[int],
    depth: Optional[np.ndarray],
    depth_already_meters: bool,
    depth_scale: float,
) -> None:
    color = (0, 200, 255)
    for i, (x, y) in enumerate(clicks):
        cv2.circle(disp, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(disp, str(i + 1), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if len(clicks) >= 2:
        pts = np.array(clicks, dtype=np.int32)
        closed = len(clicks) == 4
        cv2.polylines(disp, [pts], closed, color, 2, cv2.LINE_AA)
    if len(clicks) == 4:
        overlay = disp.copy()
        cv2.fillPoly(overlay, [np.array(clicks, dtype=np.int32)], color)
        cv2.addWeighted(overlay, 0.25, disp, 0.75, 0, disp)

    # live depth read-out under the cursor
    cx, cy = cursor
    hud = f"corners {len(clicks)}/4"
    if depth is not None and 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
        raw = float(depth[cy, cx])
        if raw > 0:
            m = raw if depth_already_meters else raw * depth_scale
            hud += f"   depth @cursor {m:.2f} m"
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(disp, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def _grab(source: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Pull one (rgb, depth) pair from a flexible source. depth may be None."""
    # FrameBundle-like
    if hasattr(source, "rgb"):
        return getattr(source, "rgb", None), getattr(source, "depth", None)
    # stream with a frames() generator
    if hasattr(source, "frames") and callable(source.frames):
        for bundle in source.frames():
            return getattr(bundle, "rgb", None), getattr(bundle, "depth", None)
        return None, None
    # (rgb, depth) tuple/list
    if isinstance(source, (tuple, list)) and len(source) == 2:
        return source[0], source[1]
    # bare rgb array
    if isinstance(source, np.ndarray):
        return source, None
    raise TypeError(f"Unsupported calibration source: {type(source)!r}")


def _contains_numpy(corners: np.ndarray, x: float, y: float) -> bool:
    """Ray-casting point-in-polygon fallback used only when cv2 is absent."""
    poly = np.asarray(corners, dtype=np.float64)
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- #
# Module-level loader
# --------------------------------------------------------------------------- #
def load_chair_calibration(path: str | Path) -> ChairCalibration:
    """Load a previously saved chair calibration from JSON."""
    path = Path(path)
    return ChairCalibration.from_dict(json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
# Standalone entry point: python -m src.scene.chair_calibration
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Interactive chair/bed calibration.")
    ap.add_argument("--save", default="config/sessions/chair_calibration.json",
                    help="where to write the calibration JSON")
    ap.add_argument("--label", default="chair", choices=["chair", "bed"])
    ap.add_argument("--image", default=None,
                    help="calibrate against a still image instead of the camera")
    ap.add_argument("--depth-meters", action="store_true",
                    help="set if the depth frame is already in metres")
    args = ap.parse_args()

    if args.image is not None:
        if cv2 is None:
            raise SystemExit("cv2 required to load an image.")
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Could not read image: {args.image}")
        src: Any = img
    else:
        try:
            from src.capture.d435_stream import D435Stream  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Could not import D435Stream ({exc}). "
                "Pass --image to calibrate against a still instead."
            )
        stream = D435Stream()
        stream.start()
        src = stream

    calib = ChairCalibration.run_interactive(
        src, label=args.label, depth_already_meters=args.depth_meters, save_path=args.save
    )
    if calib is None:
        print("Cancelled.")
    else:
        print(f"Saved calibration -> {args.save}")
        print(f"  corners: {calib.corners.tolist()}")
        print(f"  seat_depth_m: {calib.seat_depth_m}")