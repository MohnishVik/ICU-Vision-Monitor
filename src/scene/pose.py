"""Shared 2-D human pose estimator for the VisionICU scene layer.

WHAT THIS IS
------------
A single wrapper around Ultralytics YOLOv8n-pose that every module shares.
It runs the network once per frame and returns *all* detected people, each
as 17 COCO keypoints + confidences + a bounding box. The scene layer picks
which person is the patient (see person_tracker.py) and writes that one into
ROIBundle.pose_keypoints / pose_keypoints_conf / person_bbox. fall.py then
consumes those fields instead of running YOLO a second time.

WHY A SHARED ESTIMATOR
----------------------
Think of the pose network as an expensive pair of eyes. Running it twice -
once in the scene layer and again inside fall.py - doubles the GPU cost for
the same picture and can even disagree with itself frame to frame. One set
of eyes, read by everyone, is cheaper and consistent.

Two things this fixes versus the inline _LocalPose in fall.py:
  * device is read from config, so the Jetson Orin runs on CUDA (the inline
    version hardcodes CPU);
  * it returns every person, not just boxes[0], so the tracker can choose
    the patient rather than trusting detection order.

COCO-17 KEYPOINT ORDER (what Ultralytics yolov8-pose emits)
-----------------------------------------------------------
  0 nose        1 L-eye      2 R-eye      3 L-ear      4 R-ear
  5 L-shoulder  6 R-shoulder 7 L-elbow    8 R-elbow    9 L-wrist
 10 R-wrist    11 L-hip     12 R-hip     13 L-knee    14 R-knee
 15 L-ankle    16 R-ankle
fall.py reads shoulders (5, 6) and hips (11, 12); this order matches it.

INPUT COLOUR ORDER
------------------
Ultralytics expects a BGR ndarray (OpenCV's native order). Pass the D435
`rgb` frame straight through if it was captured as bgr8.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.scene.pose")

COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# edges for drawing a skeleton (COCO index pairs)
_COCO_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    (0, 5), (0, 6),                            # head -> shoulders
)

_DEFAULTS = {
    "weights": "models/public/yolov8n-pose.pt",
    "conf": 0.25,
    "iou": 0.45,
    "img_size": 640,
    "device": "cpu",
}


@dataclass
class PosePerson:
    """One detected person in a frame."""

    keypoints: np.ndarray        # (17, 2) float32 pixel coords
    keypoints_conf: np.ndarray   # (17,)  float32 in [0, 1]
    bbox: np.ndarray             # (4,)   float32 [x1, y1, x2, y2]
    score: float                 # detection/box confidence

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = (float(v) for v in self.bbox[:4])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in self.bbox[:4])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class PoseEstimator:
    """Lazy-loading YOLOv8n-pose wrapper.

    The Ultralytics/torch import is deferred to the first `infer` call, so
    importing this module (and constructing the estimator) is cheap and does
    not require the heavy deps to be installed - handy for laptop development
    before the model is in place.
    """

    def __init__(
        self,
        weights: str | Path = _DEFAULTS["weights"],
        conf: float = _DEFAULTS["conf"],
        iou: float = _DEFAULTS["iou"],
        img_size: int = _DEFAULTS["img_size"],
        device: str | int = _DEFAULTS["device"],
    ) -> None:
        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        self.device = device
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"Pose weights not found: {self.weights}. "
                "Copy yolov8n-pose.pt into models/public/."
            )
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ultralytics is required for pose inference "
                "(pip install ultralytics). "
                f"Import failed: {exc}"
            ) from exc
        log.info("Loading pose model %s on device=%s", self.weights, self.device)
        self._model = YOLO(self.weights)

    # -- inference --------------------------------------------------------- #
    def infer(self, rgb_bgr: np.ndarray) -> list[PosePerson]:
        """Run pose on one BGR frame. Returns people sorted by score (desc)."""
        self._ensure_model()
        res = self._model.predict(
            rgb_bgr, verbose=False,
            conf=self.conf, iou=self.iou, imgsz=self.img_size, device=self.device,
        )[0]

        n = len(res.boxes)
        if n == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        has_kp = res.keypoints is not None and res.keypoints.xy is not None
        kp_xy = res.keypoints.xy.cpu().numpy() if has_kp else None
        kp_cf = (
            res.keypoints.conf.cpu().numpy()
            if has_kp and res.keypoints.conf is not None
            else None
        )

        people: list[PosePerson] = []
        for i in range(n):
            if kp_xy is not None and i < len(kp_xy):
                kpts = kp_xy[i].astype(np.float32)
                confs = (
                    kp_cf[i].astype(np.float32)
                    if kp_cf is not None and i < len(kp_cf)
                    else np.ones(17, dtype=np.float32)
                )
            else:
                kpts = np.zeros((17, 2), dtype=np.float32)
                confs = np.zeros(17, dtype=np.float32)
            people.append(
                PosePerson(
                    keypoints=kpts,
                    keypoints_conf=confs,
                    bbox=boxes[i].astype(np.float32),
                    score=float(scores[i]),
                )
            )
        people.sort(key=lambda p: p.score, reverse=True)
        return people

    def infer_top(self, rgb_bgr: np.ndarray) -> Optional[PosePerson]:
        """Highest-confidence person, or None. Mirrors fall.py's old behaviour."""
        people = self.infer(rgb_bgr)
        return people[0] if people else None

    # -- display ----------------------------------------------------------- #
    def draw(
        self,
        frame: np.ndarray,
        people: list[PosePerson],
        conf_min: float = 0.3,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Overlay skeletons + boxes for debugging. Needs cv2; no-op without it."""
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover
            return frame
        out = frame.copy()
        for p in people:
            x1, y1, x2, y2 = (int(v) for v in p.bbox[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{p.score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for a, b in _COCO_SKELETON:
                if p.keypoints_conf[a] >= conf_min and p.keypoints_conf[b] >= conf_min:
                    pa = tuple(int(v) for v in p.keypoints[a])
                    pb = tuple(int(v) for v in p.keypoints[b])
                    cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)
            for k in range(17):
                if p.keypoints_conf[k] >= conf_min:
                    cv2.circle(out, tuple(int(v) for v in p.keypoints[k]),
                               3, color, -1, cv2.LINE_AA)
        return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_pose_estimator(config_path: str | Path = "config/models.yaml") -> PoseEstimator:
    """Build a PoseEstimator from config/models.yaml.

    Understands the project's native schema:

        checkpoints:
          pose:
            yolo_pose: "models/public/yolov8n-pose.pt"
        modules:
          pose: { conf: 0.25, iou: 0.45, img_size: 640 }   # optional
        device: "cuda"                                       # top-level, shared

    Resolution order for each setting (first hit wins), so an older flat
    `pose:` block or the `fall:` block still work as overrides:

        weights  : pose.weights -> checkpoints.pose.yolo_pose
                   -> checkpoints.pose.public -> fall.yolo_weights -> default
        conf/iou : pose.*       -> modules.pose.* -> fall.yolo_* -> default
        img_size : pose.img_size-> modules.pose.img_size -> fall.yolo_img_size -> default
        device   : pose.device  -> modules.pose.device -> top-level device
                   -> fall.device -> default

    Any missing key uses its default, so a partial or absent config still
    yields a usable estimator (which errors only when it needs the weights).
    """
    cfg = dict(_DEFAULTS)
    path = Path(config_path)

    if yaml is not None and path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse %s (%s); using pose defaults.", path, exc)
            raw = {}

        pose_block = raw.get("pose") or {}               # optional flat override
        checkpoints = raw.get("checkpoints") or {}
        ck_pose = checkpoints.get("pose") or {}
        modules = raw.get("modules") or {}
        mod_pose = modules.get("pose") or {}
        fall_block = raw.get("fall") or {}
        top_device = raw.get("device")                   # shared inference device

        cfg["weights"] = (
            pose_block.get("weights")
            or ck_pose.get("yolo_pose")
            or ck_pose.get("public")
            or fall_block.get("yolo_weights")
            or cfg["weights"]
        )
        cfg["conf"] = pose_block.get(
            "conf", mod_pose.get("conf", fall_block.get("yolo_conf", cfg["conf"]))
        )
        cfg["iou"] = pose_block.get(
            "iou", mod_pose.get("iou", fall_block.get("yolo_iou", cfg["iou"]))
        )
        cfg["img_size"] = pose_block.get(
            "img_size",
            mod_pose.get("img_size", fall_block.get("yolo_img_size", cfg["img_size"])),
        )
        cfg["device"] = pose_block.get(
            "device",
            mod_pose.get(
                "device",
                top_device if top_device is not None
                else fall_block.get("device", cfg["device"]),
            ),
        )
    else:
        log.info("No config at %s (or PyYAML missing); using pose defaults.", path)

    return PoseEstimator(
        weights=cfg["weights"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        img_size=cfg["img_size"],
        device=cfg["device"],
    )


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run pose on a still image.")
    ap.add_argument("image", help="path to a test image")
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--out", default=None, help="optional path to save the overlay")
    args = ap.parse_args()

    import cv2  # type: ignore

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    est = load_pose_estimator(args.config)
    ppl = est.infer(img)
    print(f"Detected {len(ppl)} person(s).")
    for i, p in enumerate(ppl):
        print(f"  [{i}] score={p.score:.2f} bbox={p.bbox.tolist()} "
              f"kpts_ok={int((p.keypoints_conf >= 0.3).sum())}/17")
    if args.out:
        cv2.imwrite(args.out, est.draw(img, ppl))
        print(f"Overlay -> {args.out}")