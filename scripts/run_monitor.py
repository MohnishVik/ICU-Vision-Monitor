#!/usr/bin/env python3
"""VisionICU - end-to-end contact-free ICU monitor.

LAUNCH FLOW
-----------
    1. open the cameras        SyncBuffer(load_camera_config())   (D435 + thermal)
    2. calibrate the chair     one interactive click at startup   (or --load-calib)
    3. per frame:
         pose.infer            -> every person in view
         tracker.update        -> the ONE locked patient
         roi_builder.build     -> face / chest / abdomen ROIs (ROIBundle)
         RR, HR, Fall, TAA     -> Measurements
         EWS                   -> NEWS2-style aggregate + badge
         HUD                   -> live overlay for the laptop/Jetson monitor
    4. on a CONFIRMED fall     -> de-identified snapshot via the privacy guard

This is the first point the whole system runs together. On the laptop it uses
CPU; on the Jetson Orin set `device: cuda` in config/models.yaml.

KEYS (in the display window)
    q / Esc  quit        r  re-calibrate the chair
    s        save a de-identified snapshot now (debug)

DESIGN NOTE
-----------
The capture layer is imported lazily inside main(), so this file compiles and
its per-frame logic can be unit-tested without pyrealsense2 or the cameras.
Fall runs with local_pose=False: pose is computed ONCE here and passed to Fall
through the ROIBundle, never a second YOLO pass.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.types import FrameBundle, ROIBundle, Measurement
from src.scene import (
    ChairCalibration,
    load_chair_calibration,
    load_pose_estimator,
    load_person_tracker,
    load_roi_builder,
    load_privacy_guard,
)
from src.modules.rr import load_rr_module
from src.modules.hr import load_hr_module
from src.modules.fall import load_fall_module
from src.modules.taa import load_taa_module

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("visionicu.run")

# BGR badge colours
_COL = {
    "OK":   (0, 180, 0),
    "LOW":  (0, 220, 220),
    "MED":  (0, 140, 255),
    "HIGH": (0, 0, 255),
}


# --------------------------------------------------------------------------- #
# EWS - NEWS2-style aggregate over the vitals we actually measure
# --------------------------------------------------------------------------- #
class EWSBoard:
    """Maps RR/HR (+ fall/apnea/TAA flags) to an aggregate severity.

    Bands follow NEWS2 (which match config/ews_rules.yaml): RR 8/11/20/24,
    HR 40/50/90/110/130. A single '3' sub-score lifts severity to at least
    MED; a confirmed fall or apnea forces HIGH.
    """

    @staticmethod
    def _rr_sub(bpm: float) -> int:
        if bpm <= 8: return 3
        if bpm <= 11: return 1
        if bpm <= 20: return 0
        if bpm <= 24: return 2
        return 3

    @staticmethod
    def _hr_sub(bpm: float) -> int:
        if bpm <= 40: return 3
        if bpm <= 50: return 1
        if bpm <= 90: return 0
        if bpm <= 110: return 1
        if bpm <= 130: return 2
        return 3

    def score(self, meas: dict[str, Measurement]) -> dict[str, Any]:
        total = 0
        any_three = False
        reasons: list[str] = []

        rr = meas.get("rr.depth")
        if rr is not None and rr.value is not None:
            s = self._rr_sub(rr.value); total += s; any_three |= (s == 3)
            if s >= 2: reasons.append(f"RR {rr.value:.0f} (sub {s})")

        hr = meas.get("hr.pos")
        if hr is not None and hr.value is not None:
            s = self._hr_sub(hr.value); total += s; any_three |= (s == 3)
            if s >= 2: reasons.append(f"HR {hr.value:.0f} (sub {s})")

        # NEWS2 aggregate -> level
        if total >= 7:
            level = "HIGH"
        elif total >= 5 or any_three:
            level = "MED"
        elif total >= 1:
            level = "LOW"
        else:
            level = "OK"

        # hard overrides from event modules
        fall = meas.get("fall.classifier")
        if fall is not None and fall.meta.get("confirmed_fall"):
            level = "HIGH"; reasons.append("FALL confirmed")
        if rr is not None and rr.meta.get("apnea"):
            level = "HIGH"; reasons.append("APNEA")
        taa = meas.get("taa.depth")
        if taa is not None and taa.meta.get("taa_flag"):
            if level in ("OK", "LOW"):
                level = "MED"
            reasons.append("paradoxical breathing")

        return {"total": total, "level": level, "color": _COL[level], "reasons": reasons}


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
@dataclass
class Components:
    pose: Any
    tracker: Any
    roi_builder: Any
    privacy: Any
    rr: Any
    hr: Any
    fall: Any
    taa: Any
    ews: EWSBoard


def build_components(
    chair: ChairCalibration,
    config_dir: str,
    patient_id: str,
    fps: float,
    image_size: tuple[int, int],
) -> Components:
    cfg = lambda name: str(Path(config_dir) / name)  # noqa: E731
    models_yaml = cfg("models.yaml")
    signal_yaml = cfg("signal.yaml")
    return Components(
        pose=load_pose_estimator(models_yaml),
        tracker=load_person_tracker(chair, signal_yaml, image_size=image_size),
        roi_builder=load_roi_builder(signal_yaml, patient_id=patient_id),
        privacy=load_privacy_guard(signal_yaml),
        rr=load_rr_module(signal_yaml, depth_fs=fps),
        hr=load_hr_module(signal_yaml, fs=fps),
        fall=load_fall_module(signal_yaml, local_pose=False),  # pose comes via ROIBundle
        taa=load_taa_module(signal_yaml, fs=fps),
        ews=EWSBoard(),
    )


# --------------------------------------------------------------------------- #
# Per-frame pipeline (standalone + testable)
# --------------------------------------------------------------------------- #
def process_frame(
    bundle: FrameBundle,
    chair: ChairCalibration,
    comp: Components,
) -> tuple[Any, ROIBundle, dict[str, Measurement]]:
    people = comp.pose.infer(bundle.rgb)
    track = comp.tracker.update(people, ts=bundle.ts_mono)
    roi = comp.roi_builder.build(bundle, track)

    meas: dict[str, Measurement] = {}
    for mod in (comp.rr, comp.hr, comp.fall, comp.taa):
        try:
            out = mod.process(bundle, roi)
            if out:
                meas.update(out)
        except Exception as exc:  # noqa: BLE001 - one module must not kill the loop
            log.exception("Module %s failed: %s", type(mod).__name__, exc)
    return track, roi, meas


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
def _fmt(m: Optional[Measurement], unit: str) -> str:
    if m is None or m.value is None:
        return "--"
    return f"{m.value:.0f}{unit} (sqi {m.sqi:.2f})"


def draw_hud(frame, chair, track, meas, ews_result, thermal_online: bool):
    if cv2 is None:
        return frame
    out = chair.draw(frame)
    if track.person is not None:
        comp_people = [track.person]
        # lightweight skeleton via pose module's drawer if available
        try:
            from src.scene.pose import PoseEstimator  # noqa: F401
            # reuse draw by constructing a throwaway estimator is wasteful; draw inline
        except Exception:
            pass
        x1, y1, x2, y2 = (int(v) for v in track.person.bbox[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), ews_result["color"], 2)

    # text panel
    lines = [
        f"RR  {_fmt(meas.get('rr.depth'), ' bpm')}",
        f"HR  {_fmt(meas.get('hr.pos'), ' bpm')}",
        f"TAA {'FLAG' if (meas.get('taa.depth') and meas['taa.depth'].meta.get('taa_flag')) else 'ok'}",
        f"Fall {_fall_state(meas.get('fall.classifier'))}",
        f"in_chair {track.in_chair_frac:.2f}  staff {track.staff_count}",
        f"thermal {'on' if thermal_online else 'off'}",
    ]
    y = 44
    cv2.rectangle(out, (0, 24), (270, 24 + 22 * (len(lines) + 1)), (0, 0, 0), -1)
    for ln in lines:
        cv2.putText(out, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    # EWS badge
    badge = f"EWS {ews_result['total']}  {ews_result['level']}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (out.shape[1] - tw - 20, 8), (out.shape[1] - 4, 8 + th + 14),
                  ews_result["color"], -1)
    cv2.putText(out, badge, (out.shape[1] - tw - 14, 8 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if ews_result["reasons"]:
        cv2.putText(out, " | ".join(ews_result["reasons"])[:60],
                    (8, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ews_result["color"], 1, cv2.LINE_AA)
    return out


def _fall_state(m: Optional[Measurement]) -> str:
    if m is None or m.value is None:
        return "--"
    if m.meta.get("confirmed_fall"):
        return "CONFIRMED"
    return {0: "normal", 1: "falling", 2: "lying"}.get(int(m.value), str(int(m.value)))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(buf, args) -> Optional[ChairCalibration]:
    if args.load_calib and Path(args.load_calib).exists():
        log.info("Loading saved chair calibration from %s", args.load_calib)
        return load_chair_calibration(args.load_calib)
    save_path = args.save_calib or "config/sessions/chair_calibration.json"
    return ChairCalibration.run_interactive(
        buf, label=args.label, save_path=save_path,
        depth_already_meters=False,  # D435 raw mm
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(buf, args) -> None:
    chair = calibrate(buf, args)
    if chair is None:
        log.info("Calibration cancelled; exiting.")
        return

    # probe one frame for the image size
    image_size = (chair.image_size if chair.image_size != (0, 0) else (480, 640))
    comp = build_components(chair, args.config_dir, args.patient_id, args.fps, image_size)
    log.info("Monitor running. Patient=%s  thermal=%s  mount~%.2fm",
             args.patient_id, buf.thermal_online(), (buf.mount_distance_m() or float('nan')))

    last_fall_saved = 0.0
    try:
        for bundle in buf.frames():
            track, roi, meas = process_frame(bundle, chair, comp)
            ews_result = comp.ews.score(meas)

            # de-identified snapshot on a confirmed fall (rate-limited)
            fall = meas.get("fall.classifier")
            if fall is not None and fall.meta.get("confirmed_fall") \
                    and (bundle.ts_mono - last_fall_saved) > 10.0:
                comp.privacy.save_frame(bundle.rgb, f"fall_{args.patient_id}", person=track.person)
                last_fall_saved = bundle.ts_mono

            if not args.no_display and cv2 is not None:
                hud = draw_hud(bundle.rgb, chair, track, meas, ews_result, buf.thermal_online())
                cv2.imshow("VisionICU", hud)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    cv2.destroyWindow("VisionICU")
                    chair = calibrate(buf, args)
                    if chair is None:
                        break
                    comp.tracker = load_person_tracker(
                        chair, str(Path(args.config_dir) / "signal.yaml"), image_size=image_size)
                    comp.roi_builder = load_roi_builder(
                        str(Path(args.config_dir) / "signal.yaml"), patient_id=args.patient_id)
                if k == ord("s"):
                    comp.privacy.save_frame(bundle.rgb, f"manual_{args.patient_id}", person=track.person)
    finally:
        if cv2 is not None and not args.no_display:
            cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="VisionICU contact-free monitor.")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--patient-id", default="patient")
    ap.add_argument("--label", default="chair", choices=["chair", "bed"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--load-calib", default=None, help="skip the click; load this calibration JSON")
    ap.add_argument("--save-calib", default=None, help="where to write the new calibration")
    ap.add_argument("--no-display", action="store_true", help="headless (no HUD window)")
    args = ap.parse_args()

    # lazy import: keeps this file importable without pyrealsense2 / hardware
    from src.capture.sync_buffer import SyncBuffer, load_camera_config

    cam_cfg = load_camera_config(str(Path(args.config_dir) / "camera.yaml"))
    with SyncBuffer(cam_cfg) as buf:
        run(buf, args)


if __name__ == "__main__":
    main()