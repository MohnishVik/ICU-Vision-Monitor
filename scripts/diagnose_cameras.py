"""
diagnose_cameras.py — unified camera diagnostic tool
Combines D435 alignment test and Waveshare thermal stream test.
Run this before any recording session to verify both cameras are working.

Usage:
    python scripts/diagnose_cameras.py              # both cameras
    python scripts/diagnose_cameras.py --d435-only
    python scripts/diagnose_cameras.py --thermal-only

Press 'q' or ESC to exit.
"""

import argparse
import time
import cv2
import numpy as np
import requests

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.capture.d435_stream import D435Stream
from src.capture.thermal_stream import ThermalStream, make_display_frame


def run_d435_diagnostic(cfg: dict) -> None:
    """D435 diagnostic: side-by-side RGB + colourised depth with HUD overlay."""
    colorizer = None
    try:
        import pyrealsense2 as rs
        colorizer = rs.colorizer()
    except ImportError:
        print("[D435] pyrealsense2 not installed — skipping D435 diagnostic.")
        return

    stream = D435Stream(cfg["d435"])
    print("[D435] Starting diagnostic... press 'q' or ESC to exit.")

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    with stream:
        for bundle in stream.frames():
            color_image = bundle.rgb.copy()
            # Re-fetch the raw depth frame for colorizer (bundle.depth is aligned uint16)
            # For display we apply colorizer directly on the numpy depth
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(bundle.depth, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Centre depth
            h, w = bundle.depth.shape
            cx, cy = w // 2, h // 2
            centre_mm = float(bundle.depth[cy, cx])
            centre_m  = centre_mm * 0.001

            # HUD
            cv2.putText(color_image, f"FPS: {fps:.1f}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(color_image, f"Centre Depth: {centre_m:.2f} m",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(color_image, f"Exp: 156  WB: 4600 (LOCKED)",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.drawMarker(color_image, (cx, cy), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(depth_display, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)

            display = np.hstack((color_image, depth_display))
            cv2.imshow("VisionICU D435 — RGB | Depth (press q)", display)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    cv2.destroyAllWindows()


def run_thermal_diagnostic(cfg: dict) -> None:
    """Waveshare ESP32-S3 thermal diagnostic: live colourised frame with temp overlay."""
    stream = ThermalStream(cfg["waveshare_thermal"])
    stream.start()

    print("[Thermal] Starting diagnostic... press 'q' or ESC to exit.")
    print(f"[Thermal] Polling → {cfg['waveshare_thermal']['full_url']}")
    print("[Thermal] Ensure Windows is connected to 'WSThermal' WiFi AP.")

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            frame, ts = stream.get_latest_frame()

            if frame is not None:
                display = make_display_frame(frame, display_size=(640, 480))

                frame_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.time()

                cv2.putText(display, f"FPS: {fps:.1f}",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"Max: {frame.max():.1f} C",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"Min: {frame.min():.1f} C",
                            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                cv2.putText(display, f"Mean: {frame.mean():.1f} C",
                            (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 180), 2)
                cv2.putText(display, f"Frame: {frame.shape[0]}x{frame.shape[1]}",
                            (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                # No frame yet — show waiting screen
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display, "Waiting for thermal stream...",
                            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(display, "Connect to WSThermal WiFi AP",
                            (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

            cv2.imshow("VisionICU Thermal — Waveshare ESP32-S3 (press q)", display)

            if cv2.waitKey(30) & 0xFF in [ord('q'), 27]:
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionICU camera diagnostics")
    parser.add_argument("--d435-only",    action="store_true")
    parser.add_argument("--thermal-only", action="store_true")
    parser.add_argument("--config", default="config/camera.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.thermal_only:
        run_thermal_diagnostic(cfg)
    elif args.d435_only:
        run_d435_diagnostic(cfg)
    else:
        # Default: run both sequentially
        print("=== D435 Diagnostic ===")
        run_d435_diagnostic(cfg)
        print("\n=== Thermal Diagnostic ===")
        run_thermal_diagnostic(cfg)
