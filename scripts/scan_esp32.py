"""
scan_esp32.py — checks the Waveshare ESP32-S3 thermal camera without crashing it

Usage:
    python scripts/scan_esp32.py                 # safe: checks /thermal/frame only (default)
    python scripts/scan_esp32.py --ip http://192.168.4.1
    python scripts/scan_esp32.py --full          # after new firmware ONLY: probes other paths

Why the default touches one path only: requests to paths the firmware does not serve make the
ESP32-S3 allocate error pages it cannot free fast enough; a handful in a row exhausts its RAM
and the board browns out and powers off (the "scan powers the camera off" symptom). The monitor
and test_waveshare_thermal.py only ever request /thermal/frame, which is why they run for hours.

So the default mode fetches /thermal/frame a few times, slowly, with "Connection: close",
decodes the 9920-byte payload (62 x 80 int16, centi-degC) and reports the frame rate.

--full probes the other paths with 2 s gaps and stops at the first path that does not answer
(the board is then struggling); power-cycle the camera afterwards before running the monitor.
"""

import argparse
import time

import requests

KNOWN = "/thermal/frame"            # what camera.yaml -> waveshare_thermal.endpoint uses
OTHER_PATHS = ["/", "/thermal", "/thermal/raw", "/raw", "/stream", "/status", "/capture"]
CLOSE = {"Connection": "close"}     # never keep a socket open on the ESP32 between requests
EXPECTED_BYTES = 62 * 80 * 2


def check_frames(base_ip: str, n: int = 5, gap_s: float = 0.5) -> bool:
    """Fetch the known endpoint n times, gap_s apart. True when every frame decodes."""
    import numpy as np
    url = f"{base_ip}{KNOWN}"
    print(f"[INFO] Checking {url} ({n} frames, {gap_s:.1f} s apart). "
          f"Windows must be on the 'WSThermal' Wi-Fi.\n")
    ok, t0 = 0, time.monotonic()
    for i in range(n):
        try:
            r = requests.get(url, timeout=2.0, headers=CLOSE)
            raw = r.content
            if r.status_code == 200 and len(raw) == EXPECTED_BYTES:
                arr = np.frombuffer(raw, dtype=np.int16).reshape(62, 80).astype(np.float32) / 100.0
                print(f"  [OK  ] frame {i + 1}: min {arr.min():.1f} C  max {arr.max():.1f} C  "
                      f"mean {arr.mean():.1f} C")
                ok += 1
            else:
                print(f"  [FAIL] frame {i + 1}: HTTP {r.status_code}, {len(raw)} bytes "
                      f"(expected {EXPECTED_BYTES}) - firmware/sensor mismatch")
        except requests.exceptions.RequestException as exc:
            print(f"  [ERR ] frame {i + 1}: {type(exc).__name__} - board off, or not on WSThermal")
        time.sleep(gap_s)
    print(f"\n[SUMMARY] {ok}/{n} frames decoded in {time.monotonic() - t0:.1f} s.")
    if ok == n:
        print(f"  The monitor will work: camera.yaml -> waveshare_thermal.endpoint: \"{KNOWN}\"")
    elif ok:
        print("  Some requests failed: move the laptop closer to the camera or check its power supply.")
    else:
        print("  No frames: power-cycle the camera, reconnect to WSThermal, and run this again.")
    return ok == n


def full_scan(base_ip: str, gap_s: float = 2.0) -> list[str]:
    """Probe other paths (only needed after flashing new firmware). Stops at the first path
    that does not answer at all, because the board is then running out of memory."""
    print("[WARN] --full sends requests to paths the firmware may not serve. This can power the "
          "ESP32 off; power-cycle it afterwards.\n")
    active: list[str] = []
    for path in [KNOWN] + OTHER_PATHS:
        url = f"{base_ip}{path}"
        try:
            r = requests.get(url, timeout=3.0, headers=CLOSE)
            print(f"  [{r.status_code}] {url}  ({len(r.content)} bytes)")
            if r.status_code == 200:
                active.append(url)
        except requests.exceptions.RequestException:
            print(f"  [ERR] {url} did not answer - stopping so the board can recover.")
            break
        time.sleep(gap_s)
    print(f"\n[SUMMARY] {len(active)} path(s) answered 200: " + (", ".join(active) or "none"))
    return active


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the ESP32-S3 thermal camera")
    parser.add_argument("--ip", default="http://192.168.4.1",
                        help="ESP32 base URL (default: http://192.168.4.1)")
    parser.add_argument("--frames", type=int, default=5, help="frames to fetch in the safe check")
    parser.add_argument("--full", action="store_true",
                        help="also probe other paths (after new firmware only; may power the board off)")
    args = parser.parse_args()

    if args.full:
        full_scan(args.ip.rstrip("/"))
    else:
        check_frames(args.ip.rstrip("/"), n=max(1, args.frames))
