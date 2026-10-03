"""
scan_esp32.py — ESP32-S3 HTTP endpoint scanner
Scans for active endpoints on the Waveshare thermal camera AP.
Run this if you flash new firmware or the thermal stream stops responding.

Usage:
    python scripts/scan_esp32.py
    python scripts/scan_esp32.py --ip 192.168.4.1

Confirmed active endpoint: /thermal/frame (returns 9920 bytes binary)
"""

import argparse
import requests
import time


ENDPOINTS = [
    "/", "/thermal", "/thermal/", "/thermal/raw",
    "/thermal/frame",     # <-- confirmed working
    "/raw", "/stream", "/index.html", "/debug",
    "/thermal/debug", "/status", "/capture"
]


def scan_esp32_endpoints(base_ip: str = "http://192.168.4.1") -> list[str]:
    print(f"[INFO] Scanning ESP32-S3 HTTP server at {base_ip}...")
    print(f"[INFO] Make sure Windows is connected to 'WSThermal' AP\n")

    active: list[str] = []

    for path in ENDPOINTS:
        url = f"{base_ip}{path}"
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                content_len = len(response.content)
                content_preview = response.content[:16].hex() if response.content else ""
                print(f"  [200 OK ] {url}  ({content_len} bytes)  [{content_preview}...]")
                active.append(url)
            elif response.status_code == 404:
                print(f"  [404    ] {url}")
            else:
                print(f"  [{response.status_code}    ] {url}")
        except requests.exceptions.RequestException:
            print(f"  [ERR    ] {url}  (connection failed)")

    print(f"\n[SUMMARY] {len(active)} working URL(s) found.")
    if active:
        print(f"  Use in camera.yaml → waveshare_thermal.endpoint")
        for u in active:
            print(f"    {u}")
    else:
        print("  CRITICAL: 0 working URLs. Check WiFi connection to WSThermal AP.")
        print("  SPIFFS partition may have failed to mount on the ESP32.")

    return active


def check_frame_payload(url: str = "http://192.168.4.1/thermal/frame") -> None:
    """Quick check: fetch one frame and verify the 9920-byte payload."""
    import numpy as np
    print(f"\n[INFO] Checking payload from {url} ...")
    try:
        r = requests.get(url, timeout=2.0)
        raw = r.content
        print(f"  Status   : {r.status_code}")
        print(f"  Bytes    : {len(raw)}  (expected 9920 = 62*80*2)")
        if len(raw) == 9920:
            arr = np.frombuffer(raw, dtype=np.int16).reshape(62, 80).astype(np.float32) / 100.0
            print(f"  Min temp : {arr.min():.2f} C")
            print(f"  Max temp : {arr.max():.2f} C")
            print(f"  Mean temp: {arr.mean():.2f} C")
            print("  [PASS] Payload decodes correctly.")
        else:
            print("  [FAIL] Unexpected byte count — firmware/sensor mismatch.")
    except Exception as e:
        print(f"  [ERR] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan ESP32-S3 endpoints")
    parser.add_argument("--ip", default="http://192.168.4.1",
                        help="ESP32 base IP (default: http://192.168.4.1)")
    parser.add_argument("--check-payload", action="store_true",
                        help="Also decode one thermal frame and print stats")
    args = parser.parse_args()

    active = scan_esp32_endpoints(args.ip)

    if args.check_payload and active:
        # Try the confirmed endpoint first
        target = f"{args.ip}/thermal/frame"
        if target in active:
            check_frame_payload(target)
        else:
            check_frame_payload(active[0])
