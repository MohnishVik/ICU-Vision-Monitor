"""
main.py — VisionICU entry point.

Starts the monitoring engine and the web dashboard in one process:

    python main.py                      # real D435 camera, dashboard at http://127.0.0.1:8000
    python main.py --demo               # 4 SIMULATED beds, no hardware needed (UI development)
    python main.py --host 0.0.0.0       # reachable from a nurse-station PC on the same network
    python main.py --reuse-calibration  # skip the click if this bed was calibrated before
    python main.py --hud [...]          # legacy OpenCV window (scripts/run_monitor.py)

Build the dashboard once before the first run:  cd frontend && npm install && npm run build
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    if "--hud" in sys.argv:
        sys.argv.remove("--hud")
        from scripts.run_monitor import main as hud_main
        hud_main()
        return

    ap = argparse.ArgumentParser(description="VisionICU monitor + web dashboard")
    ap.add_argument("--demo", action="store_true",
                    help="run 4 simulated beds (clearly labelled) instead of the camera")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 to allow other machines on the network)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--patient-id", default="bed-1")
    ap.add_argument("--label", default="Bed 1")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--reuse-calibration", action="store_true",
                    help="load this bed's saved chair calibration at startup")
    ap.add_argument("--frontend-dir", default="frontend/dist")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.demo:
        import yaml
        from src.serving.models_check import check_models, format_report
        try:
            with open(f"{args.config_dir}/signal.yaml") as f:
                sig = yaml.safe_load(f) or {}
        except FileNotFoundError:
            sig = {}
        print(format_report(check_models(".", sig)))

    import uvicorn
    from src.serving.api import create_app
    from src.serving.engine import EngineManager

    manager = EngineManager.from_args(demo=args.demo, patient_id=args.patient_id, label=args.label,
                                      config_dir=args.config_dir, fps=args.fps,
                                      reuse_calibration=args.reuse_calibration)
    app = create_app(manager, frontend_dir=args.frontend_dir)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"\n  VisionICU {'(DEMO - simulated data)' if args.demo else ''} -> "
          f"http://{shown}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
