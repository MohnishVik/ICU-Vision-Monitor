#!/usr/bin/env python3
"""
eval_weights.py — does every weight file load, and does it give sensible answers?

  python scripts/eval_weights.py            # all models, prints a table
  python scripts/eval_weights.py --json reports/weights.json

Each file is loaded exactly as the monitor loads it, then probed with an input whose right
answer is known:

  yolov8n-pose        ultralytics' sample photos (bus.jpg, zidane.jpg): people found, and
                      how many have >= 10 confident keypoints
  fall_gru (x2)       a simulated person sitting still vs falling to the floor, replayed
                      through the live 62-D pose feature extractor -> fall probability
  rr_cnn1d_lbrdic     synthetic breathing at 6-30 br/min, prepared like the live input
  rr_cnn1d_d435       (whole-frame green / chest depth) -> rate error
  hr_efficientphys    a real face crop whose skin colour pulses at 60-120 bpm -> rate error
  pain_mobilenetv3    neutral faces from the sample photos -> raw score (calm faces should
                      score low; the monitor reads pain against each patient's baseline)

These probes check that a model works and how it is biased; they are not clinical accuracy.
Clinical accuracy comes from recorded sessions scored on the Compare page
(docs/RECORDING_AND_FINETUNING.md).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

RATES_RR = (6, 10, 15, 20, 25, 30)
RATES_HR = (60, 80, 100, 120)


def _sig_cfg() -> dict:
    return yaml.safe_load(open(ROOT / "config/signal.yaml", encoding="utf-8")) or {}


def _assets() -> Path:
    import ultralytics
    return Path(ultralytics.__file__).parent / "assets"


def _faces():
    """Face crops (BGR) from ultralytics' sample photos, found with the project's pose model."""
    import cv2
    from ultralytics import YOLO
    pose = YOLO(str(ROOT / "models/public/yolov8n-pose.pt"))
    out = []
    for name in ("zidane.jpg", "bus.jpg"):
        img = cv2.imread(str(_assets() / name))
        r = pose.predict(img, verbose=False)[0]
        for k, c in zip(r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()):
            pts = k[:5][c[:5] > 0.5]
            if len(pts) < 3:
                continue
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            w = max(30.0, float(np.ptp(pts[:, 0]))) * 1.45
            crop = img[max(0, int(cy - 0.7 * w)):int(cy + 0.6 * w), max(0, int(cx - w / 2)):int(cx + w / 2)]
            if crop.size and min(crop.shape[:2]) >= 40:
                out.append(crop)
    return out


def check_pose() -> dict:
    import cv2
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "models/public/yolov8n-pose.pt"))
    res = {}
    for name in ("bus.jpg", "zidane.jpg"):
        r = m.predict(cv2.imread(str(_assets() / name)), verbose=False)[0]
        conf = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17))
        res[name] = {"people": int(len(r.boxes)), "with_10plus_keypoints": int((conf > 0.5).sum(1).__ge__(10).sum())}
    ok = res["bus.jpg"]["people"] >= 3 and res["zidane.jpg"]["people"] >= 2
    return {"ok": ok, **res}


def _pose_frames(kind: str, seconds: float = 4.0, fs: float = 30.0) -> dict:
    """frames.csv-style columns for a seated person, optionally falling at 1.5 s."""
    up = np.array([[320, 100], [312, 92], [328, 92], [300, 96], [340, 96], [280, 160], [360, 160],
                   [265, 230], [375, 230], [262, 290], [378, 290], [295, 300], [345, 300],
                   [295, 380], [345, 380], [295, 450], [345, 450]], float)
    n = int(seconds * fs)
    cols = {k: np.zeros(n) for k in ("t", "t_mono", "img_h", "img_w", "has_person",
                                      "box_x1", "box_y1", "box_x2", "box_y2")}
    for j in range(17):
        for a in ("x", "y", "c"):
            cols[f"kp{j}_{a}"] = np.zeros(n)
    rng = np.random.default_rng(0)
    for i in range(n):
        ts = i / fs
        p = up + rng.normal(0, 1.5, up.shape)
        if kind == "fall" and ts >= 1.5:
            k = min(1.0, (ts - 1.5) / 0.6)
            ang = k * math.pi / 2                                # body rotates to horizontal
            cx, cy = 320.0, 450.0
            rel = p - [cx, cy]
            rot = np.column_stack([rel[:, 0] * math.cos(ang) - rel[:, 1] * math.sin(ang),
                                   rel[:, 0] * math.sin(ang) + rel[:, 1] * math.cos(ang)])
            p = rot + [cx, cy + 10 * k]
        cols["t"][i] = cols["t_mono"][i] = ts
        cols["img_h"][i], cols["img_w"][i], cols["has_person"][i] = 480, 640, 1
        cols["box_x1"][i], cols["box_y1"][i] = p[:, 0].min() - 10, p[:, 1].min() - 10
        cols["box_x2"][i], cols["box_y2"][i] = p[:, 0].max() + 10, p[:, 1].max() + 10
        for j in range(17):
            cols[f"kp{j}_x"][i], cols[f"kp{j}_y"][i], cols[f"kp{j}_c"][i] = p[j, 0], p[j, 1], 0.9
    return cols


def check_fall(path: Path) -> dict:
    import torch
    from src.modules.fall import _load_gru
    from training.dataset import fall_features
    model, _ = _load_gru(path, path, torch.device("cpu"))
    sig = _sig_cfg()

    def prob(kind):
        f = fall_features(_pose_frames(kind), sig)
        best = 0.0
        for end in range(31, len(f), 4):
            with torch.no_grad():
                best = max(best, float(torch.sigmoid(model(torch.from_numpy(f[end - 31:end + 1][None]).float()))))
        return best
    still, fall = prob("still"), prob("fall")
    return {"ok": fall > still, "p_fall_when_still": round(still, 3), "p_fall_when_falling": round(fall, 3)}


def check_rr_cnn(path: Path, n: int, detrend: bool) -> dict:
    import torch
    from src.modules.learned import Resp1DCNN, rr_net_input
    net = Resp1DCNN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    net.eval()
    rng = np.random.default_rng(0)
    preds = {}
    for rate in RATES_RR:
        ys = []
        for trial in range(5):
            t = np.arange(n) * 10.0 / n
            x = np.sin(2 * np.pi * rate / 60 * t + rng.uniform(0, 6.3)) + rng.normal(0, 0.15, n) + 0.05 * t
            z = rr_net_input(x, detrend)
            with torch.no_grad():
                ys.append(float(net(torch.from_numpy(z.astype(np.float32))[None, None]).item()))
        preds[rate] = round(float(np.mean(ys)), 1)
    err = [abs(preds[r] - r) for r in RATES_RR]
    return {"ok": True, "predicted_by_true_rate": preds, "mae_br_min": round(float(np.mean(err)), 2),
            "spread_of_predictions": round(max(preds.values()) - min(preds.values()), 1)}


def check_efficientphys(faces) -> dict:
    from src.modules.learned import LearnedConfig, LearnedHRModule
    from src.types import FrameBundle, ROIBundle
    cfg = LearnedConfig.from_dict(_sig_cfg())
    face = faces[0]
    res = {}
    for bpm in RATES_HR:
        mod = LearnedHRModule(cfg, background=False)
        out = None
        rng = np.random.default_rng(bpm)
        for i in range(int(25 * 30)):
            ts = i / 30.0
            pulse = 1.0 + 0.012 * math.sin(2 * math.pi * bpm / 60 * ts)        # ~1 % skin colour change
            f = face.astype(np.float32).copy()
            f[:, :, 1] *= pulse                                                # strongest in green
            f[:, :, 2] *= 1.0 + 0.5 * (pulse - 1.0)
            f += rng.normal(0, 1.0, f.shape)
            roi = ROIBundle(face_roi=np.clip(f, 0, 255).astype(np.uint8), chest_roi=None, chest_depth=None,
                            abdomen_roi=None, abdomen_depth=None, limb_rois=[], h_centroid=0.5,
                            in_chair_frac=1.0, staff_count=0, lighting="bright", patient_id="p", ts=ts)
            o = mod.process(FrameBundle(rgb=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
                                        ir=None, thermal=None, ts_mono=ts), roi)
            if o:
                out = o["hr.efficientphys"]
        res[bpm] = None if out is None else round(float(out.meta.get("rate_unfiltered") or float("nan")), 1)
    err = [abs(res[b] - b) for b in RATES_HR if res[b] is not None and res[b] == res[b]]
    return {"ok": bool(err), "predicted_by_true_bpm": res,
            "mae_bpm": round(float(np.mean(err)), 1) if err else None}


def check_pain(faces) -> dict:
    from PIL import Image
    import cv2
    import torch
    from src.modules.pain import load_pain_module
    mod = load_pain_module(str(ROOT / "config/signal.yaml"))
    scores = []
    for f in faces:
        x = mod.tf(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))[None].to(mod.device)
        with torch.no_grad():
            scores.append(float(torch.sigmoid(mod.net(x)).item()))
    return {"ok": True, "neutral_face_scores": [round(s, 2) for s in scores],
            "model_threshold": mod.cfg.threshold,
            "neutral_faces_over_model_threshold": int(sum(s >= mod.cfg.threshold for s in scores))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args(argv)
    sig = _sig_cfg()
    from src.modules.fall import FallConfig
    from src.modules.learned import LearnedConfig
    fc, lc = FallConfig.from_dict(sig), LearnedConfig.from_dict(sig)
    faces = _faces()
    checks = [
        ("yolov8n-pose.pt", lambda: check_pose()),
        (fc.gru_weights_primary.name, lambda: check_fall(fc.gru_weights_primary)),
        (fc.gru_weights_fallback.name, lambda: check_fall(fc.gru_weights_fallback)),
        (lc.rr_lbrd.weights.name, lambda: check_rr_cnn(lc.rr_lbrd.weights, lc.rr_lbrd.n_samples, False)),
        (lc.rr_d435.weights.name, lambda: check_rr_cnn(lc.rr_d435.weights, lc.rr_d435.n_samples, True)),
        (lc.hr.weights.name, lambda: check_efficientphys(faces)),
        ("pain_mobilenetv3_synpain.pth", lambda: check_pain(faces)),
    ]
    report = {}
    for name, fn in checks:
        t0 = time.time()
        try:
            r = fn()
            r["loads"] = True
        except Exception as exc:  # noqa: BLE001
            r = {"loads": False, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        r["seconds"] = round(time.time() - t0, 1)
        report[name] = r
        print(f"\n{name}\n  " + "\n  ".join(f"{k}: {v}" for k, v in r.items()))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2))
    return 0 if all(r.get("loads") for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
