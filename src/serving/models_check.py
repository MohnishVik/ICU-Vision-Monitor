"""
models_check.py — which weight files the system expects, and whether each is in place.

Shown on the System page (/api/models) and printed at startup, so a missing file is
named directly instead of surfacing later as a module that quietly switched off.
Paths follow signal.yaml overrides when present.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def _expected(sig: dict) -> List[tuple]:
    fall = sig.get("fall") or {}
    ln = sig.get("learned") or {}
    pain = sig.get("pain") or {}
    get = lambda d, k, default: (d.get(k) or {}).get("weights", default)  # noqa: E731
    return [
        (fall.get("yolo_weights", "models/public/yolov8n-pose.pt"),
         "Pose: patient tracking, face blur, fall keypoints", True),
        (fall.get("gru_weights_primary", "models/public/fall_gru_pseudo.pt"),
         "Fall detection (primary GRU)", True),
        (fall.get("gru_weights_fallback", "models/public/fall_gru_pseudo_e1.pt"),
         "Fall detection (fallback GRU)", False),
        (pain.get("weights", "models/public/pain_mobilenetv3_synpain.pth"),
         "Pain expression (experimental)", False),
        (get(ln, "rr_lbrd", "models/public/rr_cnn1d_lbrdic.pth"),
         "Breathing, learned: LBRD-IC CNN (experimental)", False),
        (get(ln, "rr_d435", "models/finetuned/rr_cnn1d_d435_finetuned.pth"),
         "Breathing, learned: D435 fine-tuned CNN (experimental)", False),
        (get(ln, "hr_efficientphys", "models/public/hr_efficientphys_ubfc.pth"),
         "Heart rate, learned: EfficientPhys (experimental)", False),
    ]


def check_models(root: str | Path = ".", signal_cfg: Optional[dict] = None) -> List[dict]:
    root = Path(root)
    rows = []
    for rel, used_by, required in _expected(signal_cfg or {}):
        p = root / rel
        rows.append({"file": str(rel), "used_by": used_by, "required": required,
                     "present": p.is_file(), "size_bytes": p.stat().st_size if p.is_file() else None})
    return rows


def format_report(rows: List[dict]) -> str:
    lines = ["Model files:"]
    for r in rows:
        mark = "ok     " if r["present"] else ("MISSING" if r["required"] else "absent ")
        lines.append(f"  [{mark}] {r['file']:<48} {r['used_by']}")
    return "\n".join(lines)
