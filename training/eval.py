#!/usr/bin/env python3
"""
eval.py — score weights on held-out people, side by side with what the monitor already does.

  python training/eval.py --module rr_d435 --weights models/finetuned/rr_d435_20261028_1530.pth
  python training/eval.py --module fall --weights models/public/fall_gru_pseudo.pt --subjects S11 S12

By default it uses the test subjects in data/splits/<module>_splits.json (written by
finetune.py), so the people scored are ones the model never trained on.

Breathing: for every labelled 10 s window it reports the network's MAE, RMSE, bias and share
within +-2 br/min, and, for the same seconds, the signal-processing paths recorded in
estimates.csv (rr.depth and the fused rr) with their MAE AND coverage. A learned model should
join fusion (learned.fuse in config/signal.yaml) only if it beats them on both.
Falls: sensitivity, specificity, precision and F1 at the threshold the monitor uses (0.5).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import Windows, build, find_capture_sessions   # noqa: E402
from training.finetune import MODULES, load_model, load_signal_cfg, predict, score   # noqa: E402


def _estimates(session_dirs: List[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """session id -> {"t": seconds, "rr.depth": ..., "rr": fused} from estimates.csv."""
    out = {}
    for d in session_dirs:
        meta = json.loads((d / "meta.json").read_text())
        with open(d / "estimates.csv", newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))

        def col(name):
            return np.asarray([float(r[name]) if r.get(name) not in (None, "") else np.nan for r in rows])
        out[meta["id"]] = {"t": col("t"), "rr.depth": col("rr.depth") if rows and "rr.depth" in rows[0]
                           else np.full(len(rows), np.nan), "rr.fused": col("rr")}
    return out


def dsp_baseline(w: Windows, est: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict]:
    """Score the recorded rr.depth and fused rr at each window's end second."""
    res = {}
    for src in ("rr.depth", "rr.fused"):
        vals, labels, n = [], [], 0
        for sess, t_end, y in zip(w.session, w.t_end, w.y):
            e = est.get(sess)
            if e is None or not len(e["t"]):
                continue
            i = int(np.argmin(np.abs(e["t"] - t_end)))
            if abs(e["t"][i] - t_end) > 1.5:
                continue
            n += 1
            if np.isfinite(e[src][i]):
                vals.append(e[src][i])
                labels.append(y)
        s = score("rr", np.asarray(vals), np.asarray(labels)) if vals else {"n": 0}
        s["coverage"] = len(vals) / n if n else None
        res[src] = s
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, choices=MODULES)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--subjects", nargs="*", help="people to score (default: test split)")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--signal-yaml", default="config/signal.yaml")
    ap.add_argument("--threshold", type=float, default=0.5, help="fall probability threshold")
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("--out", help="write the report as JSON here")
    a = ap.parse_args(argv)

    sig = load_signal_cfg(a.signal_yaml)
    data = build(a.module, a.sessions, signal_cfg=sig, include_simulated=a.include_simulated)
    subjects = a.subjects
    if not subjects:
        sp = Path(a.splits_dir) / f"{a.module}_splits.json"
        if not sp.exists():
            print(f"No --subjects given and no {sp}. Run finetune.py first or name subjects.", file=sys.stderr)
            return 1
        subjects = json.loads(sp.read_text()).get("test", [])
    w = data.select(subjects)
    if not len(w):
        print(f"No labelled windows for subjects {subjects}.", file=sys.stderr)
        return 1
    model, _ = load_model(a.module, Path(a.weights))
    pred = predict(a.module, model, w)
    report = {"module": a.module, "weights": a.weights, "subjects": sorted(set(w.subject)),
              "windows": len(w), "model": score(a.module, pred, w.y, a.threshold)}
    if a.module != "fall":
        report["signal_processing_same_seconds"] = dsp_baseline(w, _estimates(find_capture_sessions(a.sessions)))
        per = {}
        for s in sorted(set(w.subject)):
            m = np.asarray([x == s for x in w.subject])
            per[s] = score(a.module, pred[m], w.y[m])
        report["per_subject"] = per
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
