"""
compare.py — score every recorded source against the reference values.

For each second of a session the reference in force is the latest reference entry at or
before that second. Seconds are left out of scoring when
  - they fall in the first `warmup_s` of the session (estimators are still filling), or
  - they fall within `settle_s` after a reference CHANGE for that vital (the subject is
    still changing pace and the windows still hold the old rate).

Per source (rr.fused, rr.depth, rr.thermal, rr.cnn_lbrd, rr.cnn_d435; hr.fused, hr.pos,
hr.efficientphys) we report, always together:
  n_scored     seconds with a reference
  coverage     fraction of those seconds where the source gave a value
  mae, rmse, bias (mean of estimate - reference), within_tol (|err| <= 2 BrPM / 5 BPM),
  pearson_r    over seconds with a value (needs >= 10 and some spread in both)
and for the learned paths the same on the network's unfiltered output (before range and
quality gates): mae_unfiltered, coverage_unfiltered.
Sums are kept so sessions can be pooled (aggregate) second by second.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

TOLERANCE = {"rr": 2.0, "hr": 5.0}


def _f(x: str) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_session(path: str | Path) -> dict:
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    # utf-8 as written by SessionRecorder; "replace" keeps older locale-encoded notes readable.
    # A malformed row (no usable time) is skipped instead of failing the whole report.
    with open(d / "estimates.csv", newline="", encoding="utf-8", errors="replace") as f:
        rows = [{k: _f(v) for k, v in r.items()} for r in csv.DictReader(f)]
    rows = [r for r in rows if r.get("t") is not None]
    refs: List[dict] = []
    with open(d / "references.csv", newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            t = _f(r.get("t"))
            if t is not None:
                refs.append({"t": t, "rr": _f(r.get("rr")), "hr": _f(r.get("hr")),
                             "note": r.get("note") or ""})
    refs.sort(key=lambda r: r["t"])
    return {"meta": meta, "rows": rows, "refs": refs, "path": str(d)}


def _ref_timeline(refs: List[dict], vital: str):
    """[(t, value)] of reference changes for one vital."""
    out = []
    for r in refs:
        v = r[vital]
        if v is not None and (not out or out[-1][1] != v):
            out.append((r["t"], v))
    return out


def _new_acc() -> dict:
    return {"n_scored": 0, "n_valued": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0, "n_within": 0,
            "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0,
            "n_unf": 0, "sum_abs_unf": 0.0}


def _finish(acc: dict, vital: str) -> dict:
    n, k = acc["n_scored"], acc["n_valued"]
    out = {"n_scored": n, "coverage": (k / n) if n else None,
           "mae": acc["sum_abs"] / k if k else None,
           "rmse": math.sqrt(acc["sum_sq"] / k) if k else None,
           "bias": acc["sum_err"] / k if k else None,
           "within_tol": acc["n_within"] / k if k else None, "tolerance": TOLERANCE[vital],
           "pearson_r": None,
           "mae_unfiltered": acc["sum_abs_unf"] / acc["n_unf"] if acc["n_unf"] else None,
           "coverage_unfiltered": (acc["n_unf"] / n) if (n and acc["n_unf"]) else None,
           "acc": acc}
    if k >= 10:
        vx = acc["sxx"] - acc["sx"] ** 2 / k
        vy = acc["syy"] - acc["sy"] ** 2 / k
        if vx > 1e-9 and vy > 1e-9:
            out["pearson_r"] = (acc["sxy"] - acc["sx"] * acc["sy"] / k) / math.sqrt(vx * vy)
    return out


def score_session(sess: dict, warmup_s: float = 30.0, settle_s: float = 20.0) -> dict:
    meta, rows = sess["meta"], sess["rows"]
    started = float(meta["started"])
    sources = list(meta.get("sources") or [])
    report = {"session": {k: meta.get(k) for k in ("id", "patient_id", "label", "condition", "notes",
                                                    "started", "ended", "rows", "simulated", "mode",
                                                    "fuse_learned")},
              "settings": {"warmup_s": warmup_s, "settle_s": settle_s, "tolerance": TOLERANCE}}
    for vital in ("rr", "hr"):
        timeline = _ref_timeline(sess["refs"], vital)
        cols = [(f"{vital}.fused", vital, None)] + [(s, s, f"{s}_unfiltered")
                                                    for s in sources if s.startswith(vital + ".")]
        accs: Dict[str, dict] = {name: _new_acc() for name, _, _ in cols}
        for r in rows:
            t = r["t"]
            if t - started < warmup_s:
                continue
            ref, changed_at = None, None
            for i, (tc, v) in enumerate(timeline):
                if tc <= t:
                    ref, changed_at = v, (tc if i > 0 else None)
            if ref is None or (changed_at is not None and t - changed_at < settle_s):
                continue
            for name, col, ucol in cols:
                a = accs[name]
                a["n_scored"] += 1
                v = r.get(col)
                if v is not None:
                    e = v - ref
                    a["n_valued"] += 1
                    a["sum_abs"] += abs(e)
                    a["sum_sq"] += e * e
                    a["sum_err"] += e
                    a["n_within"] += int(abs(e) <= TOLERANCE[vital] + 1e-9)
                    a["sx"] += v
                    a["sy"] += ref
                    a["sxx"] += v * v
                    a["syy"] += ref * ref
                    a["sxy"] += v * ref
                u = r.get(ucol) if ucol else None
                if u is not None:
                    a["n_unf"] += 1
                    a["sum_abs_unf"] += abs(u - ref)
        report[vital] = {name: _finish(acc, vital) for name, acc in accs.items()}
    return report


def aggregate(reports: List[dict]) -> dict:
    """Pool several session reports second by second (sums add; ratios are recomputed)."""
    out = {"sessions": [r["session"] for r in reports]}
    for vital in ("rr", "hr"):
        pooled: Dict[str, dict] = {}
        for rep in reports:
            for name, stats in (rep.get(vital) or {}).items():
                acc = pooled.setdefault(name, _new_acc())
                for k, v in stats["acc"].items():
                    acc[k] += v
        out[vital] = {name: _finish(acc, vital) for name, acc in pooled.items()}
    return out


def strip_internal(report: dict) -> dict:
    """Report without the pooled sums, for display."""
    out = dict(report)
    for vital in ("rr", "hr"):
        if vital in out:
            out[vital] = {k: {kk: vv for kk, vv in v.items() if kk != "acc"} for k, v in out[vital].items()}
    return out
