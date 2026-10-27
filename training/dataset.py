"""
dataset.py — turn training-capture sessions into windows the learned models can learn from.

A session recorded with "Capture training data" on holds (src/serving/session_log.py):
  frames.csv      one row per camera frame: chest/abdomen depth, whole-frame green, pose
  references.csv  the true rates you entered (and any imported with scripts/import_reference.py)
  events.csv      marked moments: fall, chair exit, CPOT score, ...
  meta.json       label (starts with the subject code, e.g. "S07_dev_paced12_bright")

Windows are built exactly as the live modules build their input:
  rr_d435   10 s of chest_depth -> 300 samples -> linear detrend -> z-score  (Resp1DCNN)
  rr_lbrd   10 s of green       -> 250 samples ->                  z-score  (Resp1DCNN)
  fall      32 consecutive frames of the 62-D pose feature, replayed through the live
            kinematics + feature extractor (FallGRUClassifier)

Labels
  RR      the reference breathing rate in force for the WHOLE window. Windows inside the
          first warmup_s of a session or within settle_s of a reference change are skipped
          (the subject is still changing pace), as in src/evaluation/compare.py.
  fall    1 if a "fall" event was marked from fall_pre_s before the window to fall_post_s after
          it (press the button as the person lands), 0 if no fall is marked within
          neg_margin_s, otherwise the window is ambiguous and skipped.

One person is one "subject": the code before the first underscore of the session label
(S07, P03). Splits are made by subject, never by window, so the same person cannot be in
both training and testing.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.evaluation.compare import load_session
from src.modules.learned import resample_window, rr_net_input

SUBJECT_RE = re.compile(r"^([A-Za-z]+\d+)[_\-]")

RR_SPECS = {
    # module: (frames.csv column, samples, seconds, detrend) — mirrors src/modules/learned.py
    "rr_d435": ("chest_depth", 300, 10.0, True),
    "rr_lbrd": ("green", 250, 10.0, False),
}


@dataclass
class Windows:
    """X: (N, ...) float32 inputs, y: (N,) float32 labels, subject/session per window."""
    X: np.ndarray
    y: np.ndarray
    subject: List[str] = field(default_factory=list)
    session: List[str] = field(default_factory=list)
    t_end: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def select(self, subjects: Iterable[str]) -> "Windows":
        keep = set(subjects)
        idx = [i for i, s in enumerate(self.subject) if s in keep]
        return Windows(self.X[idx], self.y[idx], [self.subject[i] for i in idx],
                       [self.session[i] for i in idx], [self.t_end[i] for i in idx])

    @staticmethod
    def concat(parts: List["Windows"], shape: tuple) -> "Windows":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Windows(np.zeros((0,) + shape, np.float32), np.zeros(0, np.float32))
        return Windows(np.concatenate([p.X for p in parts]), np.concatenate([p.y for p in parts]),
                       sum((p.subject for p in parts), []), sum((p.session for p in parts), []),
                       sum((p.t_end for p in parts), []))


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def subject_of(meta: dict) -> str:
    """Subject code from the session label ("S07_dev_..." -> "S07"); else the session id."""
    m = SUBJECT_RE.match(str(meta.get("label") or ""))
    return m.group(1).upper() if m else str(meta.get("id"))


def find_capture_sessions(root: str | Path) -> List[Path]:
    """Session folders under root that have a frames.csv (training capture was on)."""
    root = Path(root)
    if (root / "frames.csv").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/frames.csv"))


def load_frames(session_dir: str | Path) -> Dict[str, np.ndarray]:
    """frames.csv as float columns (empty cells -> NaN), sorted by t_mono."""
    with open(Path(session_dir) / "frames.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) if v not in ("", None) else math.nan for v in r] for r in reader if r]
    a = np.asarray(rows, dtype=np.float64).reshape(-1, len(header))
    order = np.argsort(a[:, header.index("t_mono")], kind="stable") if len(a) else np.arange(0)
    return {h: a[order, i] for i, h in enumerate(header)}


def load_events(session_dir: str | Path) -> List[dict]:
    p = Path(session_dir) / "events.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        out = []
        for r in csv.DictReader(f):
            try:
                out.append({"t": float(r["t"]), "kind": r["kind"],
                            "value": float(r["value"]) if r.get("value") else None, "note": r.get("note", "")})
            except (TypeError, ValueError, KeyError):
                continue
    return sorted(out, key=lambda e: e["t"])


def _ref_at(refs: List[dict], vital: str, t: float):
    """(value in force at t, time it took effect) from sorted reference rows."""
    val, since = None, None
    for r in refs:
        if r["t"] > t:
            break
        v = r.get(vital)
        if v is not None and v != val:
            val, since = v, r["t"]
    return val, since


# ─────────────────────────────────────────────────────────────────────────────
# Breathing windows
# ─────────────────────────────────────────────────────────────────────────────

def rr_windows(session_dir: str | Path, module: str = "rr_d435", step_s: float = 1.0,
               warmup_s: float = 30.0, settle_s: float = 20.0, max_gap_s: float = 0.5) -> Windows:
    """Labelled Resp1DCNN windows from one capture session."""
    col, n, win_s, detrend = RR_SPECS[module]
    sess = load_session(session_dir)
    meta, refs = sess["meta"], sess["refs"]
    fr = load_frames(session_dir)
    subj = subject_of(meta)
    empty = Windows(np.zeros((0, 1, n), np.float32), np.zeros(0, np.float32))
    if len(fr.get("t", [])) < 4:
        return empty
    t_wall, t_mono, v = fr["t"], fr["t_mono"], fr[col]
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return empty
    offset = float(np.median(t_wall - t_mono))              # wall = mono + offset
    tm, vv = t_mono[ok], v[ok]
    start = float(meta["started"])
    X, y, ends = [], [], []
    t_first, t_last = tm[0] + win_s, tm[-1]
    for end_m in np.arange(t_first, t_last + 1e-9, step_s):
        lo = end_m - win_s
        sel = (tm >= lo - 0.2) & (tm <= end_m + 1e-9)
        ts = tm[sel]
        if len(ts) < 4 or ts[0] > lo + max_gap_s or end_m - ts[-1] > max_gap_s \
                or np.max(np.diff(ts)) > max_gap_s:
            continue                                          # gap in the signal
        end_w, start_w = end_m + offset, lo + offset
        if start_w < start + warmup_s:
            continue
        rate, since = _ref_at(refs, "rr", end_w)
        rate0, _ = _ref_at(refs, "rr", start_w)
        if rate is None or rate0 != rate or since is None or start_w < since + settle_s:
            continue                                          # no label, or pace still changing
        x, _ = resample_window(ts, vv[sel], n, win_s)
        z = rr_net_input(x, detrend)
        if z is None:
            continue
        X.append(z.astype(np.float32)[None])
        y.append(float(rate))
        ends.append(end_w)
    if not X:
        return empty
    return Windows(np.stack(X), np.asarray(y, np.float32), [subj] * len(y), [meta["id"]] * len(y), ends)


# ─────────────────────────────────────────────────────────────────────────────
# Fall windows
# ─────────────────────────────────────────────────────────────────────────────

def fall_features(fr: Dict[str, np.ndarray], signal_cfg: Optional[dict] = None) -> np.ndarray:
    """Replay captured pose through the live kinematics + 62-D feature extractor."""
    from src.modules.fall import FallConfig, _Features62D, _Kinematics
    cfg = FallConfig.from_dict(signal_cfg or {})
    kin, feat = _Kinematics(cfg.kin), _Features62D()
    out = np.zeros((len(fr["t_mono"]), 62), np.float32)
    for i in range(len(out)):
        has = fr["has_person"][i] > 0.5 and np.isfinite(fr["box_x1"][i])
        kpts = confs = bbox = None
        if has:
            kpts = np.column_stack([[fr[f"kp{j}_x"][i] for j in range(17)],
                                    [fr[f"kp{j}_y"][i] for j in range(17)]]).astype(np.float32)
            confs = np.asarray([fr[f"kp{j}_c"][i] for j in range(17)], np.float32)
            bbox = np.asarray([fr["box_x1"][i], fr["box_y1"][i], fr["box_x2"][i], fr["box_y2"][i]], np.float32)
        hw = (int(fr["img_h"][i]) if np.isfinite(fr["img_h"][i]) else 480,
              int(fr["img_w"][i]) if np.isfinite(fr["img_w"][i]) else 640)
        data = kin.analyze(kpts, confs, bbox, hw, float(fr["t_mono"][i]))
        out[i] = feat.extract(hw, data)
    return out


def fall_windows(session_dir: str | Path, window: int = 32, stride: int = 4,
                 fall_pre_s: float = 1.5, fall_post_s: float = 2.0, neg_margin_s: float = 5.0,
                 signal_cfg: Optional[dict] = None) -> Windows:
    """Labelled FallGRUClassifier windows from one capture session."""
    meta = json.loads((Path(session_dir) / "meta.json").read_text())
    fr = load_frames(session_dir)
    empty = Windows(np.zeros((0, window, 62), np.float32), np.zeros(0, np.float32))
    if len(fr.get("t", [])) < window:
        return empty
    falls = [e["t"] for e in load_events(session_dir) if e["kind"] == "fall"]
    feats = fall_features(fr, signal_cfg)
    t = fr["t"]
    subj = subject_of(meta)
    X, y, ends = [], [], []
    for end in range(window - 1, len(t), stride):
        t0, t1 = t[end - window + 1], t[end]
        pos = any(t0 - fall_pre_s <= f <= t1 + fall_post_s for f in falls)
        near = any(t0 - neg_margin_s <= f <= t1 + neg_margin_s for f in falls)
        if not pos and near:
            continue                                          # too close to call either way
        X.append(feats[end - window + 1:end + 1])
        y.append(1.0 if pos else 0.0)
        ends.append(float(t1))
    if not X:
        return empty
    return Windows(np.stack(X).astype(np.float32), np.asarray(y, np.float32),
                   [subj] * len(y), [meta["id"]] * len(y), ends)


# ─────────────────────────────────────────────────────────────────────────────
# Whole dataset + subject split
# ─────────────────────────────────────────────────────────────────────────────

def build(module: str, root: str | Path, signal_cfg: Optional[dict] = None, **kw) -> Windows:
    """All windows for one module from every capture session under root."""
    parts = []
    for d in find_capture_sessions(root):
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("simulated") and not kw.get("include_simulated", False):
            continue
        if module in RR_SPECS:
            parts.append(rr_windows(d, module, **{k: v for k, v in kw.items()
                                                  if k in ("step_s", "warmup_s", "settle_s", "max_gap_s")}))
        elif module == "fall":
            parts.append(fall_windows(d, signal_cfg=signal_cfg, **{k: v for k, v in kw.items()
                                                                   if k in ("window", "stride")}))
        else:
            raise ValueError(f"Unknown module '{module}'. Use rr_d435, rr_lbrd or fall.")
    shape = (1, RR_SPECS[module][1]) if module in RR_SPECS else (32, 62)
    return Windows.concat(parts, shape)


def split_subjects(subjects: Iterable[str], val_frac: float = 0.2, test_frac: float = 0.2,
                   seed: int = 0) -> Dict[str, List[str]]:
    """Shuffle subjects (not windows) into train / val / test. Needs >= 3 subjects for all three."""
    subs = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    if n < 3:
        return {"train": subs, "val": [], "test": []}
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    return {"test": sorted(subs[:n_test]), "val": sorted(subs[n_test:n_test + n_val]),
            "train": sorted(subs[n_test + n_val:])}
