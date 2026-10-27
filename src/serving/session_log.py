"""
session_log.py — numbers-only recording of a monitoring session, for later comparison.

A session is started from the dashboard (or the API) with a label, a condition and the
reference values you trust for that moment: a metronome rate, breaths you counted over a
minute, a pulse-oximeter reading. References can be updated mid-session.

While a session runs, once per second the pipeline hands over every source's value:

    data/sessions/<bed>_<YYYYmmdd_HHMMSS>_<label>/
        meta.json        label, condition, notes, start/end, mode, whether learned paths were fused
        estimates.csv    t, rr, hr, ews, pain, then per source: value, sqi, unfiltered network output
        references.csv   t, rr, hr, note
        events.csv       t, kind, value, note   (fall, chair exit, talking, CPOT score, ...)
        frames.csv       training capture only: one row per camera frame with the exact inputs
                         the learned models see (chest/abdomen depth means, whole-frame mean
                         green, pose keypoints and box). See FRAME_COLUMNS.

No images, video or face crops are written; frames.csv holds numbers derived from each frame.
The directory is git-ignored (data/). Score sessions with scripts/compare_sessions.py or the
Compare page; build training windows with training/dataset.py.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

SESSIONS_ENV = "VISIONICU_SESSIONS_DIR"


# Per-frame training capture. chest_depth is the plain mean the learned D435 CNN uses;
# chest_depth_body is the body-band mean rr.depth uses; green is the whole-frame mean the
# LBRD-IC CNN uses. Keypoints are in image pixels, as the fall detector receives them.
KP_COLS = [f"kp{i}_{a}" for i in range(17) for a in ("x", "y", "c")]
FRAME_COLUMNS = (["t", "t_mono", "img_h", "img_w", "chest_depth", "chest_depth_body", "abdomen_depth",
                  "green", "has_person", "box_x1", "box_y1", "box_x2", "box_y2"] + KP_COLS)

# Decimals kept per frames.csv column: what each measurement can actually resolve
# (ms timestamps, 0.01 depth unit, 0.001 grey level, 0.1 px, 0.001 confidence). Full float
# precision would double the file (~90 MB/h at 30 fps) for no extra information.
_PRECISION = {"t": 3, "t_mono": 4, "img_h": 0, "img_w": 0, "chest_depth": 2, "chest_depth_body": 2,
              "abdomen_depth": 2, "green": 3, **{c: 3 for c in KP_COLS if c.endswith("_c")}}


def _fixed(v, decimals: int) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{f:.{decimals}f}" if f == f and f not in (float("inf"), float("-inf")) else ""


EVENT_KINDS = ("fall", "chair_exit", "sit_to_stand", "lying_down", "talking", "movement",
               "breath_hold_start", "breath_hold_end", "paradoxical_start", "paradoxical_end",
               "pain_score", "procedure_start", "procedure_end", "other")


def sessions_root() -> Path:
    return Path(os.environ.get(SESSIONS_ENV, "data/sessions"))


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-").lower()
    return s[:40] or "session"


def _num(v) -> str:
    return "" if v is None else repr(float(v))


class SessionRecorder:
    def __init__(self, root: str | Path, patient_id: str, label: str, condition: str = "",
                 sources: Iterable[str] = (), reference: Optional[dict] = None, notes: str = "",
                 simulated: bool = False, fuse_learned: bool = False, mode: str = "camera",
                 capture: bool = False):
        self.started = time.time()
        self.sources = tuple(sources)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started))
        base = f"{_slug(patient_id)}_{stamp}_{_slug(label)}"
        Path(root).mkdir(parents=True, exist_ok=True)
        for n in range(1, 100):                 # same bed, second and label: add a suffix
            self.id = base if n == 1 else f"{base}-{n}"
            self.dir = Path(root) / self.id
            try:
                self.dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        self._lock = threading.Lock()
        self.rows = 0
        self.frames = 0
        self.events = 0
        self.capture = bool(capture)
        self._last_flush = 0.0
        self.error: Optional[str] = None
        self.reference = {"rr": None, "hr": None}
        self.meta = {"id": self.id, "patient_id": patient_id, "label": label, "condition": condition,
                     "notes": notes, "started": self.started, "ended": None, "rows": 0,
                     "simulated": bool(simulated), "mode": mode, "fuse_learned": bool(fuse_learned),
                     "sources": list(self.sources), "capture": self.capture, "frames": 0, "events": 0}
        self._write_meta()

        # explicit utf-8: a free-text note (e.g. "≈15") must not fail on a cp1252 locale
        self._est_f = open(self.dir / "estimates.csv", "w", newline="", encoding="utf-8")
        self._est = csv.writer(self._est_f)
        self._est.writerow(["t", "rr", "hr", "ews", "pain"]
                           + [c for s in self.sources for c in (s, f"{s}_sqi", f"{s}_unfiltered")])
        self._ref_f = open(self.dir / "references.csv", "w", newline="", encoding="utf-8")
        self._ref = csv.writer(self._ref_f)
        self._ref.writerow(["t", "rr", "hr", "note"])
        self._evt_f = open(self.dir / "events.csv", "w", newline="", encoding="utf-8")
        self._evt = csv.writer(self._evt_f)
        self._evt.writerow(["t", "kind", "value", "note"])
        self._frm_f = None
        if self.capture:
            self._frm_f = open(self.dir / "frames.csv", "w", newline="", encoding="utf-8")
            self._frm = csv.writer(self._frm_f)
            self._frm.writerow(FRAME_COLUMNS)
        ref = reference or {}
        if ref.get("rr") is not None or ref.get("hr") is not None:
            self.add_reference(rr=ref.get("rr"), hr=ref.get("hr"), note="initial", t=self.started)
        self._est_f.flush()

    def _write_meta(self) -> None:
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.dir / "meta.json")

    def write(self, sample: dict) -> None:
        """One 1 Hz sample from VitalsPipeline (t, rr, hr, ews, pain, sources, sqi, unfiltered)."""
        src, sqi, unf = sample.get("sources") or {}, sample.get("sqi") or {}, sample.get("unfiltered") or {}
        row = [repr(float(sample["t"])), _num(sample.get("rr")), _num(sample.get("hr")),
               _num(sample.get("ews")), _num(sample.get("pain"))]
        for s in self.sources:
            row += [_num(src.get(s)), _num(sqi.get(s)), _num(unf.get(s))]
        with self._lock:
            if self._est_f.closed:
                return
            self._est.writerow(row)
            self._est_f.flush()
            self.rows += 1

    def add_reference(self, rr: Optional[float] = None, hr: Optional[float] = None, note: str = "",
                      t: Optional[float] = None) -> dict:
        if rr is None and hr is None:
            raise ValueError("Give at least one reference value (rr or hr).")
        t = time.time() if t is None else float(t)
        with self._lock:
            if self._ref_f.closed:
                raise RuntimeError("This session has already been stopped.")
            self._ref.writerow([repr(t), _num(rr), _num(hr), note.replace("\n", " ")[:200]])
            self._ref_f.flush()
            if rr is not None:
                self.reference["rr"] = float(rr)
            if hr is not None:
                self.reference["hr"] = float(hr)
        return self.status()

    def add_event(self, kind: str, value: Optional[float] = None, note: str = "",
                  t: Optional[float] = None) -> dict:
        """Mark a moment: a fall, a chair exit, a CPOT score (value), the start of talking."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"Unknown event '{kind}'. Use one of: {', '.join(EVENT_KINDS)}.")
        t = time.time() if t is None else float(t)
        with self._lock:
            if self._evt_f.closed:
                raise RuntimeError("This session has already been stopped.")
            self._evt.writerow([repr(t), kind, _num(value), note.replace("\n", " ")[:200]])
            self._evt_f.flush()
            self.events += 1
        return self.status()

    def capture_frame(self, frame: dict) -> None:
        """One camera frame's model inputs (keys from FRAME_COLUMNS; missing ones stay empty)."""
        if self._frm_f is None:
            return
        row = [("1" if frame.get(c) else "0") if c == "has_person" else _fixed(frame.get(c), _PRECISION.get(c, 1))
               for c in FRAME_COLUMNS]
        with self._lock:
            if self._frm_f.closed:
                return
            self._frm.writerow(row)
            self.frames += 1
            now = time.monotonic()
            if now - self._last_flush >= 1.0:            # ~30 rows a second: flush once a second
                self._frm_f.flush()
                self._last_flush = now

    def status(self) -> dict:
        return {"active": not self._est_f.closed, "id": self.id, "label": self.meta["label"],
                "condition": self.meta["condition"], "started": self.started, "rows": self.rows,
                "elapsed_s": time.time() - self.started, "reference": dict(self.reference),
                "capture": self.capture, "frames": self.frames, "events": self.events,
                "error": self.error}

    def stop(self) -> dict:
        with self._lock:
            for f in (self._est_f, self._ref_f, self._evt_f, self._frm_f):
                if f is not None and not f.closed:
                    f.close()
            self.meta["ended"] = time.time()
            self.meta["rows"] = self.rows
            self.meta["frames"] = self.frames
            self.meta["events"] = self.events
            self._write_meta()
        return {**self.status(), "active": False}
