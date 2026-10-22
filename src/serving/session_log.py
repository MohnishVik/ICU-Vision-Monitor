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

No images, video or face crops are written. The directory is git-ignored (data/).
Score sessions with scripts/compare_sessions.py or the Compare page.
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
                 simulated: bool = False, fuse_learned: bool = False, mode: str = "camera"):
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
        self.error: Optional[str] = None
        self.reference = {"rr": None, "hr": None}
        self.meta = {"id": self.id, "patient_id": patient_id, "label": label, "condition": condition,
                     "notes": notes, "started": self.started, "ended": None, "rows": 0,
                     "simulated": bool(simulated), "mode": mode, "fuse_learned": bool(fuse_learned),
                     "sources": list(self.sources)}
        self._write_meta()

        self._est_f = open(self.dir / "estimates.csv", "w", newline="")
        self._est = csv.writer(self._est_f)
        self._est.writerow(["t", "rr", "hr", "ews", "pain"]
                           + [c for s in self.sources for c in (s, f"{s}_sqi", f"{s}_unfiltered")])
        self._ref_f = open(self.dir / "references.csv", "w", newline="")
        self._ref = csv.writer(self._ref_f)
        self._ref.writerow(["t", "rr", "hr", "note"])
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
            self._ref.writerow([repr(t), _num(rr), _num(hr), note.replace("\n", " ")[:200]])
            self._ref_f.flush()
            if rr is not None:
                self.reference["rr"] = float(rr)
            if hr is not None:
                self.reference["hr"] = float(hr)
        return self.status()

    def status(self) -> dict:
        return {"active": not self._est_f.closed, "id": self.id, "label": self.meta["label"],
                "condition": self.meta["condition"], "started": self.started, "rows": self.rows,
                "elapsed_s": time.time() - self.started, "reference": dict(self.reference),
                "error": self.error}

    def stop(self) -> dict:
        with self._lock:
            for f in (self._est_f, self._ref_f):
                if not f.closed:
                    f.close()
            self.meta["ended"] = time.time()
            self.meta["rows"] = self.rows
            self._write_meta()
        return {**self.status(), "active": False}
