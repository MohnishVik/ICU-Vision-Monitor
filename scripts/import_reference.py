#!/usr/bin/env python3
"""
import_reference.py — bring reference readings from another device into a recorded session.

Two kinds of reference file are understood (CSV, header row required):

  Spot values   a time column plus any of hr, rr, spo2. Typical sources: a pulse oximeter's
                PC export, a bedside monitor's trend export, or values typed from the monitor
                once a minute. They are added to the session's references.csv, so the Compare
                page and scripts/compare_sessions.py score against them directly.
  Waveform      a time column plus ppg (or resp) sampled continuously (>= 25 Hz), e.g. a
                logging PPG sensor. Saved as ppg.csv / resp.csv next to the session for
                beat- or breath-level labels when fine-tuning.

Time column (name: t, time, timestamp, datetime or clock), any of:
  unix seconds (1793124605.2), ISO date-time (2026-10-28T14:03:05), or clock time HH:MM:SS[.f]
  together with --date YYYY-MM-DD.

Devices rarely share a clock. Measure the offset once (start a stopwatch on both, or clap in
front of the camera and note the device time) and pass --offset-s; it is ADDED to the device
times. Rows outside the session's start-end are dropped.

  python scripts/import_reference.py data/sessions/<id> pulseox.csv --source pulse_ox
  python scripts/import_reference.py data/sessions/<id> monitor.csv --date 2026-10-28 --offset-s -3.5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

TIME_COLS = ("t", "time", "timestamp", "datetime", "clock")
SPOT_COLS = ("hr", "rr", "spo2")
WAVE_COLS = ("ppg", "resp")


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_time(text: str, date: Optional[str]) -> float:
    """Unix seconds from a unix number, an ISO date-time, or HH:MM:SS with --date."""
    s = (text or "").strip()
    v = _num(s)
    if v is not None and v > 1e8:
        return v
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        pass
    if date:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(f"{date} {s}", fmt).timestamp()
            except ValueError:
                continue
    raise ValueError(f"Cannot read time '{text}'. Use unix seconds, ISO date-time, or HH:MM:SS with --date.")


def import_reference(session_dir: str | Path, ref_csv: str | Path, source: str = "device",
                     offset_s: float = 0.0, date: Optional[str] = None) -> dict:
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text())
    start = float(meta["started"])
    end = float(meta["ended"]) if meta.get("ended") else float("inf")

    with open(ref_csv, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        tcol = next((cols[c] for c in TIME_COLS if c in cols), None)
        if tcol is None:
            raise ValueError(f"No time column; expected one of {', '.join(TIME_COLS)}.")
        spot = [c for c in SPOT_COLS if c in cols]
        wave = [c for c in WAVE_COLS if c in cols]
        if not spot and not wave:
            raise ValueError(f"No value columns; expected any of {', '.join(SPOT_COLS + WAVE_COLS)}.")
        rows = []
        for r in reader:
            t = parse_time(r[tcol], date) + offset_s
            if start <= t <= end:
                rows.append({"t": t, **{c: _num(r[cols[c]]) for c in spot + wave}})

    summary = {"session": meta.get("id"), "kept_rows": len(rows), "spot": spot, "wave": wave}
    if spot:
        refs = d / "references.csv"
        backup = d / "references.manual.csv"
        if not backup.exists():
            shutil.copyfile(refs, backup)              # keep what was typed during the session
        added = 0
        with open(refs, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                if r.get("hr") is None and r.get("rr") is None:
                    continue
                note = f"import:{source}" + (f" spo2={r['spo2']:g}" if r.get("spo2") is not None else "")
                w.writerow([repr(r["t"]), "" if r.get("rr") is None else repr(r["rr"]),
                            "" if r.get("hr") is None else repr(r["hr"]), note])
                added += 1
        summary["references_added"] = added
    for c in wave:
        out = d / f"{c}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", c, "source"])
            for r in rows:
                if r.get(c) is not None:
                    w.writerow([repr(r["t"]), repr(r[c]), source])
        summary[f"{c}_file"] = str(out)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("reference_csv")
    ap.add_argument("--source", default="device", help="label stored with each row, e.g. pulse_ox")
    ap.add_argument("--offset-s", type=float, default=0.0, help="seconds added to device times")
    ap.add_argument("--date", help="YYYY-MM-DD for clock-only times")
    a = ap.parse_args(argv)
    try:
        s = import_reference(a.session_dir, a.reference_csv, a.source, a.offset_s, a.date)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(s, indent=2))
    if s["kept_rows"] == 0:
        print("No rows fell inside the session's time span: check --date / --offset-s.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
