"""test_sessions.py — numbers-only session recording and the source-vs-reference comparison."""
import csv
import json
import time

import pytest

from src.evaluation.compare import aggregate, load_session, score_session
from src.serving.session_log import SessionRecorder

SRCS = ("rr.depth", "rr.cnn_lbrd", "hr.pos")


def _sample(t, **vals):
    return {"t": t, "rr": vals.get("rr.depth"), "hr": vals.get("hr.pos"), "ews": 0, "pain": None,
            "sources": {s: vals.get(s) for s in SRCS},
            "sqi": {s: (0.8 if vals.get(s) is not None else None) for s in SRCS},
            "unfiltered": {s: vals.get(s) for s in SRCS}}


def test_recorder_writes_meta_estimates_and_references(tmp_path):
    rec = SessionRecorder(tmp_path, patient_id="bed-1", label="paced 15", condition="paced",
                          sources=SRCS, reference={"rr": 15.0}, simulated=False, fuse_learned=False)
    t0 = rec.started
    for k in range(3):
        rec.write(_sample(t0 + k, **{"rr.depth": 15.5, "hr.pos": 70.0}))
    rec.add_reference(rr=20.0, note="switched metronome", t=t0 + 2.5)
    info = rec.stop()
    d = tmp_path / info["id"]
    meta = json.loads((d / "meta.json").read_text())
    assert meta["label"] == "paced 15" and meta["ended"] is not None and meta["rows"] == 3
    rows = list(csv.DictReader((d / "estimates.csv").open()))
    assert len(rows) == 3 and rows[0]["rr.depth"] == "15.5" and rows[0]["rr.cnn_lbrd"] == ""
    refs = list(csv.DictReader((d / "references.csv").open()))
    assert [r["rr"] for r in refs] == ["15.0", "20.0"] and refs[1]["note"] == "switched metronome"


def test_score_reports_mae_and_coverage_after_warmup_and_settling(tmp_path):
    rec = SessionRecorder(tmp_path, patient_id="bed-1", label="t", condition="paced",
                          sources=SRCS, reference={"rr": 15.0, "hr": 70.0}, simulated=False,
                          fuse_learned=False)
    t0 = rec.started
    rec.add_reference(rr=20.0, t=t0 + 50)
    for k in range(100):
        ref = 15.0 if k < 50 else 20.0
        lbrd = (ref - 2.5) if k % 2 == 0 else None
        rec.write(_sample(t0 + k, **{"rr.depth": ref + 1.0, "rr.cnn_lbrd": lbrd, "hr.pos": 73.0}))
    info = rec.stop()
    sess = load_session(tmp_path / info["id"])
    rep = score_session(sess, warmup_s=30, settle_s=20)
    rr = rep["rr"]
    # scored seconds: 30..49 (20) + 70..99 (30) = 50
    assert rr["rr.depth"]["n_scored"] == 50
    assert rr["rr.depth"]["mae"] == pytest.approx(1.0) and rr["rr.depth"]["coverage"] == 1.0
    assert rr["rr.depth"]["bias"] == pytest.approx(1.0)
    assert rr["rr.cnn_lbrd"]["coverage"] == pytest.approx(0.5)
    assert rr["rr.cnn_lbrd"]["mae"] == pytest.approx(2.5)
    assert rr["rr.cnn_lbrd"]["bias"] == pytest.approx(-2.5)
    assert rr["rr.cnn_lbrd"]["within_tol"] == 0.0           # RR tolerance is +-2 BrPM
    assert rr["rr.depth"]["within_tol"] == 1.0
    assert rep["hr"]["hr.pos"]["mae"] == pytest.approx(3.0)
    assert rep["hr"]["hr.pos"]["within_tol"] == 1.0          # HR tolerance is +-5 BPM


def test_aggregate_pools_seconds_across_sessions(tmp_path):
    reps = []
    for err in (1.0, 3.0):
        rec = SessionRecorder(tmp_path, patient_id="bed-1", label=f"e{err}", condition="rest",
                              sources=SRCS, reference={"rr": 12.0}, simulated=False, fuse_learned=False)
        for k in range(40):
            rec.write(_sample(rec.started + k, **{"rr.depth": 12.0 + err}))
        reps.append(score_session(load_session(tmp_path / rec.stop()["id"]), warmup_s=30, settle_s=0))
        time.sleep(1.01)                                    # distinct session ids
    agg = aggregate(reps)
    assert agg["rr"]["rr.depth"]["n_scored"] == 20 and agg["rr"]["rr.depth"]["mae"] == pytest.approx(2.0)


def test_session_api_roundtrip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.serving.api import create_app
    from src.serving.engine import EngineManager
    monkeypatch.setenv("VISIONICU_SESSIONS_DIR", str(tmp_path))
    manager = EngineManager.from_args(demo=True)
    with TestClient(create_app(manager, frontend_dir="nope")) as c:
        r = c.post("/api/patients/bed-1/session", json={"label": "demo check", "condition": "rest",
                                                         "reference": {"rr": 15, "hr": 74}})
        assert r.status_code == 200 and r.json()["active"] is True
        assert c.post("/api/patients/bed-1/session", json={"label": "again"}).status_code == 409
        assert c.post("/api/patients/bed-1/session/reference", json={"hr": 75}).status_code == 200
        time.sleep(2.5)
        stop = c.delete("/api/patients/bed-1/session").json()
        assert stop["active"] is False and stop["rows"] >= 1
        sid = stop["id"]
        listed = c.get("/api/sessions").json()
        assert any(s["id"] == sid for s in listed)
        rep = c.get(f"/api/sessions/{sid}/report?warmup_s=0&settle_s=0").json()
        assert "rr.depth" in rep["rr"] and "hr.efficientphys" in rep["hr"]
        json.dumps(rep, allow_nan=False)
        assert c.get("/api/sessions/..%2F..%2Fetc/report").status_code in (400, 404)
        assert c.get("/api/sessions/no_such_session/report").status_code == 404


def test_two_sessions_in_the_same_second_get_distinct_ids(tmp_path):
    a = SessionRecorder(tmp_path, patient_id="bed-1", label="same", sources=SRCS)
    b = SessionRecorder(tmp_path, patient_id="bed-1", label="same", sources=SRCS)
    assert a.id != b.id and (tmp_path / b.id / "meta.json").exists()
    a.stop(), b.stop()
