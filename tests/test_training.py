"""
test_training.py — training capture -> windows -> fine-tune -> eval, on synthetic sessions.

Three made-up people breathe at known rates in front of a pretend camera; one of them also
"falls" (pose drops to the floor) with the fall marked as an event. The whole chain must run
without a camera or the git-tracked weights, and the windows must be prepared exactly as the
live modules prepare their input.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.modules.learned import Resp1DCNN, resample_window, rr_net_input
from src.serving.session_log import FRAME_COLUMNS, SessionRecorder
from training import dataset
from training.finetune import main as finetune_main
from training.eval import main as eval_main

FS = 30.0
UPRIGHT = np.array([[320, 100], [312, 92], [328, 92], [300, 96], [340, 96], [280, 160], [360, 160],
                    [265, 230], [375, 230], [262, 290], [378, 290], [295, 300], [345, 300],
                    [295, 380], [345, 380], [295, 450], [345, 450]], float)


def _record(root, label, rate_schedule, dur_s, fall_at=None, seed=0):
    """A capture session: chest depth breathing at the scheduled rate, optional fall."""
    rng = np.random.default_rng(seed)
    rec = SessionRecorder(root, "bed-1", label, condition="Paced breathing (metronome)",
                          sources=("rr.depth",), capture=True)
    t0 = rec.started
    phase, last_rate = 0.0, None
    for i in range(int(dur_s * FS)):
        ts = i / FS
        rate = [r for (start, r) in rate_schedule if ts >= start][-1]
        if rate != last_rate:
            rec.add_reference(rr=rate, note="metronome", t=t0 + ts)
            last_rate = rate
        phase += 2 * math.pi * rate / 60.0 / FS
        pose = UPRIGHT.copy()
        if fall_at is not None and ts >= fall_at:
            k = min(1.0, (ts - fall_at) / 0.6)              # falls over 0.6 s, then lies still
            pose[:, 1] = pose[:, 1] * (1 - k) + (430 + (pose[:, 1] - 270) * 0.15) * k
        row = {"t": t0 + ts, "t_mono": 1000.0 + ts, "img_h": 480, "img_w": 640,
               "chest_depth": 1600 + 4 * math.sin(phase) + rng.normal(0, 0.2),
               "chest_depth_body": 1600 + 4 * math.sin(phase), "abdomen_depth": 1650 + 3 * math.sin(phase),
               "green": 90 + 0.5 * math.sin(phase) + rng.normal(0, 0.05), "has_person": True,
               "box_x1": pose[:, 0].min() - 10, "box_y1": pose[:, 1].min() - 10,
               "box_x2": pose[:, 0].max() + 10, "box_y2": pose[:, 1].max() + 10}
        for j in range(17):
            row[f"kp{j}_x"], row[f"kp{j}_y"], row[f"kp{j}_c"] = pose[j, 0], pose[j, 1], 0.9
        rec.capture_frame(row)
        if i % int(FS) == 0:
            rec.write({"t": t0 + ts, "rr": rate, "sources": {"rr.depth": rate + 0.5}, "sqi": {"rr.depth": 0.8}})
    if fall_at is not None:
        rec.add_event("fall", note="landed", t=t0 + fall_at + 0.6)
    rec.stop()
    return rec.dir


def _by_label(root, label):
    return next(p for p in dataset.find_capture_sessions(root)
                if json.loads((p / "meta.json").read_text())["label"] == label)


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    for k, (sub, rates) in enumerate((("S01", [(0, 10), (60, 20)]), ("S02", [(0, 15)]), ("S03", [(0, 12)]))):
        _record(root, f"{sub}_dev_paced", rates, 100, seed=k)
    _record(root, "S01_dev_falls", [(0, 14)], 40, fall_at=20, seed=9)
    _record(root, "S02_dev_falls", [(0, 14)], 40, fall_at=25, seed=10)
    return root


def test_frames_csv_has_the_capture_columns(sessions):
    d = dataset.find_capture_sessions(sessions)[0]
    with open(d / "frames.csv", newline="") as f:
        assert next(csv.reader(f)) == FRAME_COLUMNS
    meta = json.loads((d / "meta.json").read_text())
    assert meta["capture"] is True and meta["frames"] > 0


def test_subject_code_comes_from_the_label():
    assert dataset.subject_of({"label": "S07_dev_paced12", "id": "x"}) == "S07"
    assert dataset.subject_of({"label": "p3-icu-rest", "id": "x"}) == "P3"
    assert dataset.subject_of({"label": "no code", "id": "sess-1"}) == "sess-1"


def test_rr_windows_match_live_preparation_and_labels(sessions):
    d = _by_label(sessions, "S01_dev_paced")
    w = dataset.rr_windows(d, "rr_d435")
    assert w.X.shape[1:] == (1, 300) and len(w) > 10
    assert set(np.unique(w.y)) <= {10.0, 20.0}
    fr = dataset.load_frames(d)
    end_m = w.t_end[0] - float(np.median(fr["t"] - fr["t_mono"]))
    sel = (fr["t_mono"] >= end_m - 10.2) & (fr["t_mono"] <= end_m + 1e-9)
    x, _ = resample_window(fr["t_mono"][sel], fr["chest_depth"][sel], 300, 10.0)
    assert np.allclose(w.X[0, 0], rr_net_input(x, True), atol=1e-5)
    # nothing labelled during the warm-up or right after the pace change at 60 s
    t0 = json.loads((d / "meta.json").read_text())["started"]
    rel = np.asarray(w.t_end) - t0
    assert rel.min() >= 40 - 1e-6
    assert not np.any((rel > 60) & (rel < 90))


def test_fall_windows_are_labelled_around_the_marked_fall(sessions):
    d = _by_label(sessions, "S01_dev_falls")
    w = dataset.fall_windows(d)
    assert w.X.shape[1:] == (32, 62)
    assert w.y.sum() >= 1 and (w.y == 0).sum() > 10
    pos_ends = np.asarray(w.t_end)[w.y > 0.5]
    t0 = json.loads((d / "meta.json").read_text())["started"]
    # fall marked at 20.6 s; positive if the mark is within [window start - 1.5 s, window end + 2 s]
    assert np.all((pos_ends - t0 > 18.5) & (pos_ends - t0 < 23.3))


def test_split_is_by_subject_and_disjoint():
    s = dataset.split_subjects([f"S{i:02d}" for i in range(10)], seed=1)
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 10


def test_finetune_and_eval_rr_end_to_end(sessions, tmp_path):
    init = tmp_path / "init.pth"
    torch.save(Resp1DCNN().state_dict(), init)
    out = tmp_path / "rr.pth"
    rc = finetune_main(["--module", "rr_d435", "--sessions", str(sessions), "--init", str(init),
                        "--out", str(out), "--epochs", "2", "--batch", "16",
                        "--splits-dir", str(tmp_path / "splits")])
    assert rc == 0 and out.exists()
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["split"]["test"] and not set(report["split"]["train"]) & set(report["split"]["test"])
    assert report["test_after"]["n"] > 0
    m = Resp1DCNN()
    m.load_state_dict(torch.load(out, weights_only=True))      # loads exactly as the monitor does
    assert eval_main(["--module", "rr_d435", "--weights", str(out), "--sessions", str(sessions),
                      "--splits-dir", str(tmp_path / "splits"), "--out", str(tmp_path / "eval.json")]) == 0
    ev = json.loads((tmp_path / "eval.json").read_text())
    assert ev["model"]["n"] > 0 and "rr.depth" in ev["signal_processing_same_seconds"]


def test_finetune_fall_end_to_end(sessions, tmp_path):
    from src.modules.fall import FallGRUClassifier, _load_gru
    init = tmp_path / "fall_init.pt"
    net = FallGRUClassifier()
    torch.save({"model_state_dict": net.state_dict(),
                "model_config": {"input_size": 62, "hidden_size": 128, "num_layers": 2, "dropout": 0.2}}, init)
    out = tmp_path / "fall.pt"
    rc = finetune_main(["--module", "fall", "--sessions", str(sessions), "--init", str(init), "--out", str(out),
                        "--epochs", "2", "--batch", "32", "--min-subjects", "2",
                        "--splits-dir", str(tmp_path / "splits")])
    assert rc == 0
    model, info = _load_gru(out, out, torch.device("cpu"))     # the monitor's own loader
    assert info["slot"] == "primary"


def test_finetune_refuses_too_few_people(sessions, tmp_path):
    init = tmp_path / "init.pth"
    torch.save(Resp1DCNN().state_dict(), init)
    assert finetune_main(["--module", "rr_d435", "--sessions", str(sessions), "--init", str(init),
                          "--min-subjects", "10", "--splits-dir", str(tmp_path / "s")]) == 1


def test_capture_row_uses_the_live_module_inputs():
    from types import SimpleNamespace
    from src.modules.rr import _mean_from_depth_roi
    from src.serving.capture import capture_row
    from src.types import FrameBundle
    rgb = np.full((48, 64, 3), 40, np.uint8)
    rgb[:, :, 1] = 120
    chest = np.full((10, 10), 1600, np.uint16)
    chest[:, :2] = 3800                                      # wall behind the shoulder
    roi = SimpleNamespace(chest_depth=chest, abdomen_depth=np.full((8, 8), 1650, np.uint16))
    person = SimpleNamespace(keypoints=UPRIGHT, keypoints_conf=np.full(17, 0.8), bbox=np.array([1, 2, 3, 4.0]))
    b = FrameBundle(rgb=rgb, depth=np.zeros((48, 64), np.uint16), ir=None, thermal=None, ts_mono=5.0)
    row = capture_row(b, roi, person, 1000.0)
    assert row["green"] == 120.0
    assert row["chest_depth"] == _mean_from_depth_roi(chest, 5, 30)           # learned D435 input
    assert row["chest_depth_body"] == 1600.0                                  # wall dropped (rr.depth)
    assert row["kp5_x"] == UPRIGHT[5, 0] and row["kp16_c"] == 0.8 and row["box_x2"] == 3.0
    empty = capture_row(b, None, None, 1000.0)
    assert empty["chest_depth"] is None and empty["has_person"] is False


def test_import_reference_spot_and_waveform(tmp_path):
    from scripts.import_reference import import_reference
    rec = SessionRecorder(tmp_path, "bed-1", "S05_dev_rest", sources=("rr.depth",))
    t0 = rec.started
    rec.stop()
    ref = tmp_path / "pulseox.csv"
    with open(ref, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "HR", "SpO2", "PPG"])
        for k in range(5):
            w.writerow([t0 - 3.0 + k * 0.5, 70 + k, 98, 0.1 * k])
    s = import_reference(rec.dir, ref, source="pulse_ox", offset_s=3.0)
    assert s["kept_rows"] >= 1 and s["references_added"] >= 1 and Path(s["ppg_file"]).exists()
    from src.evaluation.compare import load_session
    refs = load_session(rec.dir)["refs"]
    assert any("import:pulse_ox" in r["note"] for r in refs)
    assert (rec.dir / "references.manual.csv").exists()


def test_api_capture_and_events_in_demo_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.serving.api import create_app
    from src.serving.engine import EngineManager
    monkeypatch.setenv("VISIONICU_SESSIONS_DIR", str(tmp_path))
    app = create_app(EngineManager.from_args(demo=True), frontend_dir="does/not/exist")
    with TestClient(app) as c:
        assert "fall" in c.get("/api/session-event-kinds").json()["kinds"]
        s = c.post("/api/patients/bed-1/session", json={"label": "S09_demo", "capture": True}).json()
        assert s["capture"] is True
        assert c.post("/api/patients/bed-1/session/event", json={"kind": "talking"}).status_code == 200
        assert c.post("/api/patients/bed-1/session/event",
                      json={"kind": "pain_score", "value": 3, "note": "CPOT"}).json()["events"] == 2
        assert c.post("/api/patients/bed-1/session/event", json={"kind": "nonsense"}).status_code == 422
        import time as _t
        _t.sleep(1.5)
        stopped = c.delete("/api/patients/bed-1/session").json()
        assert stopped["frames"] > 5
        assert c.post("/api/patients/bed-1/session/event", json={"kind": "fall"}).status_code == 409
    d = next(Path(tmp_path).glob("bed-1_*"))
    assert (d / "frames.csv").exists() and len((d / "events.csv").read_text().splitlines()) == 3
