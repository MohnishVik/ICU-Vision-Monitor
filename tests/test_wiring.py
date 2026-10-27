"""test_wiring.py — pain + learned paths in the vitals pipeline, and the model-file check."""
import json
import time

import yaml

from src.serving.models_check import check_models
from src.serving.state import VitalsPipeline, SceneInfo
from src.types import Measurement


def _pipeline(signal_cfg=None):
    with open("config/ews_rules.yaml") as f:
        rules = yaml.safe_load(f)
    return VitalsPipeline("bed-1", "Bed 1", rules, signal_cfg or {})


def _m(src, v, sqi=0.9, **meta):
    return Measurement(v, sqi, time.time(), src, 1.0, meta)


def _scene():
    return SceneInfo(has_patient=True, in_chair_frac=1.0, calibrated=True)


def test_learned_rr_is_shown_but_not_fused_by_default():
    p = _pipeline()
    p.step(time.time(), {"rr.depth": _m("rr.depth", 15.0), "rr.cnn_lbrd": _m("rr.cnn_lbrd", 15.9,
                                                                              experimental=True)}, _scene())
    rr = p.snapshot()["vitals"]["rr"]
    assert rr["raw"] == 15.0
    assert rr["sources"]["rr.cnn_lbrd"]["value"] == 15.9
    assert rr["sources"]["rr.cnn_lbrd"]["role"] == "display"
    assert rr["sources"]["rr.depth"]["role"] == "fused"
    assert rr["learned_fused"] is False


def test_learned_rr_joins_fusion_when_enabled():
    p = _pipeline({"learned": {"fuse": True}})
    p.step(time.time(), {"rr.depth": _m("rr.depth", 15.0), "rr.cnn_lbrd": _m("rr.cnn_lbrd", 16.0)}, _scene())
    rr = p.snapshot()["vitals"]["rr"]
    assert 15.0 < rr["raw"] < 16.0
    assert rr["sources"]["rr.cnn_lbrd"]["role"] == "fused" and rr["learned_fused"] is True


def test_learned_hr_listed_under_hr_sources():
    p = _pipeline()
    p.step(time.time(), {"hr.pos": _m("hr.pos", 72.0), "hr.efficientphys": _m("hr.efficientphys", 100.0)},
           _scene())
    hr = p.snapshot()["vitals"]["hr"]
    assert hr["raw"] == 72.0 and hr["sources"]["hr.efficientphys"]["value"] == 100.0


def test_pain_snapshot_reports_probability_and_never_alarms():
    p = _pipeline()
    p.modules["pain"] = {"ok": True, "state": "ok", "detail": "", "errors": 0}
    p.step(time.time(), {"pain.mobilenetv3": _m("pain.mobilenetv3", 0.62, sqi=0.8, threshold=0.2962,
                                                pain_detected=True, experimental=True,
                                                probability_raw=0.66, reason=None)}, _scene())
    s = p.snapshot()
    pain = s["pain"]
    assert pain["available"] is True and pain["experimental"] is True
    assert pain["probability"] == 0.62 and pain["pain_detected"] is True
    assert pain["threshold"] == 0.2962 and pain["stale"] is False
    assert s["alarms"] == []
    json.dumps(s, allow_nan=False)


def test_pain_unavailable_explains_why():
    p = _pipeline()
    p.modules["pain"] = {"ok": False, "state": "unavailable",
                         "detail": "Weights missing - Pain weights not found: x.pth", "errors": 0}
    p.step(time.time(), {"rr.depth": _m("rr.depth", 15.0)}, _scene())
    pain = p.snapshot()["pain"]
    assert pain["available"] is False and "x.pth" in pain["reason"]


def test_history_carries_every_source_and_pain():
    p = _pipeline()
    p.step(time.time(), {"rr.depth": _m("rr.depth", 15.0), "rr.cnn_d435": _m("rr.cnn_d435", 6.4),
                         "pain.mobilenetv3": _m("pain.mobilenetv3", 0.3)}, _scene())
    h = p.history(minutes=5)
    assert h["sources"]["rr.cnn_d435"] == [6.4]
    assert h["sources"]["rr.depth"] == [15.0]
    assert h["pain"] == [0.3]


def test_frozen_snapshot_ages_to_stale_when_steps_stop():
    """Frames stall or calibration is cleared: step() stops, the last values must not stay live."""
    p = _pipeline()
    t0 = time.time()
    for k in range(3):
        p.step(t0 - 20 + k, {"rr.depth": _m("rr.depth", 15.0), "hr.pos": _m("hr.pos", 72.0)}, _scene())
    s = p.snapshot()
    for key in ("rr", "hr"):
        assert s["vitals"][key]["value"] is None and s["vitals"][key]["stale"] is True
        assert s["vitals"][key]["sources"] == {} and s["vitals"][key]["age_s"] > 5
    assert s["ews"]["total"] is None and s["ews"]["level"] == "unknown"
    json.dumps(s, allow_nan=False)
    p.step(time.time(), {"rr.depth": _m("rr.depth", 15.0)}, _scene())      # fresh again
    assert p.snapshot()["vitals"]["rr"]["stale"] is False


def test_model_check_lists_every_expected_file(tmp_path):
    (tmp_path / "models" / "public").mkdir(parents=True)
    (tmp_path / "models" / "public" / "yolov8n-pose.pt").write_bytes(b"x" * 10)
    rows = {r["file"]: r for r in check_models(root=tmp_path)}
    assert rows["models/public/yolov8n-pose.pt"]["present"] is True
    assert rows["models/public/pain_mobilenetv3_synpain.pth"]["present"] is False
    for f in ("models/public/fall_gru_pseudo.pt", "models/public/fall_gru_pseudo_e1.pt",
              "models/public/rr_cnn1d_lbrdic.pth", "models/finetuned/rr_cnn1d_d435_finetuned.pth",
              "models/public/hr_efficientphys_ubfc.pth"):
        assert f in rows and rows[f]["used_by"]
