"""test_fusion.py — inverse-variance fusion, Kalman, trends and the alarm state machine."""
import math
import re

from src.fusion import (
    AlarmManager, Condition, EWSScorer, FusionConfig, StalenessWatchdog, TrendTracker,
    VitalKalman, explain, fuse, load_rules,
)
from src.types import Measurement


def m(value, sqi, source):
    return Measurement(value, sqi, 100.0, source, 1.0, {})


# ---------------------------------------------------------------- fusion ----
def test_fuse_weights_toward_higher_quality_path():
    out = fuse([m(14.0, 0.95, "rr.depth"), m(18.0, 0.40, "rr.thermal")], "rr.fused")
    assert out.value is not None and 14.0 < out.value < 16.0
    assert out.meta["weights"]["rr.depth"] > out.meta["weights"]["rr.thermal"]


def test_fuse_sqi_gate_and_none_propagation():
    out = fuse([m(None, 0.9, "rr.depth"), m(15.0, 0.1, "rr.thermal")], "rr.fused")
    assert out.value is None and out.meta["reason"] == "no_valid_input"


def test_fuse_agreement_gate_refuses_to_average_disagreeing_paths():
    out = fuse([m(12.0, 0.9, "rr.depth"), m(20.0, 0.9, "rr.thermal")], "rr.fused",
               FusionConfig(agreement_gate_bpm=4.0))
    assert out.value is None and out.meta["reason"] == "paths_disagree"


def test_fuse_drops_disagreeing_secondary_instead_of_blanking():
    cfg = FusionConfig(agreement_gate_bpm=10.0, secondary=frozenset({"hr.efficientphys"}))
    out = fuse([m(72.0, 0.9, "hr.pos"), m(100.0, 0.9, "hr.efficientphys")], "hr.fused", cfg)
    assert out.value == 72.0
    assert out.meta["rejected"]["hr.efficientphys"] == "disagrees_with_primary"


def test_fuse_blends_agreeing_secondary():
    cfg = FusionConfig(agreement_gate_bpm=4.0, secondary=frozenset({"rr.cnn_lbrd"}))
    out = fuse([m(14.0, 0.9, "rr.depth"), m(16.0, 0.9, "rr.cnn_lbrd")], "rr.fused", cfg)
    assert out.value is not None and 14.0 < out.value < 16.0
    assert set(out.meta["inputs"]) == {"rr.depth", "rr.cnn_lbrd"}


def test_fuse_secondary_alone_when_no_primary():
    cfg = FusionConfig(agreement_gate_bpm=4.0, secondary=frozenset({"rr.cnn_lbrd"}))
    out = fuse([m(None, 0.9, "rr.depth"), m(15.0, 0.8, "rr.cnn_lbrd")], "rr.fused", cfg)
    assert out.value == 15.0


def test_fuse_passes_through_the_actionable_reason():
    moving = Measurement(None, 0.0, 100.0, "rr.depth", 0.0, {"reason": "motion"})
    weak = Measurement(None, 0.0, 100.0, "rr.thermal", 0.0, {})
    assert fuse([moving, weak], "rr.fused").meta["reason"] == "motion"
    dark = Measurement(None, 0.0, 100.0, "hr.pos", 0.0, {"reason": "low_light"})
    assert fuse([dark], "hr.fused").meta["reason"] == "low_light"
    assert fuse([weak], "rr.fused").meta["reason"] == "no_valid_input"


def test_learned_fuse_source_selection():
    from src.serving.state import learned_fuse_sources
    known = ("rr.cnn_lbrd", "rr.cnn_d435", "hr.efficientphys")
    assert learned_fuse_sources({"learned": {"fuse": True}}, known) == known
    assert learned_fuse_sources({"learned": {"fuse": False}}, known) == ()
    assert learned_fuse_sources({"learned": {"fuse": ["rr.cnn_lbrd", "hr.efficientphys"]}}, known) \
        == ("rr.cnn_lbrd", "hr.efficientphys")


def test_staleness_watchdog():
    w = StalenessWatchdog(timeout_s=5.0)
    assert w.is_stale("rr.fused", 0.0)                       # never seen -> stale
    w.observe(m(15.0, 0.9, "rr.fused"), now=10.0)
    assert not w.is_stale("rr.fused", 14.0)
    assert w.is_stale("rr.fused", 15.1)


# ---------------------------------------------------------------- kalman ----
def test_kalman_slew_limit_and_no_invention():
    k = VitalKalman()
    assert k.update(0.0, 70.0, 0.9) == 70.0
    jumped = k.update(1.0, 150.0, 0.9)                        # impossible 80 bpm jump in 1 s
    assert jumped is not None and jumped < 74.0
    assert k.update(2.0, None, 0.9) is None                   # never invents a value


# ---------------------------------------------------------------- trend -----
def test_trend_slope_and_window():
    t = TrendTracker()
    for i in range(41):                                       # RR rises 16 -> 26 over 40 min
        t.add(i * 60.0, 16.0 + i * 0.25)
    st = t.window(40 * 60.0, 3600)
    assert math.isclose(st.slope_per_min, 0.25, rel_tol=1e-6)
    assert st.first == 16.0 and st.last == 26.0
    t.add(41 * 60.0, None)                                    # None never enters the buffer
    assert len(t) == 41


def test_explanation_text():
    scorer = EWSScorer(load_rules("config/ews_rules.yaml"))
    rr, hr = TrendTracker(), TrendTracker()
    for i in range(41):
        rr.add(i * 60.0, 16.0 + i * 0.25)
        hr.add(i * 60.0, 88.0 + i * 0.6)
    now = 40 * 60.0
    ews = scorer.score({"rr": 26, "hr": 112})
    text = explain({"rr": rr.window(now, 3600), "hr": hr.window(now, 3600)}, ews)
    # endpoints average a short edge (<= 30 s); with 1 sample/min that is the exact endpoint
    assert "RR 16→26 over 40 min" in text, text
    assert "HR 88→112 over 40 min" in text, text
    assert "urgent clinical assessment" in text
    # the sentence must quote the same live value the readout shows
    live = explain({"rr": rr.window(now, 3600), "hr": hr.window(now, 3600)}, ews,
                   current={"rr": 27.4, "hr": 113.0})
    assert "RR 16→27" in live and "HR 88→113" in live, live


# ---------------------------------------------------------------- alarms ----
RED = Condition("red", "EWS red", "x")


def test_alarm_dwell_escalation_and_clear():
    a = AlarmManager(load_rules("config/ews_rules.yaml"))
    assert a.update(0.0, {"ews": RED}) == []                  # inside dwell -> pending
    act = a.update(2.0, {"ews": RED})
    assert len(act) == 1 and act[0].state == "active" and act[0].action == "log_only"
    assert a.update(13.0, {"ews": RED})[0].action == "dashboard_highlight"
    assert a.update(33.0, {"ews": RED})[0].action == "chime"
    assert a.actionable_this_shift(33.0) == 1
    assert a.update(63.0, {"ews": RED})[0].action == "phone_push"
    a.update(64.0, {"ews": None})                             # hysteresis hold
    assert len(a.open_alarms()) == 1
    a.update(69.0, {"ews": None})
    assert a.open_alarms() == [] and a.history[0].state == "cleared"


def test_alarm_blip_shorter_than_dwell_never_alarms():
    a = AlarmManager(load_rules("config/ews_rules.yaml"))
    a.update(0.0, {"ews": RED})
    a.update(1.0, {"ews": None})
    assert a.open_alarms() == [] and a.history == []


def test_staff_presence_caps_escalation():
    a = AlarmManager(load_rules("config/ews_rules.yaml"))
    a.update(0.0, {"ews": RED}, staff_count=2)
    act = a.update(40.0, {"ews": RED}, staff_count=2)
    assert act[0].action == "log_only" and act[0].suppressed


def test_fall_latches_until_acknowledged():
    a = AlarmManager(load_rules("config/ews_rules.yaml"))
    fall = Condition("red", "Fall", "Confirmed fall")
    a.update(0.0, {"fall": fall})
    a.update(30.0, {"fall": None})                            # event gone, alarm must persist
    assert len(a.open_alarms()) == 1
    assert a.acknowledge(now=31.0) == 1
    assert a.open_alarms() == []
