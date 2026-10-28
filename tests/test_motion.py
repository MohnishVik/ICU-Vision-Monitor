"""test_motion.py — movement detection restarts the RR/HR windows instead of poisoning them."""
import numpy as np

from src.modules.motion import MotionGate
from src.modules.rr import RRConfig, RRModule
from src.types import FrameBundle
import yaml

FS = 30.0


def test_gate_ignores_breathing_and_noise():
    g = MotionGate(FS, abs_thresh=20.0)
    t = np.arange(0, 60, 1 / FS)
    z = 1200 + 5 * np.sin(2 * np.pi * 0.25 * t) + np.random.default_rng(0).normal(0, 0.5, t.size)
    assert not any(g.update(v, ti) for v, ti in zip(z, t))


def test_gate_flags_a_posture_change_then_settles():
    g = MotionGate(FS, abs_thresh=20.0, settle_s=1.0)
    t = np.arange(0, 10, 1 / FS)
    z = np.where(t < 5, 1200.0, 1100.0)                      # leaned 10 cm towards the camera
    flags = np.array([g.update(v, ti) for v, ti in zip(z, t)])
    assert flags[(t > 5.0) & (t < 5.6)].any()
    assert not flags[t > 7].any()
    assert g.recent(9.0, within_s=5)


def test_relative_gate_for_skin_brightness():
    g = MotionGate(FS, rel_thresh=0.05)
    t = np.arange(0, 6, 1 / FS)
    lum = np.where(t < 3, 80.0, 95.0) * (1 + 0.003 * np.sin(2 * np.pi * 1.2 * t))   # pulse is 0.3 %
    flags = [g.update(v, ti) for v, ti in zip(lum, t)]
    assert any(flags) and not any(flags[:80])


def _rr():
    cfg = RRConfig.from_dict(yaml.safe_load(open("config/signal.yaml", encoding="utf-8")))
    return RRModule(cfg, depth_fs=FS), cfg


def _run_rr(level_at, seconds=90, jitter_at=None):
    """Feed the RR module a 15 br/min chest; level_at(ts) gives posture (mm), jitter_at(ts)
    True means the patient is moving about (large random depth changes)."""
    rr, _ = _rr()
    rng = np.random.default_rng(1)
    depth = np.zeros((48, 64), np.uint16)
    rgb = np.zeros((48, 64, 3), np.uint8)
    out_rows = []
    for i in range(int(seconds * FS)):
        ts = i / FS
        z = level_at(ts) + 6 * np.sin(2 * np.pi * 0.25 * ts)
        if jitter_at is not None and jitter_at(ts):
            z += rng.uniform(-120, 120)
        depth[:] = int(z)
        m = rr.process(FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=None, ts_mono=ts)).get("rr.depth")
        if m is not None:
            out_rows.append((ts, m.value, m.meta.get("reason")))
    return out_rows


def test_brief_shift_keeps_the_breathing_reading():
    rows = _run_rr(lambda ts: 1200 if ts < 30 else 1050)          # leans 15 cm closer at 30 s
    after = [(ts, v) for ts, v, _ in rows if 30 <= ts <= 60]
    assert all(v is not None for _, v in after), "reading dropped after a brief shift"
    assert abs(np.median([v for _, v in after]) - 15.0) < 1.0
    assert not any(r == "motion" for _, _, r in rows)


def test_frequent_shifts_no_longer_blank_breathing():
    # shifts every 15 s: the old restart-on-movement never filled a 20 s window again
    rows = _run_rr(lambda ts: 1200 + 60 * (int(ts // 15) % 2))
    late = [v for ts, v, _ in rows if ts > 40]
    shown = [v for v in late if v is not None]
    assert len(shown) >= 0.8 * len(late)
    assert abs(np.median(shown) - 15.0) < 1.0


def test_sustained_movement_reports_motion_then_recovers():
    rows = _run_rr(lambda ts: 1200, seconds=80, jitter_at=lambda ts: 30 <= ts < 38)  # 8 s moving
    assert any(r == "motion" and 30 <= ts <= 60 for ts, _, r in rows)
    late = [v for ts, v, _ in rows if ts > 65 and v is not None]
    assert late and abs(np.median(late) - 15.0) < 1.0
