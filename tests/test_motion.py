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


def test_rr_reports_motion_and_recovers_with_a_clean_window():
    rr, cfg = _rr()
    depth = np.zeros((48, 64), np.uint16)
    rgb = np.zeros((48, 64, 3), np.uint8)
    reasons, values = [], []
    for i in range(int(70 * FS)):
        ts = i / FS
        level = 1200 if ts < 30 else 1050                    # patient shifts at 30 s
        depth[:] = int(level + 6 * np.sin(2 * np.pi * 0.25 * ts))   # 15 br/min
        out = rr.process(FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=None, ts_mono=ts))
        m = out.get("rr.depth")
        if m is not None:
            reasons.append((ts, m.meta.get("reason")))
            values.append((ts, m.value))
    assert any(r == "motion" and 30 <= ts <= 52 for ts, r in reasons)
    late = [v for ts, v in values if ts > 55 and v is not None]
    assert late and abs(np.median(late) - 15.0) < 1.0      # clean window after the shift
