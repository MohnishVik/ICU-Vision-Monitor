"""test_no_patient.py — an empty chair never produces a breathing or heart rate.

Seen live: with nobody in view, fixed fallback boxes read wall colour noise as 128 bpm and
depth noise as 7.7 br/min. When the pose system runs and finds nobody, the ROI crops are
None, and every vital-sign module must report a reason instead of a number."""
import numpy as np
import yaml

from src.fusion.inv_variance import fuse
from src.modules.hr import HRConfig, HRModule
from src.modules.rr import RRConfig, RRModule
from src.modules.taa import TAAModule, TAAConfig
from src.types import FrameBundle, ROIBundle

FS = 30.0
SIG = yaml.safe_load(open("config/signal.yaml", encoding="utf-8"))


def _empty_roi():
    return ROIBundle(face_roi=None, chest_roi=None, chest_depth=None, abdomen_roi=None, abdomen_depth=None,
                     limb_rois=[], h_centroid=0.0, in_chair_frac=0.0, staff_count=0, lighting="dim",
                     patient_id="bed-1", ts=0.0)


def _frames(seconds, start=0.0):
    rng = np.random.default_rng(0)
    for i in range(int(seconds * FS)):
        ts = start + i / FS
        # a periodic "something" in the room: 2.1 Hz colour flicker and 0.13 Hz depth sway
        rgb = np.full((48, 64, 3), 120, np.uint8) + np.uint8(3 + 3 * np.sin(2 * np.pi * 2.1 * ts))
        depth = np.full((48, 64), 1500 + 5 * np.sin(2 * np.pi * 0.13 * ts) + rng.normal(0, 0.3), np.uint16)
        yield FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=None, ts_mono=ts)


def test_rr_reports_no_patient_instead_of_reading_an_empty_chair():
    rr = RRModule(RRConfig.from_dict(SIG), depth_fs=FS)
    outs = [rr.process(b, _empty_roi()).get("rr.depth") for b in _frames(40)]
    outs = [m for m in outs if m is not None]
    assert outs and all(m.value is None and m.meta["reason"] == "no_patient" for m in outs)
    assert fuse(outs[-1:], "rr.fused").meta["reason"] == "no_patient"


def test_rr_without_a_pose_system_still_uses_the_fixed_box():
    rr = RRModule(RRConfig.from_dict(SIG), depth_fs=FS)
    vals = [m.value for b in _frames(40) if (m := rr.process(b, None).get("rr.depth")) is not None]
    assert any(v is not None for v in vals)


def test_hr_reports_no_face_and_drops_its_window():
    hr = HRModule(HRConfig.from_dict(SIG), fs=FS)
    face = np.full((60, 60, 3), (90, 120, 170), np.uint8)
    roi_face = _empty_roi(); roi_face.face_roi = face
    for b in _frames(15):
        hr.process(b, roi_face)
    assert len(hr._buf.ts) > 0
    outs = [(b.ts_mono, m) for b in _frames(10, start=15.0)
            if (m := hr.process(b, _empty_roi()).get("hr.pos")) is not None]
    late = [m for ts, m in outs if ts > 16.5]           # after the 1 s blink hold
    assert late and all(m.value is None and m.meta["reason"] == "no_face" for m in late)
    assert len(hr._buf.ts) == 0                     # the face-window does not span the absence


def test_taa_stays_silent_on_an_empty_chair():
    taa = TAAModule(TAAConfig.from_dict(SIG), fs=FS)
    assert not any(taa.process(b, _empty_roi()) for b in _frames(40))
