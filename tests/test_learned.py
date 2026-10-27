"""test_learned.py — learned RR/HR paths: fidelity to the training notebooks + runtime behaviour.

Needs torch and the git-ignored weight files; skipped where either is missing (e.g. CI).
  models/public/rr_cnn1d_lbrdic.pth            (baseline_rr.ipynb, LBRD-IC)
  models/finetuned/rr_cnn1d_d435_finetuned.pth (D435 fine-tune)
  models/public/hr_efficientphys_ubfc.pth      (baseline_hr.ipynb, UBFC-rPPG)
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

RR_LBRD = Path("models/public/rr_cnn1d_lbrdic.pth")
RR_D435 = Path("models/finetuned/rr_cnn1d_d435_finetuned.pth")
HR_EP = Path("models/public/hr_efficientphys_ubfc.pth")
pytestmark = pytest.mark.skipif(not (RR_LBRD.exists() and RR_D435.exists() and HR_EP.exists()),
                                reason="learned weights not present")

import scipy.signal  # noqa: E402

from src.modules.learned import (  # noqa: E402
    EfficientPhys, LearnedConfig, LearnedHRModule, LearnedRRModule, Resp1DCNN, load_learned_modules,
)
from src.types import FrameBundle, ROIBundle  # noqa: E402


def _bundle(ts, green=0.0, depth=1000.0):
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[:, :, 1] = np.uint8(np.clip(round(green), 0, 255))
    return FrameBundle(rgb=rgb, depth=np.full((48, 64), depth, np.uint16), ir=None, thermal=None,
                       ts_mono=ts)


def _roi(face=None, chest_depth=None):
    return ROIBundle(face_roi=face, chest_roi=None, chest_depth=chest_depth, abdomen_roi=None,
                     abdomen_depth=None, limb_rois=[], h_centroid=0.5, in_chair_frac=1.0,
                     staff_count=0, lighting="bright", patient_id="bed-1", ts=0.0)


def _load(cls, path):
    m = cls()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def _zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else x - np.mean(x)


# ---------------------------------------------------------------- RR: LBRD-IC CNN --
def test_lbrd_input_matches_the_training_notebook():
    """baseline_rr.ipynb: whole-frame mean green, first 250 frames of a 25 fps clip, z-scored.
    Fed at exactly 25 fps, the module must see that same vector and give the same output."""
    mod = LearnedRRModule(LearnedConfig(device="cpu"), use_d435=False)
    rng = np.random.default_rng(0)
    green = 120 + 10 * np.sin(2 * np.pi * 0.25 * np.arange(251) / 25) + rng.normal(0, 1, 251)
    green = np.round(green)          # frames are uint8
    out = {}
    for i, g in enumerate(green):
        out = mod.process(_bundle(i / 25.0, green=g), _roi()) or out
    m = out["rr.cnn_lbrd"]
    x = torch.tensor(_zscore(green[-250:].astype(np.float32)), dtype=torch.float32)[None, None]
    with torch.no_grad():
        ref = float(_load(Resp1DCNN, RR_LBRD)(x).item())
    assert m.meta["raw_output"] == pytest.approx(ref, abs=1e-4)


def test_d435_input_matches_the_finetune_script():
    """train_finetune_rr_realsense.py: 300-frame chest-depth window, linear detrend, z-score."""
    mod = LearnedRRModule(LearnedConfig(device="cpu"), use_lbrd=False)
    t = np.arange(301) / 30.0
    depth = np.round(1000 + 4 * np.sin(2 * np.pi * 0.3 * t) + 0.5 * t)
    out = {}
    for ti, d in zip(t, depth):
        chest = np.full((20, 20), d, np.uint16)
        out = mod.process(_bundle(ti, depth=d), _roi(chest_depth=chest)) or out
    m = out["rr.cnn_d435"]
    w = scipy.signal.detrend(depth[-300:].astype(np.float32), type="linear")
    x = torch.tensor(_zscore(w).astype(np.float32))[None, None]
    with torch.no_grad():
        ref = float(_load(Resp1DCNN, RR_D435)(x).item())
    assert m.meta["raw_output"] == pytest.approx(ref, abs=1e-4)


def test_rr_paths_wait_for_a_full_window_then_update_each_second():
    mod = LearnedRRModule(LearnedConfig(device="cpu"))
    times = []
    for i in range(int(14 * 30)):
        ts = i / 30.0
        out = mod.process(_bundle(ts, green=100 + 5 * np.sin(ts), depth=1000 + 3 * np.sin(ts)), _roi())
        if "rr.cnn_lbrd" in out:
            times.append(ts)
    assert times and times[0] >= 10.0 - 1e-6
    assert len(times) in (4, 5)                       # ~10, 11, 12, 13 s


def test_flat_input_reports_none_with_reason():
    mod = LearnedRRModule(LearnedConfig(device="cpu"))
    out = {}
    for i in range(330):
        out = mod.process(_bundle(i / 30.0, green=90, depth=1000), _roi()) or out
    for src in ("rr.cnn_lbrd", "rr.cnn_d435"):
        assert out[src].value is None and out[src].meta["reason"] == "flat_signal"


# ---------------------------------------------------------------- HR: EfficientPhys --
def _face(i, fs=30.0, hr_hz=1.2):
    """72x72 BGR skin patch whose green channel pulses at hr_hz."""
    f = np.full((90, 80, 3), (90, 120, 170), np.float32)
    f[:, :, 1] += 3.0 * np.sin(2 * np.pi * hr_hz * i / fs)
    return np.clip(f, 0, 255).astype(np.uint8)


def test_efficientphys_chunk_matches_the_training_notebook():
    """baseline_hr.ipynb: 72x72 BGR frames, (f[t+1]-f[t])/(f[t+1]+f[t]), 72-frame chunk -> BVP."""
    mod = LearnedHRModule(LearnedConfig(device="cpu"))
    faces = [_face(i) for i in range(73)]
    for i, f in enumerate(faces):
        mod.process(_bundle(i / 30.0), _roi(face=f))
    import cv2
    fr = np.array([cv2.resize(f, (72, 72)).astype(np.float32) for f in faces])
    diff = (fr[1:] - fr[:-1]) / (fr[1:] + fr[:-1] + 1e-6)
    x = torch.from_numpy(np.transpose(diff, (3, 0, 1, 2))).unsqueeze(0).float()
    with torch.no_grad():
        ref = _load(EfficientPhys, HR_EP)(x).squeeze().numpy()
    np.testing.assert_allclose(mod.last_chunk_bvp, ref, atol=1e-4)


def test_efficientphys_emits_a_rate_after_its_window():
    mod = LearnedHRModule(LearnedConfig(device="cpu"))
    emitted = []
    for i in range(int(22 * 30)):
        out = mod.process(_bundle(i / 30.0), _roi(face=_face(i)))
        if "hr.efficientphys" in out:
            emitted.append(out["hr.efficientphys"])
    assert emitted
    m = emitted[-1]
    assert m.source == "hr.efficientphys" and m.meta["experimental"] is True
    assert m.value is None or 40.0 <= m.value <= 180.0


def test_losing_the_face_resets_the_hr_buffer():
    mod = LearnedHRModule(LearnedConfig(device="cpu"))
    for i in range(100):
        mod.process(_bundle(i / 30.0), _roi(face=_face(i)))
    out = mod.process(_bundle(100 / 30.0), _roi(face=None))
    assert mod.buffered_frames == 0
    assert out == {} or out["hr.efficientphys"].meta["reason"] == "no_face"


# ---------------------------------------------------------------- loading --
def test_loader_reports_each_missing_weight_by_path(tmp_path):
    cfg = tmp_path / "signal.yaml"
    cfg.write_text("learned:\n  rr_lbrd:\n    weights: models/public/nope_rr.pth\n")
    mods, errors = load_learned_modules(str(cfg))
    assert "nope_rr.pth" in errors["rr_lbrd"]
    assert "hr_learned" in mods          # the other paths still load


def test_an_unreadable_weight_file_disables_only_its_own_path(tmp_path):
    bad = tmp_path / "pointer.pth"
    bad.write_text("version https://git-lfs.github.com/spec/v1\n")      # an un-pulled LFS pointer
    cfg = tmp_path / "signal.yaml"
    cfg.write_text(f"learned:\n  rr_lbrd:\n    weights: {bad.as_posix()}\n")
    mods, errors = load_learned_modules(str(cfg))
    assert "pointer.pth" in errors["rr_lbrd"]
    assert mods["rr_learned"].sources == ("rr.cnn_d435",)


def test_fuse_flag_defaults_off():
    assert LearnedConfig().fuse is False


def test_background_inference_matches_inline_and_does_not_block_the_frame_loop():
    """On a CPU one EfficientPhys chunk takes ~0.7 s; in the camera engine it runs on a worker
    thread so the frame loop (depth RR, fall, ...) never waits for it."""
    inline = LearnedHRModule(LearnedConfig(device="cpu"))
    bg = LearnedHRModule(LearnedConfig(device="cpu"), background=True)
    import time as _t
    worst = 0.0
    for i in range(73):
        inline.process(_bundle(i / 30.0), _roi(face=_face(i)))
        t0 = _t.perf_counter()
        bg.process(_bundle(i / 30.0), _roi(face=_face(i)))
        worst = max(worst, _t.perf_counter() - t0)
    bg.drain()
    np.testing.assert_allclose(bg.last_chunk_bvp, inline.last_chunk_bvp, atol=1e-6)
    assert worst < 0.1                       # the chunk itself ran on the worker
