"""test_thermal_registration.py — thermal<->RGB homography, nostril ROI, and RR source switching."""
import json

import numpy as np
import pytest

from src.scene.thermal_registration import (
    ThermalRegistration, fit_homography, load_thermal_registration, mean_in_roi,
    nostril_box_rgb, registration_path, thermal_roi_from_rgb_box, warmest_patch_c,
)

# a realistic rig: thermal FOV sits inside the RGB view, slight rotation + perspective
H_TRUE = np.array([[6.9, 0.35, 42.0], [-0.25, 7.05, 18.0], [0.0004, -0.0003, 1.0]])


def _project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _pairs(th_pts, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rgb = _project(H_TRUE, np.asarray(th_pts, float)) + rng.normal(0, noise_px, (len(th_pts), 2))
    return [{"rgb": list(r), "thermal": list(t)} for r, t in zip(rgb, th_pts)]


SPREAD = [[8, 6], [70, 8], [74, 55], [10, 52], [40, 30], [22, 40], [60, 20]]


def test_fit_recovers_known_homography_exactly():
    H = fit_homography(np.array(SPREAD[:4], float), _project(H_TRUE, np.array(SPREAD[:4], float)))
    assert np.allclose(H, H_TRUE / H_TRUE[2, 2], atol=1e-6)


def test_fit_with_click_noise_reports_small_error_and_maps_back():
    reg = ThermalRegistration.fit(_pairs(SPREAD, noise_px=2.0))
    assert reg.n_pairs == 7 and reg.error_px < 4.0 and reg.quality == "good"
    rt = reg.to_thermal(reg.to_rgb([[33.0, 27.0]]))
    assert np.allclose(rt, [[33.0, 27.0]], atol=1e-6)


def test_four_pairs_fit_exactly_and_are_flagged_unchecked():
    reg = ThermalRegistration.fit(_pairs(SPREAD[:4], noise_px=3.0))
    assert reg.error_px < 1e-6 and reg.quality == "unchecked"


def test_rejects_too_few_collinear_and_bunched_points():
    with pytest.raises(ValueError, match="at least 4"):
        ThermalRegistration.fit(_pairs(SPREAD[:3]))
    with pytest.raises(ValueError, match="bunched|line"):
        ThermalRegistration.fit(_pairs([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]))
    with pytest.raises(ValueError, match="bunched|line"):
        ThermalRegistration.fit(_pairs([[40, 30], [41, 30], [41, 31], [40, 31]]))


def test_rejects_a_mismatched_pair():
    pairs = _pairs(SPREAD)
    pairs[2]["rgb"] = [100.0, 400.0]          # clicked the wrong spot in the colour image
    with pytest.raises(ValueError, match="disagree|fold"):
        ThermalRegistration.fit(pairs)


def test_rejects_points_outside_images():
    pairs = _pairs(SPREAD)
    pairs[0]["thermal"] = [90.0, 10.0]
    with pytest.raises(ValueError, match="outside"):
        ThermalRegistration.fit(pairs)


def test_save_load_roundtrip_and_camera_yaml_fallback(tmp_path):
    reg = ThermalRegistration.fit(_pairs(SPREAD, noise_px=1.0))
    reg.save(registration_path(tmp_path, "bed-1"))
    back = load_thermal_registration(tmp_path, "bed-1")
    assert back is not None and np.allclose(back.H, reg.H) and back.source == "session"
    assert load_thermal_registration(tmp_path, "bed-2") is None
    (tmp_path / "camera.yaml").write_text(
        "waveshare_thermal:\n  homography_matrix: " + json.dumps(H_TRUE.tolist()) + "\n")
    fixed = load_thermal_registration(tmp_path, "bed-2")
    assert fixed is not None and fixed.source == "camera.yaml"


def _face_kpts(nose=(320, 150), eye_d=30.0):
    k = np.zeros((17, 2), np.float32)
    k[0] = nose
    k[1] = (nose[0] - eye_d / 2, nose[1] - 12)
    k[2] = (nose[0] + eye_d / 2, nose[1] - 12)
    return k, np.full(17, 0.9, np.float32)


def test_nostril_box_sits_below_nose_and_scales_with_face():
    k, c = _face_kpts()
    x0, y0, x1, y1 = nostril_box_rgb(k, c)
    assert x0 < 320 < x1 and y0 > 150 and y1 > y0
    k2, c2 = _face_kpts(eye_d=60.0)
    bx = nostril_box_rgb(k2, c2)
    assert (bx[2] - bx[0]) == pytest.approx(2 * (x1 - x0))
    c[0] = 0.1                                        # nose not confident -> no box
    assert nostril_box_rgb(k, c) is None


def test_thermal_roi_maps_to_the_right_place_with_minimum_size():
    reg = ThermalRegistration(H=H_TRUE)
    target = np.array([[40.0, 20.0]])                 # thermal pixel we expect
    cx, cy = _project(H_TRUE, target)[0]
    roi = thermal_roi_from_rgb_box(reg, (cx - 5, cy - 5, cx + 5, cy + 5), min_px=3)
    x0, y0, x1, y1 = roi
    assert x1 - x0 >= 3 and y1 - y0 >= 3
    assert x0 <= 40 < x1 and y0 <= 20 < y1
    assert thermal_roi_from_rgb_box(reg, (-900, -900, -880, -880)) is None   # off the sensor


def _breathing_frames(rr_bpm=15.0, seconds=30.0, fs=8.0, seed=0):
    """Thermal frames where ONLY the nostril pixels breathe; a hotter static forehead
    dominates the warmest-2 % patch, so the old fallback cannot see the breathing."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    frames = []
    for i in range(n):
        t = i / fs
        f = np.full((62, 80), 24.0, np.float32) + rng.normal(0, 0.05, (62, 80)).astype(np.float32)
        f[6:18, 30:50] = 36.5                                  # forehead/eyes: hottest, steady
        f[20:23, 39:42] = 33.0 + 1.2 * np.sin(2 * np.pi * rr_bpm / 60.0 * t)   # nostrils
        frames.append((t, f))
    return frames


def _roi(thermal_roi):
    from src.types import ROIBundle
    return ROIBundle(None, None, None, None, None, [], 0.0, 0.0, 0, "bright", "p", 0.0,
                     thermal_roi=thermal_roi)


def test_rr_reads_breathing_at_the_nostrils_and_resets_on_source_change():
    from src.modules.rr import load_rr_module
    from src.types import FrameBundle

    rr = load_rr_module("config/signal.yaml")
    depth = np.zeros((480, 640), np.uint16)
    rgb = np.zeros((480, 640, 3), np.uint8)
    out = {}
    for t, f in _breathing_frames():
        b = FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=f, ts_mono=100.0 + t,
                        thermal_ts=100.0 + t)
        out.update(rr.process(b, _roi((39, 20, 42, 23))))
    m = out["rr.thermal"]
    assert m.meta["roi"] == "nostril"
    assert m.value is not None and abs(m.value - 15.0) < 1.5, m

    # ROI lost (nose out of view): the buffer must restart, not splice in another signal
    t0 = 100.0 + 30.0
    b = FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=_breathing_frames()[0][1],
                    ts_mono=t0, thermal_ts=t0)
    rr.process(b, _roi(None))
    assert rr._thermal_src == "warmest_patch" and len(rr._thermal.sig) == 1


def test_warmest_patch_fallback_misses_the_breathing_that_nostril_roi_catches():
    from src.modules.rr import load_rr_module
    from src.types import FrameBundle

    rr = load_rr_module("config/signal.yaml")
    depth = np.zeros((480, 640), np.uint16)
    rgb = np.zeros((480, 640, 3), np.uint8)
    out = {}
    for t, f in _breathing_frames():
        b = FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=f, ts_mono=t, thermal_ts=t)
        out.update(rr.process(b, None))
    m = out.get("rr.thermal")
    assert m is None or m.value is None or abs(m.value - 15.0) > 1.5


def _noisy_rois(cx, cy, n, sigma, seed=0):
    """Integer 3x3 ROIs from a noisy centre estimate (what pose jitter produces)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x, y = cx + rng.normal(0, sigma), cy + rng.normal(0, sigma)
        w = 3 if rng.random() > 0.15 else 4                      # face-size noise flips 3 <-> 4
        x0, y0 = int(round(x - w / 2)), int(round(y - 1.5))
        out.append((x0, y0, x0 + w, y0 + 3))
    return out


def test_stabilizer_holds_still_and_converges_on_the_true_spot():
    from src.scene.thermal_registration import RoiStabilizer
    st = RoiStabilizer()
    seq = [st.update(r) for r in _noisy_rois(40.5, 19.5, 300, sigma=0.6)]
    tail = seq[100:]                                              # after warm-up
    changes = sum(a != b for a, b in zip(tail, tail[1:]))
    assert changes <= 3, f"ROI moved {changes} times while the face was still"
    x0, y0, x1, y1 = tail[-1]
    assert x0 <= 40.5 < x1 and y0 <= 19.5 < y1                    # centred on the true spot


def test_stabilizer_snaps_when_the_patient_really_moves():
    from src.scene.thermal_registration import RoiStabilizer
    st = RoiStabilizer()
    for r in _noisy_rois(40.5, 19.5, 60, sigma=0.4):
        st.update(r)
    moved = st.update((50, 30, 53, 33))                           # head moved ~10 px
    assert moved == (50, 30, 53, 33)
    assert st.update(None) is None                                # face lost -> reset


def test_mean_in_roi_and_warmest_patch():
    t = np.full((62, 80), 24.0, np.float32)
    t[14:26, 34:46] = 34.0                     # a face-sized warm region (144 px > the 2 % = 99 px)
    assert mean_in_roi(t, (39, 18, 42, 21)) == pytest.approx(34.0)
    assert warmest_patch_c(t) == pytest.approx(34.0)
    assert mean_in_roi(None, (0, 0, 3, 3)) is None
