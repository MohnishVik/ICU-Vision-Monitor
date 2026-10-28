"""
test_roi_boxes.py — the chest window must not jump when hips flicker or keypoints wobble.

A patient sitting close to the camera (or behind a table, or under a blanket) often has
hips at the image edge, so their confidence flickers. The chest window used to span
shoulders to hips and switched shape with every flicker; the mean depth jumped, looked
like movement, and the breathing reading kept disappearing.
"""
import numpy as np

from src.scene.person_tracker import PatientTrack
from src.scene.pose import PosePerson
from src.scene.roi_bundle import BoxStabilizer, ROIBuilder
from src.types import FrameBundle

KP = np.array([[320, 100], [312, 92], [328, 92], [300, 96], [340, 96], [260, 170], [380, 170],
               [240, 250], [400, 250], [235, 320], [405, 320], [280, 400], [360, 400],
               [280, 470], [360, 470], [280, 479], [360, 479]], np.float32)


def _track(kp, conf):
    p = PosePerson(keypoints=kp.astype(np.float32), keypoints_conf=conf.astype(np.float32),
                   bbox=np.array([200, 60, 440, 479], np.float32), score=0.9)
    return PatientTrack(person=p, in_chair_frac=1.0, h_centroid=0.5, staff_count=0,
                        score=1.0, is_locked=True, ts=0.0)


def _bundle():
    depth = np.full((480, 640), 1500, np.uint16)
    return FrameBundle(rgb=np.zeros((480, 640, 3), np.uint8), depth=depth, ir=None, thermal=None, ts_mono=0.0)


def test_chest_box_ignores_flickering_hips():
    b = ROIBuilder(patient_id="t")
    conf_hips = np.full(17, 0.9)
    conf_no_hips = conf_hips.copy()
    conf_no_hips[[11, 12]] = 0.1
    shapes = set()
    for i in range(60):
        roi = b.build(_bundle(), _track(KP, conf_hips if i % 3 else conf_no_hips))
        shapes.add(roi.chest_depth.shape)
    assert len(shapes) == 1, f"chest window changed shape: {shapes}"


def test_chest_box_survives_a_flickering_shoulder():
    b = ROIBuilder(patient_id="t")
    conf = np.full(17, 0.9)
    one_shoulder = conf.copy()
    one_shoulder[5] = 0.1
    # every 4th frame loses a shoulder (not the first: with no history there is nothing to hold)
    shapes = {b.build(_bundle(), _track(KP, one_shoulder if i % 4 == 3 else conf)).chest_depth.shape
              for i in range(60)}
    assert len(shapes) == 1, f"chest window changed shape: {shapes}"


def test_chest_box_sits_below_the_shoulders_between_them():
    b = ROIBuilder(patient_id="t")
    b.cfg.stabilize = False
    box = b._chest_box(KP, np.full(17, 0.9), None, 640, 480)
    x1, y1, x2, y2 = box
    assert 260 < x1 < x2 < 380 and 170 < y1 < y2 < 320


def test_keypoint_wobble_does_not_move_the_box():
    rng = np.random.default_rng(0)
    b = ROIBuilder(patient_id="t")
    boxes = set()
    for _ in range(90):
        kp = KP + rng.normal(0, 3.0, KP.shape).astype(np.float32)   # ~3 px detector noise
        roi = b.build(_bundle(), _track(kp, np.full(17, 0.9)))
        boxes.add(roi.chest_depth.shape)
    assert len(boxes) <= 2


def test_stabilizer_follows_a_real_move():
    s = BoxStabilizer()
    s.update((100, 100, 200, 200), 640, 480)
    moved = s.update((300, 100, 400, 200), 640, 480)          # patient shifted 200 px
    assert moved[0] > 250
