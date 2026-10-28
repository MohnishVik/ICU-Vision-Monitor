"""test_pain.py — pain module: fidelity to the SynPAIN training pipeline + runtime behaviour.

Needs torchvision and the weight file models/public/pain_mobilenetv3_synpain.pth
(git-ignored); the whole module is skipped where either is missing (e.g. CI).
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

WEIGHTS = Path("models/public/pain_mobilenetv3_synpain.pth")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"{WEIGHTS} not present")

from src.modules.pain import PainConfig, PainModule, load_pain_module  # noqa: E402
from src.types import FrameBundle, ROIBundle  # noqa: E402


def _bundle(ts):
    return FrameBundle(rgb=np.zeros((480, 640, 3), np.uint8), depth=np.zeros((480, 640), np.uint16),
                       ir=None, thermal=None, ts_mono=ts)


def _roi(face):
    return ROIBundle(face_roi=face, chest_roi=None, chest_depth=None, abdomen_roi=None,
                     abdomen_depth=None, limb_rois=[], h_centroid=0.5, in_chair_frac=1.0,
                     staff_count=0, lighting="bright", patient_id="bed-1", ts=0.0)


def _face(seed, h=96, w=80):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)   # BGR


def _reference_probability(face_bgr):
    """The notebook's eval path, verbatim: PIL RGB -> eval_tf -> model -> sigmoid."""
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()
    x = eval_tf(Image.fromarray(face_bgr[:, :, ::-1].copy())).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(m(x)).item())


def test_matches_the_training_notebook_eval_pipeline():
    face = _face(1)
    out = PainModule(PainConfig(device="cpu")).process(_bundle(0.0), _roi(face))
    m = out["pain.mobilenetv3"]
    assert m.meta["probability_raw"] == pytest.approx(_reference_probability(face), abs=1e-5)


def test_no_face_reports_none_with_reason():
    m = PainModule(PainConfig(device="cpu")).process(_bundle(0.0), _roi(None))["pain.mobilenetv3"]
    assert m.value is None and m.sqi == 0.0 and m.meta["reason"] == "no_face"


def test_face_too_small_is_not_scored():
    m = PainModule(PainConfig(device="cpu")).process(_bundle(0.0), _roi(_face(2, 20, 20)))["pain.mobilenetv3"]
    assert m.value is None and m.meta["reason"] == "face_too_small"


def test_scores_once_per_interval_and_averages_the_window():
    mod = PainModule(PainConfig(device="cpu", update_every_s=1.0, window=3))
    emitted = []
    for i in range(31):                                   # 3 s at 10 Hz
        out = mod.process(_bundle(i * 0.1), _roi(_face(i)))
        if out:
            emitted.append(out["pain.mobilenetv3"])
    assert len(emitted) == 4                              # t = 0, 1, 2, 3 s
    last = emitted[-1]
    raws = [e.meta["probability_raw"] for e in emitted[-3:]]
    assert last.value == pytest.approx(float(np.mean(raws)), abs=1e-9)


def test_nothing_is_flagged_while_the_calm_baseline_is_learned():
    m = PainModule(PainConfig(device="cpu")).process(_bundle(0.0), _roi(_face(3)))["pain.mobilenetv3"]
    assert m.meta["baseline_state"] == "learning" and m.meta["threshold"] is None
    assert m.meta["pain_detected"] is False
    assert m.meta["model_threshold"] == pytest.approx(0.2962)
    assert m.meta["experimental"] is True and 0.0 <= m.value <= 1.0


def _learned(scores, **cfg):
    mod = PainModule(PainConfig(device="cpu", **cfg))
    for s in scores:
        mod._learn(s)
    return mod


def test_baseline_cancels_the_models_high_bias_on_real_faces():
    # a calm real face scores ~0.80 (the model's bias); it must not read as pain
    rng = np.random.default_rng(0)
    mod = _learned(np.clip(0.80 + rng.normal(0, 0.02, 60), 0, 1))
    thr = mod.personal_threshold()
    assert mod.baseline == pytest.approx(0.80, abs=0.02)
    assert thr == pytest.approx(mod.baseline + 0.10, abs=0.02)   # min rise dominates a steady face
    assert 0.82 < thr and not (0.83 >= thr)                       # calm wobble is not flagged
    assert 0.95 >= thr                                            # a clear rise is


def test_noisy_baseline_widens_the_margin():
    rng = np.random.default_rng(1)
    mod = _learned(np.clip(0.5 + rng.normal(0, 0.08, 60), 0, 1))
    assert mod.personal_threshold() - mod.baseline > 0.15


def test_reset_relearns_the_baseline():
    mod = _learned([0.7] * 60)
    assert mod.baseline is not None
    mod.reset_baseline()
    assert mod.baseline is None and mod.personal_threshold() is None


def test_missing_weights_raise_a_clear_error(tmp_path):
    cfg = tmp_path / "signal.yaml"
    cfg.write_text("pain:\n  weights: models/public/does_not_exist.pth\n")
    with pytest.raises(FileNotFoundError, match="does_not_exist.pth"):
        load_pain_module(str(cfg))
