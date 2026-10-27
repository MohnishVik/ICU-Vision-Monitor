"""
augmentations.py — label-preserving changes applied to training windows, never to test windows.

Breathing windows (already z-scored, shape (1, n)):
  sign flip     a chest moving towards the camera on inhale and one moving away carry the same
                rate; the camera can see either depending on posture
  noise         sensor noise at a random level
  drift         a slow ramp, like a patient slowly sliding down the chair
  re-z-score    after the above, so the input stays in the range the network was trained on

Fall windows (shape (32, 62): 34 normalised keypoint coords, 17 confidences, box, kinematics):
  jitter        small random keypoint wobble, as from the pose detector
  shift/scale   the whole body a little left/right/up/down or nearer/farther
"""
from __future__ import annotations

import numpy as np


def augment_rr(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = x.astype(np.float64).copy()
    if rng.random() < 0.5:
        z = -z
    n = z.shape[-1]
    z += rng.normal(0.0, rng.uniform(0.0, 0.3), size=z.shape)
    if rng.random() < 0.5:
        z += rng.uniform(-0.5, 0.5) * np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / max(z.std(), 1e-6)
    return z.astype(np.float32)


def augment_fall(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = x.astype(np.float64).copy()
    k = f[:, 0:34].reshape(len(f), 17, 2)
    present = f[:, 61:62] > 0.5                              # only move frames with a person
    scale = rng.uniform(0.92, 1.08)
    shift = rng.uniform(-0.05, 0.05, size=2)
    centre = k.reshape(-1, 2).mean(axis=0)
    k = np.where(present[:, :, None], (k - centre) * scale + centre + shift
                 + rng.normal(0.0, 0.004, size=k.shape), k)
    f[:, 0:34] = k.reshape(len(f), 34)
    box = f[:, 51:55].reshape(len(f), 2, 2)
    box = np.where(present[:, :, None], (box - centre) * scale + centre + shift, box)
    f[:, 51:55] = box.reshape(len(f), 4)
    f[:, 55] = f[:, 53] - f[:, 51]
    f[:, 56] = f[:, 54] - f[:, 52]
    return f.astype(np.float32)
