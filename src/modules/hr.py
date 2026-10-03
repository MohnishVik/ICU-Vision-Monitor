"""
hr.py — Heart Rate module (rPPG)
Two parallel models fused by inverse-variance weighting.

Models:
  1) POS  — Plane-Orthogonal-to-Skin, ~50 lines NumPy, always on, no weights
  2) EfficientPhys — hr_efficientphys_finetuned.pth
     Input: normalised difference frames (frame_t - frame_{t-1}), shape (T=150, H=72, W=72, 3)

ROI priority: face -> neck -> forehead -> upper chest (scored by visibility x stability x SNR)

CRITICAL: auto-exposure must be locked (enforced in camera.yaml).
          Auto-exposure creates brightness oscillations identical to rPPG pulse signal.

Output: Measurement(value=hr_bpm | None, sqi, ts, source="hr.fused", ...)
"""
# TODO: implement — paste your existing HR/rPPG code here when ready
# Key functions:
#   run_pos(rgb_roi_sequence) -> (hr_bpm, sigma)
#   run_efficientphys(diff_frames, model) -> (hr_bpm, sigma)
#   select_roi(face_roi, neck_roi, forehead_roi) -> best_roi
#   fuse_hr(pos_result, effphys_result) -> Measurement
