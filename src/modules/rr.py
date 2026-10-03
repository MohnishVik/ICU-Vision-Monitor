"""
rr.py — Respiratory Rate module
Three parallel signal paths + TCN deep model, fused by inverse-variance weighting.

Paths:
  A) Optical Flow (RGB/IR)       — no training needed
  B) Depth Chest Displacement    — no training needed; produces TAA flag as bonus
  C) Thermal Airflow             — inference only (NOT in training data); skipped if thermal=None
  D) TCN deep model              — rr_tcn_finetuned.pth (2-channel: flow + depth only)

TCN input shape: (batch, 2, 750)  — channels: [optical_flow_signal, depth_disp_signal]
Thermal is EXCLUDED from TCN input. It only participates in runtime fusion.

Output: Measurement(value=rr_bpm | None, sqi, ts, source="rr.fused", ...)
        meta includes: taa_flag (bool), taa_phase_diff_rad (float), paths_active (list)
"""
# TODO: implement — paste your existing RR code here when ready
# Key functions to implement:
#   compute_optical_flow_rr(chest_roi_gray_sequence) -> (rr, sigma)
#   compute_depth_rr(chest_depth_sequence, abdomen_depth_sequence) -> (rr, sigma, taa_flag)
#   compute_thermal_rr(thermal_roi_sequence) -> (rr, sigma) | (None, None)
#   run_tcn(flow_signal, depth_signal, model) -> (rr_mean, log_var)
#   fuse_rr_paths(path_results) -> Measurement
