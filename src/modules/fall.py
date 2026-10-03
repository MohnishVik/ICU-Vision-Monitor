"""
fall.py — Fall / Chair-Exit Detection module
Three layers applied sequentially.

Layer 1: Geometric rules on depth (no weights, always on, works in darkness)
          h(t) = centroid height above floor; fall if h < 0.35m, outside chair, immobile 5s+
Layer 2: YOLO11-pose keypoints lifted to 3D -> ~30 feature vector per window
Layer 3: Trained classifier (fall_model_finetuned.pt) — LOCKED, no retraining

Patient state machine:
  IN_CHAIR -> SITTING_EDGE (warn) -> OUT_OF_CHAIR (alert) -> ON_FLOOR (critical)

Suppression: staff_count >= 2 -> log only; dwell >= 1.5s required before escalation

Output: Measurement(value=state_code | None, sqi, ts, source="fall.classifier", ...)
        meta includes: patient_state (str), h_centroid (float), in_chair_frac (float)
"""
# TODO: implement — paste your existing Fall detection code here when ready
# Key functions:
#   apply_geometry_rules(h_centroid, in_chair_frac, prev_state) -> PatientState
#   extract_fall_features(keypoints_3d, h_centroid, h_history) -> np.ndarray (30,)
#   run_classifier(features, model) -> (state_probs, PatientState)
#   apply_suppression(state, staff_count, dwell_s) -> PatientState
