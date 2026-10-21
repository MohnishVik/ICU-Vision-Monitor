"""
pain.py — Pain Assessment module
Estimates a CPOT-proxy score from facial AUs and body movement.

Current weights available:
  pain_xgboost_synpain.joblib  — Method 1: 75.4% acc, AUC 0.841 (SynPAIN)
  pain_deepmlp_synpain.pth     — Method 2: 77.1% acc, AUC 0.850 (SynPAIN)
  pain_temporal_tcn.pth        — Method 3: PENDING (absent = auto-skipped)

Feature extraction:
  Primary:  OpenFace 2.x (17 AUs) — set use_openface=true in models.yaml when installed
  Fallback: MediaPipe Face Mesh (10 AU-equivalent landmark distances, inter-ocular norm'd)

Feature engineering (58 features):
  21 pairwise interaction terms (fi x fj)
   4 composite scores (upper_face, lower_face, orbital, total_pain)
   3 ratio features
  20 polynomial terms (fi^2, fi^3)

Inference priority: TCN > DeepMLP > XGBoost (uses best available weight)

Gating: RASS <= -4 OR neuromuscular blockade -> output "not_assessable"

Output: Measurement(value=cpot_proxy | None, sqi, ts, source="pain.{model}", ...)
        meta includes: cpot_facial (float), cpot_body (float), assessable (bool)
"""
# TODO: implement — paste your existing Pain/AU code here when ready
# Key functions:
#   extract_aus_mediapipe(face_roi) -> np.ndarray (10,)
#   extract_aus_openface(face_roi) -> np.ndarray (17,)
#   engineer_features(raw_aus) -> np.ndarray (58,)
#   run_xgboost(features, model) -> (pain_prob, confidence)
#   run_deepmlp(features, model) -> (pain_prob, confidence)
#   run_tcn(au_sequence, model) -> (cpot_score, uncertainty)
#   apply_gating(rass_score, nmb_active) -> bool
