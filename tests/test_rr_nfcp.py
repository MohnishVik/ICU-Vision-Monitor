import numpy as np
import scipy.signal
from src.modules.rr import RespiratoryRateEstimator

def test_rr_synthetic_sine():
    fs = 20.0
    duration_sec = 15.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # 0.25 Hz = 15 breaths per minute
    target_bpm = 15.0
    target_hz = target_bpm / 60.0
    synthetic_resp = np.sin(2 * np.pi * target_hz * t)

    estimator = RespiratoryRateEstimator(fs=fs)
    pred_bpm, sqi = estimator._spectral_peak(synthetic_resp)

    assert pred_bpm is not None, "Peak extraction failed on clean sine"
    assert abs(pred_bpm - target_bpm) <= 0.5, f"Expected {target_bpm} bpm, got {pred_bpm}"
    assert sqi > 0.5, f"SQI should be high on clean sine, got {sqi}"
