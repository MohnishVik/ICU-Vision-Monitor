import numpy as np
from src.modules.hr import HeartRateEstimator

def test_hr_synthetic_pos():
    fs = 20.0
    duration_sec = 10.0
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    # 1.2 Hz = 72 beats per minute
    target_bpm = 72.0
    target_hz = target_bpm / 60.0
    pulse = 0.05 * np.sin(2 * np.pi * target_hz * t)

    # Synthetic RGB with green channel modulation (rPPG physiological response)
    rgb_signals = []
    for p in pulse:
        r = 140.0
        g = 110.0 + (p * 50.0)
        b = 90.0
        rgb_signals.append([r, g, b])

    estimator = HeartRateEstimator(fs=fs)
    bvp = estimator._compute_pos(rgb_signals)
    pred_bpm, sqi = estimator._extract_hr_welch(bvp)

    assert pred_bpm is not None, "Cardiac peak extraction failed"
    assert abs(pred_bpm - target_bpm) <= 1.0, f"Expected {target_bpm} bpm, got {pred_bpm}"
    assert sqi > 0.3, f"Cardiac SQI should be positive, got {sqi}"
