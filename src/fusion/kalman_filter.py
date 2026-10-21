"""
kalman_filter.py — per-patient Kalman smoothing on fused HR and RR
Applies physiological rate-of-change constraints.
TAA flag fed in as external input to suppress cough/motion phase spikes.

Model: scalar random walk. State x (bpm), variance P.
    predict:  P += q * dt
    update:   K = P / (P + R),  x += K * clamp(z - x),  P *= (1 - K)
R comes from the measurement's SQI (low quality -> large R -> small trust).
The innovation is clamped to max_rate_bpm_per_s * dt so one bad window cannot
teleport the estimate (real HR/RR cannot jump 40 bpm in one second).
While taa_active, R is inflated so asynchronous-breathing artefacts barely move RR.

A gap longer than reset_after_s resets the filter, and the output is None while
there is no current measurement — the filter never invents a value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class KalmanConfig:
    q_per_s: float = 0.5               # process noise (bpm^2 per second)
    r_min: float = 1.0                 # measurement noise at sqi=1 (bpm^2)
    r_max: float = 25.0                # measurement noise at sqi=0 (bpm^2)
    max_rate_bpm_per_s: float = 3.0    # physiological slew limit
    reset_after_s: float = 15.0
    taa_r_inflation: float = 4.0


RR_KALMAN = KalmanConfig(q_per_s=0.2, r_min=0.8, r_max=16.0, max_rate_bpm_per_s=1.0)
HR_KALMAN = KalmanConfig(q_per_s=1.0, r_min=2.0, r_max=36.0, max_rate_bpm_per_s=3.0)


class VitalKalman:
    def __init__(self, cfg: Optional[KalmanConfig] = None):
        self.cfg = cfg or KalmanConfig()
        self.x: Optional[float] = None
        self.P: float = 0.0
        self.t: Optional[float] = None

    def reset(self) -> None:
        self.x, self.P, self.t = None, 0.0, None

    def update(self, ts: float, z: Optional[float], sqi: float,
               taa_active: bool = False) -> Optional[float]:
        if z is None or not math.isfinite(z):
            if self.t is not None and (ts - self.t) > self.cfg.reset_after_s:
                self.reset()
            return None
        if self.x is None or self.t is None or (ts - self.t) > self.cfg.reset_after_s:
            self.x, self.P, self.t = float(z), self.cfg.r_max, ts
            return self.x
        dt = max(1e-3, ts - self.t)
        self.P += self.cfg.q_per_s * dt
        q = min(1.0, max(0.0, float(sqi)))
        R = self.cfg.r_max - q * (self.cfg.r_max - self.cfg.r_min)
        if taa_active:
            R *= self.cfg.taa_r_inflation
        K = self.P / (self.P + R)
        limit = self.cfg.max_rate_bpm_per_s * dt
        innov = max(-limit, min(limit, float(z) - self.x))
        self.x += K * innov
        self.P *= (1.0 - K)
        self.t = ts
        return self.x
