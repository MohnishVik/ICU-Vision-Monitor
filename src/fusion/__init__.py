"""Fusion layer: combine per-module Measurements into clinically usable vitals.

    inv_variance  -> SQI-gated inverse-variance fusion + staleness watchdog
    kalman_filter -> physiologically constrained smoothing
    trend         -> 5/15/60/240-minute slope, mean, variability
    alarm_fsm     -> NEWS2-style sub-score, escalation state machine, explanations
"""
from __future__ import annotations

from .alarm_fsm import (
    Alarm,
    AlarmManager,
    Condition,
    EWSResult,
    EWSScorer,
    explain,
    load_rules,
    parse_range,
)
from .inv_variance import FusionConfig, StalenessWatchdog, fuse
from .kalman_filter import HR_KALMAN, RR_KALMAN, KalmanConfig, VitalKalman
from .trend import DEFAULT_WINDOWS_S, TrendStats, TrendTracker

__all__ = [
    "Alarm", "AlarmManager", "Condition", "EWSResult", "EWSScorer", "explain", "load_rules",
    "parse_range", "FusionConfig", "StalenessWatchdog", "fuse", "HR_KALMAN", "RR_KALMAN",
    "KalmanConfig", "VitalKalman", "DEFAULT_WINDOWS_S", "TrendStats", "TrendTracker",
]
