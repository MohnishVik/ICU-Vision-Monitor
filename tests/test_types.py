"""test_types.py — Measurement contract validation"""
import pytest
from src.types import Measurement, FrameBundle, ROIBundle
import numpy as np, time

def test_measurement_none_value_is_valid():
    m = Measurement(value=None, sqi=0.0, ts=time.time(),
                    source="rr.depth", latency_ms=5.0)
    assert m.value is None

def test_measurement_sqi_bounds():
    for sqi in [0.0, 0.5, 1.0]:
        m = Measurement(value=15.0, sqi=sqi, ts=time.time(),
                        source="hr.pos", latency_ms=2.0)
        assert 0.0 <= m.sqi <= 1.0

def test_frame_bundle_thermal_optional():
    rgb   = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640),    dtype=np.uint16)
    fb = FrameBundle(rgb=rgb, depth=depth, ir=None, thermal=None, ts_mono=time.monotonic())
    assert fb.thermal is None
