import time

import numpy as np

from src.types import FrameBundle, Measurement


def test_measurement_none_value_is_valid():
    measurement = Measurement(
        value=None,
        sqi=0.0,
        ts=time.monotonic(),
        source="test",
        latency_ms=0.0,
    )
    assert measurement.value is None


def test_measurement_sqi_bounds():
    measurement = Measurement(
        value=1,
        sqi=0.8,
        ts=time.monotonic(),
        source="test",
        latency_ms=0.0,
    )
    assert 0.0 <= measurement.sqi <= 1.0


def test_frame_bundle_thermal_optional():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)

    fb = FrameBundle(
        rgb=rgb,
        depth=depth,
        ir=None,
        ts_mono=time.monotonic(),
    )

    assert fb.rgb is rgb
    assert fb.depth is depth
    assert fb.ir is None
