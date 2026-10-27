"""
Exposure locking in D435Stream, exercised with fake sensor/pipeline objects
so it runs without a camera attached.
"""
import pytest

rs = pytest.importorskip("pyrealsense2")

from src.capture.d435_stream import D435Stream


class _Range:
    def __init__(self, lo, hi, step=1.0):
        self.min, self.max, self.step = lo, hi, step


class FakeSensor:
    RANGES = {
        rs.option.exposure: _Range(1, 10000),
        rs.option.gain: _Range(0, 128),
        rs.option.white_balance: _Range(2800, 6500, 10),
    }

    def __init__(self):
        self.options = {}
        self.history = []

    def supports(self, option):
        return option in self.RANGES or option in (
            rs.option.enable_auto_exposure,
            rs.option.enable_auto_white_balance,
            rs.option.auto_exposure_priority,
        )

    def get_option_range(self, option):
        return self.RANGES[option]

    def set_option(self, option, value):
        self.options[option] = value
        self.history.append((option, value))

    def get_option(self, option):
        return self.options.get(option, 0)


class FakeColorFrame:
    def __init__(self, meta):
        self.meta = meta

    def __bool__(self):
        return True

    def supports_frame_metadata(self, key):
        return key in self.meta

    def get_frame_metadata(self, key):
        return self.meta[key]


class FakeFrames:
    def __init__(self, meta):
        self.color = FakeColorFrame(meta)

    def get_color_frame(self):
        return self.color


class FakePipeline:
    def __init__(self, meta):
        self.meta = meta

    def wait_for_frames(self, timeout_ms=0):
        return FakeFrames(self.meta)


CFG = {
    "locked_exposure": 156,
    "locked_gain": 64,
    "locked_white_balance": 4600,
    "auto_lock_settle_s": 0.05,
}


def _stream(mode, meta):
    s = D435Stream({**CFG, "exposure_mode": mode})
    s.pipeline = FakePipeline(meta)
    return s


def test_auto_lock_freezes_metered_values():
    md = rs.frame_metadata_value
    s = _stream("auto_lock", {md.actual_exposure: 250, md.gain_level: 40, md.white_balance: 3904})
    sensor = FakeSensor()
    applied = s._lock_rgb_settings(sensor, fps=30)

    assert applied["exposure"] == 250
    assert applied["gain"] == 40
    assert applied["white_balance"] == 3900          # snapped to the 10 K step
    # auto ends OFF, and the metered values are written after it is switched off
    assert sensor.options[rs.option.enable_auto_exposure] == 0
    assert sensor.options[rs.option.enable_auto_white_balance] == 0
    off_at = sensor.history.index((rs.option.enable_auto_exposure, 0))
    assert sensor.history.index((rs.option.exposure, 250)) > off_at


def test_auto_lock_caps_exposure_to_one_frame():
    md = rs.frame_metadata_value
    s = _stream("auto_lock", {md.actual_exposure: 900})
    applied = s._lock_rgb_settings(FakeSensor(), fps=30)
    assert applied["exposure"] <= 10000 / 30          # 33.3 ms keeps 30 fps


def test_auto_lock_falls_back_when_metadata_missing():
    s = _stream("auto_lock", {})
    applied = s._lock_rgb_settings(FakeSensor(), fps=30)
    assert applied["exposure"] == 156
    assert applied["white_balance"] == 4600


def test_frames_raise_after_repeated_timeouts():
    class DeadPipeline:
        def wait_for_frames(self, timeout_ms=0):
            raise RuntimeError("Frame didn't arrive within 5000")

    s = D435Stream(CFG)
    s.pipeline, s._running = DeadPipeline(), True
    with pytest.raises(RuntimeError, match="no frames"):
        next(s.frames())


def test_fixed_mode_never_enables_auto():
    s = _stream("fixed", {})
    sensor = FakeSensor()
    applied = s._lock_rgb_settings(sensor, fps=30)
    assert (rs.option.enable_auto_exposure, 1) not in sensor.history
    assert applied == {"mode": "fixed", "exposure": 156, "gain": 64, "white_balance": 4600}
