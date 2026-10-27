"""
Exposure locking in D435Stream, exercised with a simulated camera so it runs
without hardware. The fake sensor renders frames whose brightness follows
exposure x gain (plus gamma / brightness lift) and whose red/blue balance
follows the white-balance setting, like the real colour sensor.
"""
import numpy as np
import pytest

rs = pytest.importorskip("pyrealsense2")

from src.capture.d435_stream import D435Stream


class _Range:
    def __init__(self, lo, hi, step=1.0, default=0.0):
        self.min, self.max, self.step, self.default = lo, hi, step, default


class FakeSensor:
    RANGES = {
        rs.option.exposure: _Range(1, 10000),
        rs.option.gain: _Range(0, 128),
        rs.option.white_balance: _Range(2800, 6500, 10),
        rs.option.gamma: _Range(100, 500, 1),
        rs.option.brightness: _Range(-64, 64, 1),
        rs.option.enable_auto_exposure: _Range(0, 1),
        rs.option.enable_auto_white_balance: _Range(0, 1),
        rs.option.auto_exposure_priority: _Range(0, 1),
    }

    def __init__(self, scene=1.0, auto_luma=110.0, auto_rb=1.0):
        self.options = {rs.option.exposure: 156.0, rs.option.gain: 64.0, rs.option.white_balance: 4600.0,
                        rs.option.gamma: 300.0, rs.option.brightness: 0.0}
        self.history = []
        self.scene, self.auto_luma, self.auto_rb = scene, auto_luma, auto_rb

    def supports(self, option):
        return option in self.RANGES

    def get_option_range(self, option):
        return self.RANGES[option]

    def set_option(self, option, value):
        self.options[option] = value
        self.history.append((option, value))

    def get_option(self, option):
        return self.options.get(option, 0)

    # how the simulated camera renders a frame under the current settings
    def render(self):
        o = self.options
        if o.get(rs.option.enable_auto_exposure):
            luma, rb = self.auto_luma, self.auto_rb
        else:
            luma = self.scene * o[rs.option.exposure] * o[rs.option.gain] / 400.0
            luma *= (o[rs.option.gamma] / 300.0) ** 2
            luma = luma + o[rs.option.brightness]
            rb = (o[rs.option.white_balance] / 4600.0) ** 1.5
        luma = float(np.clip(luma, 0, 255))
        b = luma / (0.114 + 0.587 + 0.299 * rb)
        img = np.empty((8, 8, 3), np.float64)
        img[..., 0], img[..., 1], img[..., 2] = b, b, b * rb
        return np.clip(img, 0, 255)


class FakeColorFrame:
    def __init__(self, img):
        self.img = img

    def __bool__(self):
        return True

    def get_data(self):
        return self.img


class FakeFrames:
    def __init__(self, img):
        self.color = FakeColorFrame(img)

    def get_color_frame(self):
        return self.color


class FakePipeline:
    def __init__(self, sensor):
        self.sensor = sensor

    def wait_for_frames(self, timeout_ms=0):
        return FakeFrames(self.sensor.render())


CFG = {
    "locked_exposure": 156,
    "locked_gain": 64,
    "locked_white_balance": 4600,
    "auto_lock_settle_s": 0.02,
}


def _lock(mode, sensor, **cfg):
    s = D435Stream({**CFG, **cfg, "exposure_mode": mode})
    s.pipeline = FakePipeline(sensor)
    return s._lock_rgb_settings(sensor, fps=30), sensor


def _luma(sensor):
    return D435Stream._luma(sensor.render())


def test_auto_lock_reaches_auto_brightness_with_auto_off():
    applied, sensor = _lock("auto_lock", FakeSensor(scene=2.0, auto_luma=110))
    assert sensor.options[rs.option.enable_auto_exposure] == 0
    assert sensor.options[rs.option.enable_auto_white_balance] == 0
    assert abs(_luma(sensor) - 110) <= 0.06 * 110
    assert applied["exposure"] <= 10000 / 30          # 33.3 ms keeps 30 fps


def test_dim_room_lifts_gamma_at_the_exposure_cap():
    # even at 33 ms and full gain this room renders darker than the camera's auto mode
    applied, sensor = _lock("auto_lock", FakeSensor(scene=0.2, auto_luma=90))
    assert applied["exposure"] == pytest.approx(333, abs=1)
    assert applied["gain"] == 128
    assert applied["gamma"] == 500                     # gamma alone is not enough here...
    assert applied["brightness"] > 0                   # ...so the offset makes up the rest
    assert abs(_luma(sensor) - 90) <= 0.06 * 90


def test_bright_room_lowers_exposure():
    applied, sensor = _lock("auto_lock", FakeSensor(scene=20.0, auto_luma=100))
    assert applied["exposure"] < 156
    assert abs(_luma(sensor) - 100) <= 0.06 * 100


def test_white_balance_matches_auto_colour_cast():
    applied, sensor = _lock("auto_lock", FakeSensor(scene=2.0, auto_rb=1.2))
    rb = D435Stream._red_blue(sensor.render())
    assert rb == pytest.approx(1.2, rel=0.03)
    assert applied["white_balance"] > 4600


def test_target_luma_override():
    _, sensor = _lock("auto_lock", FakeSensor(scene=2.0, auto_luma=60), target_luma=120)
    assert abs(_luma(sensor) - 120) <= 0.06 * 120


def test_fixed_mode_never_enables_auto_and_resets_tone():
    sensor = FakeSensor()
    sensor.options[rs.option.gamma] = 480.0               # left over from a previous run
    applied, sensor = _lock("fixed", sensor)
    assert (rs.option.enable_auto_exposure, 1) not in sensor.history
    assert sensor.options[rs.option.gamma] == 300
    assert applied == {"mode": "fixed", "exposure": 156, "gain": 64, "white_balance": 4600}


def test_frames_raise_after_repeated_timeouts():
    class DeadPipeline:
        def wait_for_frames(self, timeout_ms=0):
            raise RuntimeError("Frame didn't arrive within 5000")

    s = D435Stream(CFG)
    s.pipeline, s._running = DeadPipeline(), True
    with pytest.raises(RuntimeError, match="no frames"):
        next(s.frames())
