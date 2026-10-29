"""
test_thermal_stream.py — the ESP32 thermal reader must never overload the board.

The ESP32-S3 powers off when it receives bursts of requests to unknown paths (RAM
exhaustion). These tests drive ThermalStream against a fake board and check it only ever
asks for the one configured endpoint, paces itself, rebuilds its session after a failure
and backs off while the board is unreachable.
"""
import threading
import time

import numpy as np
import requests

from src.capture.thermal_stream import ThermalStream

FRAME = (np.full((62, 80), 3103, np.int16)).tobytes()    # 31.03 C everywhere


class FakeBoard:
    """Counts every request; can be switched off (connection errors) and back on."""

    def __init__(self):
        self.urls, self.sessions, self.up = [], 0, True
        self.lock = threading.Lock()

    def factory(self):
        board = self

        class Session:
            def __init__(self):
                with board.lock:
                    board.sessions += 1

            def get(self, url, timeout=0):
                with board.lock:
                    board.urls.append((time.monotonic(), url))
                if not board.up:
                    raise requests.exceptions.ConnectionError("board off")
                r = requests.models.Response()
                r.status_code, r._content = 200, FRAME
                return r

            def close(self):
                pass
        return Session()


CFG = {"base_ip": "http://192.168.4.1", "endpoint": "/thermal/frame", "timeout_s": 0.2,
       "retry_delay_s": 0.05, "max_backoff_s": 0.2, "max_poll_hz": 20}


def _run(board, seconds, cfg=CFG):
    s = ThermalStream(cfg, session_factory=board.factory)
    s.start()
    time.sleep(seconds)
    return s


def test_decodes_frames_and_reports_online():
    board = FakeBoard()
    s = _run(board, 0.4)
    try:
        frame, ts = s.get_latest_frame()
        assert frame.shape == (62, 80) and abs(float(frame[0, 0]) - 31.03) < 1e-4
        assert s.is_online() and ts is not None
    finally:
        s.stop()


def test_only_the_configured_endpoint_is_ever_requested():
    board = FakeBoard()
    board.up = False                                   # unreachable at start: no scanning allowed
    s = _run(board, 0.8)
    board.up = True
    time.sleep(0.5)
    s.stop()
    assert board.urls, "it should keep retrying"
    assert {u for _, u in board.urls} == {"http://192.168.4.1/thermal/frame"}
    assert s.frames_ok > 0                              # picked up again once the board was back


def test_polling_is_paced():
    board = FakeBoard()
    s = _run(board, 1.0)                               # max 20 Hz
    s.stop()
    times = [t for t, _ in board.urls]
    rate = (len(times) - 1) / (times[-1] - times[0])
    assert rate <= 22, f"polled at {rate:.1f} Hz"


def test_backs_off_and_rebuilds_the_session_while_unreachable():
    board = FakeBoard()
    board.up = False
    s = _run(board, 1.0)                               # back-off 0.05 -> 0.1 -> 0.2 -> 0.2 ...
    s.stop()
    n = len(board.urls)
    assert 3 <= n <= 10, f"{n} requests in 1 s while unreachable"
    assert board.sessions >= n                          # a fresh session after every failure


def test_endpoint_wins_over_a_stale_full_url():
    s = ThermalStream({**CFG, "endpoint": "/new/frame", "full_url": "http://192.168.4.1/thermal/frame"},
                      session_factory=FakeBoard().factory)
    assert s.url == "http://192.168.4.1/new/frame"


def test_stop_during_a_request_does_not_crash_the_thread():
    board = FakeBoard()
    s = _run(board, 0.2)
    s._close_session()                                  # what stop() does from another thread
    time.sleep(0.2)
    assert s._thread.is_alive()                         # it simply reconnects
    s.stop()


def test_wrong_size_payload_is_not_a_frame():
    s = ThermalStream(CFG, session_factory=FakeBoard().factory)
    assert s._decode_frame(b"\x00" * 100) is None


# ── the thermal camera is independent of the D435 ─────────────────────────────────

def test_a_d435_that_fails_to_open_does_not_stop_a_borrowed_thermal_stream():
    import pytest
    from src.capture.sync_buffer import SyncBuffer, load_camera_config
    board = FakeBoard()
    s = _run(board, 0.2)
    try:
        buf = SyncBuffer(load_camera_config("config/camera.yaml"), thermal=s)

        def no_camera():
            raise RuntimeError("No device connected")
        buf._d435.start = no_camera
        with pytest.raises(RuntimeError):
            buf.start()
        buf._d435.stop = lambda: None
        buf.stop()
        n = len(board.urls)
        time.sleep(0.3)
        assert len(board.urls) > n and s.is_online()   # still polling after the D435 gave up
    finally:
        s.stop()


def test_engine_shows_thermal_while_the_d435_is_down():
    import types
    from src.serving.engine import CameraEngine
    board = FakeBoard()
    s = _run(board, 0.2)
    published = []
    eng = types.SimpleNamespace(_stop=threading.Event(), _thermal_stream=s, _last_frame_wall=0.0,
                                _publish_thermal=lambda f, roi, now: published.append(roi))
    th = threading.Thread(target=CameraEngine._thermal_publish_loop, args=(eng,), daemon=True)
    th.start()
    try:
        time.sleep(0.5)
        assert published and all(r is None for r in published)
        eng._last_frame_wall = time.time() + 60          # D435 running: the frame loop publishes
        n = len(published)
        time.sleep(0.4)
        assert len(published) == n
    finally:
        eng._stop.set()
        th.join(timeout=2)
        s.stop()


def test_d435_errors_get_plain_advice():
    from src.serving.engine import d435_hint
    assert "Another program" in d435_hint("xioctl(VIDIOC_S_FMT) failed: Device or resource busy")
    assert "Another program" in d435_hint("failed to set power state")
    assert "not detected" in d435_hint("No device connected")
    assert "USB 3" in d435_hint("Couldn't resolve requests")
    assert d435_hint("something new")
