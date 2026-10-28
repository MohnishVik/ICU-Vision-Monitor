"""test_serving.py — the web API in demo mode and in camera mode with no hardware attached."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.serving.engine import EngineManager


def _wait(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture(scope="module")
def demo_client():
    manager = EngineManager.from_args(demo=True)
    app = create_app(manager, frontend_dir="does/not/exist")
    with TestClient(app) as client:
        assert _wait(lambda: all(p["version"] for p in client.get("/api/patients").json()))
        yield client


def _strict(obj):
    json.dumps(obj, allow_nan=False)            # raises if any NaN/inf leaked


def test_health_and_patients(demo_client):
    h = demo_client.get("/api/health").json()
    assert h["mode"] == "demo" and len(h["patients"]) == 4
    pts = demo_client.get("/api/patients").json()
    assert [p["patient_id"] for p in pts] == ["bed-1", "bed-2", "bed-3", "bed-4"]
    assert all(p["simulated"] for p in pts)
    _strict(pts)


def test_vitals_snapshot_contract(demo_client):
    s = demo_client.get("/api/patients/bed-1/vitals").json()
    _strict(s)
    for key in ("vitals", "ews", "fall", "taa", "pain", "scene", "alarms", "trends", "modules",
                "metrics", "version"):
        assert key in s
    assert set(s["vitals"]) == {"rr", "hr"}
    assert s["pain"]["available"] is True and s["pain"]["experimental"] is True
    assert {"rr.depth", "rr.cnn_lbrd", "rr.cnn_d435"} <= set(s["vitals"]["rr"]["sources"])
    # signal.yaml fuses only EfficientPhys; both breathing CNNs (constant-output) are display-only
    assert s["vitals"]["rr"]["sources"]["rr.cnn_lbrd"]["role"] == "display"
    assert s["vitals"]["rr"]["sources"]["rr.cnn_d435"]["role"] == "display"
    assert s["vitals"]["hr"]["sources"]["hr.efficientphys"]["role"] == "fused"
    assert len(s["scene"]["keypoints"]) == 17 and len(s["scene"]["chair"]) == 4


def test_models_endpoint(demo_client):
    m = demo_client.get("/api/models").json()
    files = {r["file"] for r in m["files"]}
    assert "models/public/yolov8n-pose.pt" in files and "missing_required" in m
    _strict(m)


def test_history(demo_client):
    time.sleep(1.2)
    h = demo_client.get("/api/patients/bed-1/history?minutes=5").json()
    assert len(h["t"]) == len(h["rr"]) == len(h["hr"]) >= 1
    _strict(h)


def test_unknown_patient_404(demo_client):
    assert demo_client.get("/api/patients/nope/vitals").status_code == 404


def test_fall_alarm_latches_and_ack(demo_client):
    # bed-3 starts 130 s into its 300 s cycle: the fall happens ~20 s later. Drive it directly.
    src_alarms = lambda: demo_client.get("/api/patients/bed-3/vitals").json()["alarms"]  # noqa: E731
    r = demo_client.post("/api/patients/bed-3/ack", json={"alarm_id": None})
    assert r.status_code == 200 and "acknowledged" in r.json()
    assert isinstance(src_alarms(), list)


def test_calibration_roundtrip(demo_client):
    st = demo_client.get("/api/patients/bed-2/calibration").json()
    assert st["calibrated"] is True and len(st["corners"]) == 4
    frame = demo_client.get("/api/patients/bed-2/calibration/frame")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    bad = demo_client.post("/api/patients/bed-2/calibration",
                           json={"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert bad.status_code == 422                       # degenerate quad rejected
    ok = demo_client.post("/api/patients/bed-2/calibration",
                          json={"corners": [[200, 200], [420, 200], [430, 400], [190, 400]]})
    assert ok.status_code == 200 and ok.json()["calibrated"]


def test_frame_jpg(demo_client):
    assert _wait(lambda: demo_client.get("/api/patients/bed-1/frame.jpg").status_code == 200)


def test_mjpeg_over_real_socket_and_disconnect_cleanup():
    """An endless MJPEG stream can't be closed by TestClient, so use a real uvicorn server."""
    import socket
    import threading

    import httpx
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    manager = EngineManager.from_args(demo=True)
    server = uvicorn.Server(uvicorn.Config(create_app(manager, frontend_dir="does/not/exist"),
                                           host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert _wait(lambda: _ok(f"{base}/api/health"), timeout=10)
        src = manager.get("bed-1")
        with httpx.stream("GET", f"{base}/api/patients/bed-1/stream.mjpg", timeout=10) as r:
            assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
            buf, clients_mid = b"", None
            for chunk in r.iter_bytes():
                buf += chunk
                frames = buf.count(b"--frame")
                if frames >= 3 and clients_mid is None:
                    clients_mid = src.stream_clients            # sampled while still streaming
                if frames >= 6:
                    break
            frames = buf.count(b"--frame")
            assert frames >= 6, f"stream ended early after {frames} frames"
            assert b"Content-Type: image/jpeg" in buf
            assert clients_mid == 1, f"viewer count while streaming was {clients_mid}"
        assert _wait(lambda: src.stream_clients == 0, timeout=5)   # server noticed the hang-up
    finally:
        server.should_exit = True
        th.join(timeout=10)


def _ok(url):
    import httpx
    try:
        return httpx.get(url, timeout=1).status_code == 200
    except Exception:
        return False


def test_websockets(demo_client):
    with demo_client.websocket_connect("/ws/stream/bed-4") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "tick" and msg["data"]["patient_id"] == "bed-4"
    with demo_client.websocket_connect("/ws/ward") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "ward" and len(msg["data"]) == 4


def test_spa_fallback_when_not_built(demo_client):
    r = demo_client.get("/patients/bed-1")
    assert r.status_code == 200 and "npm run build" in r.text


def _realsense_attached() -> bool:
    try:
        import pyrealsense2 as rs
        return len(rs.context().query_devices()) > 0
    except Exception:  # noqa: BLE001 - no driver means no camera
        return False


@pytest.mark.skipif(_realsense_attached(), reason="a RealSense camera is plugged in, so camera mode opens it")
def test_camera_mode_without_hardware_degrades_cleanly(tmp_path):
    """No pyrealsense2 / no camera / no weights: server stays up and reports why."""
    manager = EngineManager.from_args(demo=False, patient_id="bed-1", label="Bed 1")
    app = create_app(manager, frontend_dir="does/not/exist")
    with TestClient(app) as client:
        assert _wait(lambda: client.get("/api/patients/bed-1/vitals").json()["status"]
                     == "camera_offline", timeout=90)   # cold torch import can be slow
        s = client.get("/api/patients/bed-1/vitals").json()
        _strict(s)
        assert s["status_detail"]
        mods = s["modules"]
        assert mods["pain"]["state"] in ("ok", "degraded", "unavailable")   # weights are git-ignored
        assert mods["rr_learned"]["state"] in ("ok", "degraded", "unavailable")
        assert mods["hr_learned"]["state"] in ("ok", "degraded", "unavailable")
        assert mods["fall"]["state"] in ("unavailable", "degraded", "ok")
        assert mods["pose"]["state"] in ("unavailable", "ok")
        h = client.get("/api/health").json()
        assert h["status"] == "degraded"
        assert client.get("/api/patients/bed-1/calibration/frame").status_code == 503
        r = client.post("/api/patients/bed-1/calibration",
                        json={"corners": [[200, 200], [420, 200], [430, 400], [190, 400]]})
        assert r.status_code == 409                      # no frame yet -> clear message, no crash


# ------------------------------------------------------------------ thermal (ESP32) --
def _demo_pairs():
    """Point pairs consistent with the demo rig (thermal x7 + offset into RGB)."""
    th = [[8, 6], [72, 7], [74, 56], [9, 54], [40, 30], [24, 44]]
    return [{"thermal": t, "rgb": [7.0 * t[0] + 40.0, 7.1 * t[1] + 20.0]} for t in th]


def test_thermal_payload_contract(demo_client):
    assert _wait(lambda: demo_client.get("/api/patients/bed-1/thermal").json()["online"])
    p = demo_client.get("/api/patients/bed-1/thermal").json()
    _strict(p)
    assert (p["w"], p["h"]) == (80, 62) and len(p["data"]) == 80 * 62
    assert 15.0 < p["min"] < p["max"] < 45.0
    assert p["roi_source"] == "nostril" and len(p["roi"]) == 4
    assert len(p["trace"]["t"]) == len(p["trace"]["c"]) >= 1
    assert p["registration"]["registered"] and p["registration"]["quality"] == "good"
    s = demo_client.get("/api/patients/bed-1/vitals").json()["scene"]
    assert s["thermal_registered"] and s["thermal_roi"] == p["roi"] and len(s["nostril_box"]) == 4


def test_thermal_snapshot_alignment_overlay_and_removal(demo_client):
    snap = demo_client.post("/api/patients/bed-2/thermal/snapshot")
    assert snap.status_code == 200
    sj = snap.json()
    assert sj["rgb_size"] == [480, 640] and len(sj["thermal"]["data"]) == 80 * 62
    img = demo_client.get(f"/api/patients/bed-2/thermal/snapshot/{sj['snapshot_id']}/rgb.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert demo_client.get("/api/patients/bed-2/thermal/snapshot/nope/rgb.jpg").status_code == 404

    ok = demo_client.post("/api/patients/bed-2/thermal/registration", json={"pairs": _demo_pairs()})
    assert ok.status_code == 200 and ok.json()["registered"] and ok.json()["error_px"] < 1.0

    bad = _demo_pairs()
    bad[2]["rgb"] = [60.0, 440.0]
    r = demo_client.post("/api/patients/bed-2/thermal/registration", json={"pairs": bad})
    assert r.status_code == 422
    assert ("disagree" in r.json()["detail"]) or ("fold" in r.json()["detail"])
    few = demo_client.post("/api/patients/bed-2/thermal/registration", json={"pairs": _demo_pairs()[:3]})
    assert few.status_code == 422

    ov = demo_client.get("/api/patients/bed-2/thermal/overlay.jpg")
    assert ov.status_code == 200 and ov.headers["content-type"] == "image/jpeg"

    gone = demo_client.delete("/api/patients/bed-2/thermal/registration").json()
    assert gone["registered"] is False
    assert _wait(lambda: demo_client.get("/api/patients/bed-2/thermal").json()["roi_source"]
                 == "warmest_patch")
    assert demo_client.get("/api/patients/bed-2/thermal/overlay.jpg").status_code == 409
    # restore for any later test
    demo_client.post("/api/patients/bed-2/thermal/registration", json={"pairs": _demo_pairs()})


def test_thermal_without_hardware_is_explicit():
    manager = EngineManager.from_args(demo=False, patient_id="bed-9", label="Bed 9")
    app = create_app(manager, frontend_dir="does/not/exist")
    with TestClient(app) as client:
        p = client.get("/api/patients/bed-9/thermal").json()
        _strict(p)
        assert p["online"] is False and "data" not in p
        assert client.post("/api/patients/bed-9/thermal/snapshot").status_code == 409
        assert client.get("/api/patients/bed-9/thermal/overlay.jpg").status_code == 409
