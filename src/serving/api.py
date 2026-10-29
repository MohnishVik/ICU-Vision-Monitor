"""
api.py — FastAPI REST endpoints + static frontend serving
Serves compiled frontend/dist/ as static files (no separate web server needed).

Endpoints (all under /api, matching frontend/vite.config.ts):
  GET    /health                          — module liveness + stale-value check
  GET    /patients/{id}/session          — recording status; POST starts ({label, condition, reference})
  POST   /patients/{id}/session/reference — add a reference reading ({rr?, hr?, note})
  POST   /patients/{id}/session/event     — mark an event ({kind, value?, note}); kinds from
                                             GET /session-event-kinds
  DELETE /patients/{id}/session          — stop recording
  GET    /sessions                        — recorded sessions (newest first)
  GET    /sessions/{sid}/report           — MAE + coverage per source (?warmup_s=&settle_s=)
  GET    /models                          — expected weight files and whether each is present
  GET    /patients                        — active patient list + current states
  GET    /patients/{id}/vitals            — latest snapshot (vitals, EWS, alarms, scene)
  GET    /patients/{id}/history           — time-series for trend charts (?minutes=&max_points=)
  POST   /patients/{id}/ack               — nurse acknowledges alarm ({"alarm_id": int|null})
  GET    /patients/{id}/calibration       — chair calibration status
  GET    /patients/{id}/calibration/frame — still JPEG (faces blurred) to click the corners on
  POST   /patients/{id}/calibration       — {"corners": [[x,y] x4], "label": "chair"|"bed"}
  POST   /patients/{id}/calibration/reuse — load this bed's last saved calibration
  DELETE /patients/{id}/calibration       — clear (re-calibrate)
  GET    /patients/{id}/thermal           — thermal grid, breathing ROI, breath trace, alignment
  GET    /patients/{id}/thermal/registration — thermal<->RGB alignment status
  POST   /patients/{id}/thermal/registration — {"pairs": [{"rgb": [x,y], "thermal": [x,y]}, ...]}
  DELETE /patients/{id}/thermal/registration — remove alignment (back to warmest spot)
  POST   /patients/{id}/thermal/snapshot  — freeze a colour + thermal pair for marking points
  GET    /patients/{id}/thermal/snapshot/{sid}/rgb.jpg — the frozen colour frame
  GET    /patients/{id}/thermal/overlay.jpg — colour frame with the aligned thermal on top
  GET    /patients/{id}/frame.jpg         — latest frame (faces blurred)
  GET    /patients/{id}/stream.mjpg       — live MJPEG (faces blurred)
WebSockets (src/serving/websocket.py): /ws/stream/{id}, /ws/ward
"""
from __future__ import annotations

import asyncio
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.serving import __version__
from src.serving.engine import STREAM_FPS, EngineManager, PatientSource
from src.serving.websocket import build_ws_router, summarize

_NO_CACHE = {"Cache-Control": "no-store, max-age=0"}


class AckBody(BaseModel):
    alarm_id: Optional[int] = None


class CalibrationBody(BaseModel):
    corners: List[List[float]] = Field(..., min_length=4, max_length=4)
    label: str = "chair"


class PointPair(BaseModel):
    rgb: List[float] = Field(..., min_length=2, max_length=2)
    thermal: List[float] = Field(..., min_length=2, max_length=2)


class Reference(BaseModel):
    rr: Optional[float] = Field(None, ge=0, le=80)
    hr: Optional[float] = Field(None, ge=20, le=250)


class SessionBody(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    condition: str = Field("", max_length=80)
    notes: str = Field("", max_length=500)
    reference: Reference = Reference()
    capture: bool = False          # also write frames.csv (training capture, numbers only)


class ReferenceBody(Reference):
    note: str = Field("", max_length=200)


class EventBody(BaseModel):
    kind: str = Field(..., min_length=1, max_length=40)
    value: Optional[float] = None      # e.g. a CPOT (0-8) or BPS (3-12) score for pain_score
    note: str = Field("", max_length=200)


_SID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")


class FusionBody(BaseModel):
    source: str = Field(..., min_length=3, max_length=40)
    fused: Optional[bool] = None       # None = back to the signal.yaml default


class AlignmentBody(BaseModel):
    pairs: List[PointPair] = Field(..., min_length=1, max_length=64)


def _source(manager: EngineManager, patient_id: str) -> PatientSource:
    try:
        return manager.get(patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'") from None


def build_api_router(manager: EngineManager) -> APIRouter:
    r = APIRouter(prefix="/api")

    @r.get("/health")
    def health() -> dict:
        patients = []
        degraded = False
        for s in manager.sources.values():
            snap = s.snapshot()
            mods = snap.get("modules") or {}
            vit = snap.get("vitals") or {}
            stale = {k: bool((v or {}).get("stale", True)) for k, v in vit.items()}
            if snap.get("status") != "running" or any(
                    (m or {}).get("state") in ("error", "unavailable") for m in mods.values()):
                degraded = True
            patients.append({"patient_id": s.patient_id, "status": snap.get("status"),
                             "status_detail": snap.get("status_detail"), "modules": mods,
                             "stale": stale, "metrics": snap.get("metrics")})
        return {"status": "degraded" if degraded else "ok", "version": __version__,
                "mode": manager.mode, "uptime_s": time.time() - manager.started,
                "server_time": time.time(), "patients": patients}

    @r.get("/models")
    def models() -> dict:
        from src.serving.models_check import check_models
        any_src = next(iter(manager.sources.values()), None)
        rows = check_models(".", any_src.signal_cfg if any_src is not None else None)
        return {"mode": manager.mode, "files": rows,
                "missing_required": [r["file"] for r in rows if r["required"] and not r["present"]]}

    # ---- recording sessions + comparison -------------------------------------------
    @r.get("/patients/{patient_id}/session")
    def session_status(patient_id: str) -> dict:
        return _source(manager, patient_id).session_status()

    @r.post("/patients/{patient_id}/session")
    def start_session(patient_id: str, body: SessionBody) -> dict:
        src = _source(manager, patient_id)
        try:
            return src.start_session(body.label, body.condition, body.reference.model_dump(), body.notes,
                                     capture=body.capture)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.get("/patients/{patient_id}/fusion")
    def fusion_get(patient_id: str) -> dict:
        """Which estimators make up the RR and HR on screen, and the signal.yaml defaults."""
        return _source(manager, patient_id).pipeline.fusion_status()

    @r.post("/patients/{patient_id}/fusion")
    def fusion_set(patient_id: str, body: FusionBody) -> dict:
        """Include or exclude one estimator from the fused value (saved per bed)."""
        src = _source(manager, patient_id)
        try:
            return src.set_fusion(body.source, body.fused)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown estimator '{body.source}'") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.post("/patients/{patient_id}/pain/baseline")
    def pain_baseline(patient_id: str) -> dict:
        """Relearn the calm pain-expression baseline (press when the patient is settled)."""
        src = _source(manager, patient_id)
        try:
            return src.reset_pain_baseline()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.get("/session-event-kinds")
    def session_event_kinds() -> dict:
        from src.serving.session_log import EVENT_KINDS
        return {"kinds": list(EVENT_KINDS)}

    @r.post("/patients/{patient_id}/session/event")
    def add_event(patient_id: str, body: EventBody) -> dict:
        src = _source(manager, patient_id)
        try:
            return src.add_session_event(body.kind, body.value, body.note)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @r.post("/patients/{patient_id}/session/reference")
    def add_reference(patient_id: str, body: ReferenceBody) -> dict:
        src = _source(manager, patient_id)
        try:
            return src.add_session_reference(body.rr, body.hr, body.note)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @r.delete("/patients/{patient_id}/session")
    def stop_session(patient_id: str) -> dict:
        src = _source(manager, patient_id)
        try:
            return src.stop_session()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.get("/sessions")
    def sessions() -> list:
        import json as _json

        from src.serving.session_log import sessions_root
        root = sessions_root()
        out = []
        if root.is_dir():
            for d in root.iterdir():
                try:
                    out.append(_json.loads((d / "meta.json").read_text()))
                except Exception:  # noqa: BLE001 - half-written or foreign folder
                    continue
        return sorted(out, key=lambda m: m.get("started") or 0, reverse=True)

    @r.get("/sessions/{session_id}/report")
    def session_report(session_id: str, warmup_s: float = Query(30.0, ge=0, le=600),
                       settle_s: float = Query(20.0, ge=0, le=600)) -> dict:
        from src.evaluation.compare import load_session, score_session, strip_internal
        from src.serving.session_log import sessions_root
        if not _SID.match(session_id):
            raise HTTPException(status_code=400, detail="Invalid session id")
        root = sessions_root().resolve()
        d = (root / session_id).resolve()
        if d.parent != root or not (d / "meta.json").is_file():
            raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
        return strip_internal(score_session(load_session(d), warmup_s=warmup_s, settle_s=settle_s))

    @r.get("/patients")
    def patients() -> list:
        return [summarize(s.snapshot()) for s in manager.sources.values()]

    @r.get("/patients/{patient_id}/vitals")
    def vitals(patient_id: str) -> dict:
        return _source(manager, patient_id).snapshot()

    @r.get("/patients/{patient_id}/history")
    def history(patient_id: str, minutes: float = Query(60.0, gt=0, le=240),
                max_points: int = Query(720, ge=10, le=5000)) -> dict:
        return _source(manager, patient_id).history(minutes, max_points)

    @r.post("/patients/{patient_id}/ack")
    def ack(patient_id: str, body: Optional[AckBody] = None) -> dict:
        n = _source(manager, patient_id).acknowledge(body.alarm_id if body else None)
        return {"acknowledged": n}

    # ---------------------------------------------------------------- calibration
    @r.get("/patients/{patient_id}/calibration")
    def calibration(patient_id: str) -> dict:
        return _source(manager, patient_id).calibration_status()

    @r.get("/patients/{patient_id}/calibration/frame")
    def calibration_frame(patient_id: str) -> Response:
        data = _source(manager, patient_id).calibration_frame()   # type: ignore[attr-defined]
        if data is None:
            raise HTTPException(status_code=503, detail="No camera frame available yet.")
        return Response(content=data, media_type="image/jpeg", headers=_NO_CACHE)

    @r.post("/patients/{patient_id}/calibration")
    def set_calibration(patient_id: str, body: CalibrationBody) -> dict:
        src = _source(manager, patient_id)
        try:
            return src.set_calibration(body.corners, body.label)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.post("/patients/{patient_id}/calibration/reuse")
    def reuse_calibration(patient_id: str) -> dict:
        try:
            return _source(manager, patient_id).reuse_calibration()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @r.delete("/patients/{patient_id}/calibration")
    def clear_calibration(patient_id: str) -> dict:
        return _source(manager, patient_id).clear_calibration()

    # ---------------------------------------------------------------- thermal (ESP32)
    @r.get("/patients/{patient_id}/thermal")
    def thermal(patient_id: str) -> dict:
        """Latest thermal grid (centi-degC), breathing ROI, breath trace, alignment status."""
        return _source(manager, patient_id).thermal_payload()

    @r.get("/patients/{patient_id}/thermal/registration")
    def thermal_registration(patient_id: str) -> dict:
        return _source(manager, patient_id).thermal_registration_status()

    @r.post("/patients/{patient_id}/thermal/registration")
    def set_thermal_registration(patient_id: str, body: AlignmentBody) -> dict:
        try:
            return _source(manager, patient_id).set_thermal_registration(
                [p.model_dump() for p in body.pairs])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @r.delete("/patients/{patient_id}/thermal/registration")
    def clear_thermal_registration(patient_id: str) -> dict:
        return _source(manager, patient_id).clear_thermal_registration()

    @r.post("/patients/{patient_id}/thermal/snapshot")
    def thermal_snapshot(patient_id: str) -> dict:
        try:
            return _source(manager, patient_id).thermal_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @r.get("/patients/{patient_id}/thermal/snapshot/{snapshot_id}/rgb.jpg")
    def thermal_snapshot_rgb(patient_id: str, snapshot_id: str) -> Response:
        data = _source(manager, patient_id).snapshot_rgb(snapshot_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Snapshot expired - take a new one.")
        return Response(content=data, media_type="image/jpeg", headers=_NO_CACHE)

    @r.get("/patients/{patient_id}/thermal/overlay.jpg")
    def thermal_overlay(patient_id: str) -> Response:
        src = _source(manager, patient_id)
        if src.thermal_reg is None:
            raise HTTPException(status_code=409, detail="The thermal camera is not aligned yet.")
        data = src.thermal_overlay_jpeg()
        if data is None:
            raise HTTPException(status_code=503, detail="Waiting for colour and thermal frames.")
        return Response(content=data, media_type="image/jpeg", headers=_NO_CACHE)

    # ---------------------------------------------------------------- video
    @r.get("/patients/{patient_id}/frame.jpg")
    def frame(patient_id: str) -> Response:
        data, _ = _source(manager, patient_id).latest_jpeg()
        if data is None:
            raise HTTPException(status_code=503, detail="No frame available yet.")
        return Response(content=data, media_type="image/jpeg", headers=_NO_CACHE)

    @r.get("/patients/{patient_id}/stream.mjpg")
    async def stream(patient_id: str) -> StreamingResponse:
        src = _source(manager, patient_id)

        async def gen():
            src.stream_clients += 1
            last = -1
            try:
                while True:
                    data, v = src.latest_jpeg()
                    if data is not None and v != last:
                        last = v
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    await asyncio.sleep(0.5 / STREAM_FPS)
            finally:
                src.stream_clients = max(0, src.stream_clients - 1)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers=_NO_CACHE)

    return r


_NOT_BUILT = """<!doctype html><html><head><meta charset="utf-8"><title>VisionICU</title>
<style>body{font:16px/1.5 system-ui;background:#0b0f14;color:#d6dde6;display:grid;place-items:center;
min-height:100vh;margin:0}main{max-width:560px;padding:32px}code{background:#18202a;padding:2px 6px;
border-radius:4px}a{color:#5fd4c4}</style></head><body><main><h1>VisionICU server is running</h1>
<p>The dashboard has not been built yet. Build it once:</p>
<p><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></p>
<p>then reload this page. For live UI development run <code>npm run dev</code> and open
<a href="http://localhost:5173">localhost:5173</a>. API docs: <a href="/docs">/docs</a>.</p>
</main></body></html>"""


def create_app(manager: EngineManager, frontend_dir: str | Path = "frontend/dist",
               start_engines: bool = True) -> FastAPI:
    dist = Path(frontend_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_engines:
            manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="VisionICU", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
                       allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(build_api_router(manager))
    app.include_router(build_ws_router(manager))

    @app.exception_handler(Exception)
    async def _unhandled(_request, exc: Exception):  # never leak a stack trace as HTML
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404, detail="Not found")
        index = dist / "index.html"
        if not index.exists():
            return HTMLResponse(_NOT_BUILT)
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers=_NO_CACHE)    # client-side routes

    return app
