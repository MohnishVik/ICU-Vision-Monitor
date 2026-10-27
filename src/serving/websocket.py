"""
websocket.py — WS /ws/stream/{patient_id}
Pushes Measurement JSON on every inference tick.
Payload: {value, sqi, source, latency_ms, ts, meta}
Stale values are flagged — never silently shown as live normal readings.

Two channels:
  /ws/stream/{patient_id}  {"type": "tick", "data": <full patient snapshot>}  (~4 Hz max)
  /ws/ward                 {"type": "ward", "data": [<patient summary>, ...]} (1 Hz)

Snapshots are already strict-JSON-safe (NaN/inf -> null, numpy -> python), see
src/serving/state.py, so the browser's JSON.parse never chokes.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:  # pragma: no cover
    from src.serving.engine import EngineManager

TICK_PERIOD_S = 0.25
WARD_PERIOD_S = 1.0


def _vital_summary(v: Optional[dict]) -> Dict[str, Any]:
    v = v or {}
    return {"value": v.get("value"), "stale": bool(v.get("stale", True)),
            "sqi": v.get("sqi", 0.0), "subscore": v.get("subscore"), "reason": v.get("reason")}


def summarize(snap: dict) -> dict:
    """Compact per-patient card for the ward overview."""
    vitals = snap.get("vitals") or {}
    ews = snap.get("ews") or {}
    fall = snap.get("fall") or {}
    taa = snap.get("taa") or {}
    scene = snap.get("scene") or {}
    alarms = snap.get("alarms") or []
    rank = {"red": 2, "amber": 1}
    open_alarms = [a for a in alarms if a.get("state") in ("active", "acknowledged")]
    top = max(open_alarms, key=lambda a: rank.get(a.get("severity"), 0), default=None)
    return {
        "patient_id": snap.get("patient_id"), "label": snap.get("label"),
        "mode": snap.get("mode"), "simulated": bool(snap.get("simulated")),
        "status": snap.get("status"), "status_detail": snap.get("status_detail"),
        "ts": snap.get("ts"), "version": snap.get("version"),
        "rr": _vital_summary(vitals.get("rr")), "hr": _vital_summary(vitals.get("hr")),
        "apnea": bool((vitals.get("rr") or {}).get("apnea")),
        "ews": {"total": ews.get("total"), "level": ews.get("level", "unknown"),
                "explanation": ews.get("explanation", ""), "partial": ews.get("partial", True)},
        "fall": {"state_name": fall.get("state_name"), "patient_state": fall.get("patient_state")},
        "taa_flag": bool(taa.get("flag")),
        "scene": {"has_patient": scene.get("has_patient"), "staff_count": scene.get("staff_count"),
                  "thermal_online": scene.get("thermal_online"), "calibrated": scene.get("calibrated"),
                  "lighting": scene.get("lighting")},
        "alarm_count": len([a for a in open_alarms if a.get("state") == "active"]),
        "top_alarm": top,
        "pain": {k: (snap.get("pain") or {}).get(k) for k in ("available", "probability", "pain_detected")},
        "recording": bool((snap.get("session") or {}).get("active")),
    }


async def _until_closed(ws: WebSocket) -> None:
    """Return once the client goes away (anything it sends is ignored). Lets a sender
    that has nothing new to send still notice the disconnect instead of looping forever."""
    try:
        while (await ws.receive()).get("type") != "websocket.disconnect":
            pass
    except Exception:  # noqa: BLE001 - any receive failure means the socket is gone
        return


def _tick_key(snap: dict) -> tuple:
    """Changes when the snapshot does, including when a frozen snapshot is aged to stale."""
    vit = snap.get("vitals") or {}
    return (snap.get("version"),) + tuple(bool((vit.get(k) or {}).get("stale")) for k in ("rr", "hr"))


def build_ws_router(manager: "EngineManager") -> APIRouter:
    r = APIRouter()

    @r.websocket("/ws/stream/{patient_id}")
    async def stream(ws: WebSocket, patient_id: str) -> None:
        if patient_id not in manager.sources:
            await ws.close(code=4404)
            return
        await ws.accept()
        src = manager.get(patient_id)
        last = None
        closed = asyncio.ensure_future(_until_closed(ws))
        try:
            while not closed.done():
                snap = src.snapshot()
                key = _tick_key(snap)
                if key != last:
                    last = key
                    await ws.send_json({"type": "tick", "data": snap})
                await asyncio.sleep(TICK_PERIOD_S)
        except (WebSocketDisconnect, RuntimeError, ConnectionError, asyncio.CancelledError):
            return
        finally:
            closed.cancel()

    @r.websocket("/ws/ward")
    async def ward(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json({"type": "ward",
                                    "data": [summarize(s.snapshot()) for s in manager.sources.values()]})
                await asyncio.sleep(WARD_PERIOD_S)
        except (WebSocketDisconnect, RuntimeError, ConnectionError, asyncio.CancelledError):
            return

    return r
