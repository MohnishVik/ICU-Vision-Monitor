"""
websocket.py — WS /stream/{patient_id}
Pushes Measurement JSON on every inference tick.
Payload: {value, sqi, source, latency_ms, ts, meta}
Stale values are flagged — never silently shown as live normal readings.
"""
# TODO: implement
