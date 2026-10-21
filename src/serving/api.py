"""
api.py — FastAPI REST endpoints + static frontend serving
Serves compiled frontend/dist/ as static files (no separate web server needed).

Endpoints:
  GET  /health                  — module liveness + stale-value check
  GET  /patients                — active patient list + current states
  GET  /patients/{id}/vitals    — latest Measurement per module
  GET  /patients/{id}/history   — time-series for trend charts
  POST /patients/{id}/ack       — nurse acknowledges alarm
"""
# TODO: implement
