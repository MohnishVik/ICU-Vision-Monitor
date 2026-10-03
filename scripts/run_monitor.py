"""
run_monitor.py — main Jetson deployment entry point
Boot sequence:
  1. Load configs (camera.yaml, models.yaml, ews_rules.yaml)
  2. Start D435 stream + thermal stream (optional)
  3. Start sync_buffer
  4. Load all model weights
  5. Start scene pipeline (privacy -> detect -> track -> pose -> geometry)
  6. Start module inference loop (rr, hr, fall, pain)
  7. Start fusion layer (inv_variance -> kalman -> trend -> alarm_fsm)
  8. Start FastAPI server (api.py + websocket.py)
"""
# TODO: implement
