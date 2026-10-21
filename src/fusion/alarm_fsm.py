"""
alarm_fsm.py — EWS scoring + alarm state machine
NEWS2/RRT-18 scoring from camera-derived vitals.
4-stage escalation: log -> dashboard -> chime -> phone push.
Hysteresis + dwell to prevent rapid state flipping.
Explainability: "RR 16->26 over 40min. HR 88->112. Suggest assessment."
Target: <= 3 actionable alerts per nurse per shift.
"""
# TODO: implement
