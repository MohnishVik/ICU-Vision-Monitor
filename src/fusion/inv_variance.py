"""
inv_variance.py — inverse-variance weighted fusion of Measurement objects
w_i = (1/sigma_i^2) / sum(1/sigma_j^2)
SQI gate: sqi < threshold -> suppress (never impute as normal)
value=None propagates cleanly — never treated as a normal reading.
Staleness watchdog: no output > 5s -> mark stale in UI.
"""
# TODO: implement
