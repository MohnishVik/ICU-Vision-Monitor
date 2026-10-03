"""
test_rr_nfcp.py — NFCP on synthetic sine wave
Known input frequency must produce expected RR within ±0.5 bpm.
"""
# TODO: implement once rr.py NFCP function is written
# import numpy as np
# from src.modules.rr import nfcp_rate
# def test_nfcp_known_frequency():
#     fs = 25
#     rr_true = 15.0  # bpm
#     freq = rr_true / 60.0
#     t = np.linspace(0, 30, 30 * fs)
#     signal = np.sin(2 * np.pi * freq * t)
#     rr_est = nfcp_rate(signal, fs=fs)
#     assert abs(rr_est - rr_true) < 0.5
