import time
import os
import collections
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import cv2
from src.types import Measurement, FrameBundle, ROIBundle

class EfficientPhysModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.temporal = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = self.spatial(x_reshaped).view(B, T, 64)
        return self.temporal(feats).squeeze(-1)

class HeartRateEstimator:
    def __init__(self, model_path="models/finetuned/hr_efficientphys_finetuned.pth", fs=20.0):
        self.fs = fs
        self.window_len = int(fs * 7.5)
        self.raw_crops = collections.deque(maxlen=self.window_len)
        self.rgb_signals = collections.deque(maxlen=self.window_len)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = EfficientPhysModel().to(self.device)

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[HR MODULE] Model initialization note: {e}")
        self.model.eval()

    def _compute_pos(self, rgb_list):
        C = np.array(rgb_list, dtype=np.float32)
        N = len(C)
        w_size = max(10, int(self.fs * 1.6))
        H = np.zeros(N, dtype=np.float32)

        for m in range(N - w_size + 1):
            Cn = C[m:m + w_size]
            means = np.mean(Cn, axis=0) + 1e-6
            Cn_norm = Cn / means

            S1 = Cn_norm[:, 1] - Cn_norm[:, 2]
            S2 = Cn_norm[:, 1] + Cn_norm[:, 2] - 2 * Cn_norm[:, 0]

            alpha = (np.std(S1) + 1e-6) / (np.std(S2) + 1e-6)
            P = S1 + alpha * S2
            H[m:m + w_size] += (P - np.mean(P))

        return H

    def _extract_hr_welch(self, bvp_sig):
        if len(bvp_sig) <= 16:
            return None, 0.0

        nyq = 0.5 * self.fs
        safe_padlen = min(15, len(bvp_sig) - 2)
        b, a = scipy.signal.butter(2, [0.70 / nyq, min(0.99, 3.5 / nyq)], btype='bandpass')
        filtered = scipy.signal.filtfilt(b, a, bvp_sig, padlen=safe_padlen)

        freqs, psd = scipy.signal.welch(filtered, fs=self.fs, nperseg=len(filtered), nfft=16384)
        v_idx = np.where((freqs >= 0.70) & (freqs <= 3.5))[0]
        if len(v_idx) < 3:
            return None, 0.0

        sub_f = freqs[v_idx]
        sub_p = psd[v_idx]
        peak_idx = np.argmax(sub_p)
        peak_f = sub_f[peak_idx]
        snr = sub_p[peak_idx] / (np.mean(sub_p) + 1e-6)
        sqi = float(np.clip(snr / 4.0, 0.0, 1.0))
        return peak_f * 60.0, sqi

    def process(self, roi_bundle: ROIBundle, frame_bundle: FrameBundle) -> Measurement:
        t0 = time.time()

        face_rgb = roi_bundle.face_roi
        if face_rgb is None or face_rgb.size == 0:
            return Measurement(
                value=None, sqi=0.0, ts=frame_bundle.ts_mono,
                source="hr.no_face", latency_ms=(time.time() - t0) * 1000.0,
                meta={"reason": "Face ROI missing"}
            )

        resized_crop = cv2.resize(face_rgb, (72, 72))
        self.raw_crops.append(resized_crop)

        r_val = float(np.mean(face_rgb[:, :, 2]))
        g_val = float(np.mean(face_rgb[:, :, 1]))
        b_val = float(np.mean(face_rgb[:, :, 0]))
        self.rgb_signals.append([r_val, g_val, b_val])

        if len(self.rgb_signals) < int(self.fs * 3.0):
            return Measurement(
                value=None, sqi=0.0, ts=frame_bundle.ts_mono,
                source="hr.buffering", latency_ms=(time.time() - t0) * 1000.0,
                meta={"buffer_len": len(self.rgb_signals)}
            )

        pos_raw = self._compute_pos(self.rgb_signals)
        hr_pos, sqi_pos = self._extract_hr_welch(pos_raw)

        hr_deep, sqi_deep = None, 0.0
        if len(self.raw_crops) >= 30:
            crops_arr = np.array(self.raw_crops, dtype=np.float32)
            diff_frames = np.diff(crops_arr, axis=0) / 255.0

            t_diff = torch.from_numpy(diff_frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                deep_bvp = self.model(t_diff).squeeze(0).cpu().numpy()

            hr_deep, sqi_deep = self._extract_hr_welch(deep_bvp)

        candidates = []
        if hr_pos is not None and sqi_pos > 0.15:
            candidates.append((hr_pos, max(0.05, 1.0 - sqi_pos), "pos"))
        if hr_deep is not None and sqi_deep > 0.20:
            candidates.append((hr_deep, max(0.05, 1.0 - sqi_deep), "efficientphys"))

        if not candidates:
            return Measurement(
                value=None, sqi=0.0, ts=frame_bundle.ts_mono,
                source="hr.low_snr", latency_ms=(time.time() - t0) * 1000.0,
                meta={"reason": "Low SNR"}
            )

        inv_vars = [1.0 / (c[1]**2) for c in candidates]
        total_inv = sum(inv_vars)
        weights = [iv / total_inv for iv in inv_vars]
        hr_fused = sum(w * c[0] for w, c in zip(weights, candidates))
        fused_sqi = float(np.clip(1.0 - np.sqrt(1.0 / total_inv), 0.0, 1.0))

        return Measurement(
            value=float(np.clip(hr_fused, 40.0, 200.0)),
            sqi=fused_sqi,
            ts=frame_bundle.ts_mono,
            source="hr.fused",
            latency_ms=(time.time() - t0) * 1000.0,
            meta={"active_paths": [c[2] for c in candidates]}
        )

_hr_estimator = None

def estimate_hr(roi_bundle: ROIBundle, frame_bundle: FrameBundle) -> Measurement:
    global _hr_estimator
    if _hr_estimator is None:
        _hr_estimator = HeartRateEstimator()
    return _hr_estimator.process(roi_bundle, frame_bundle)
