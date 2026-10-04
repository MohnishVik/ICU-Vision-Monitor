import time
import os
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import cv2
from src.types import Measurement, FrameBundle, ROIBundle

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))

class RespiratoryNet1D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualBlock1D(128)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x).flatten(1)
        return self.regressor(x).squeeze(-1)

class RespiratoryRateEstimator:
    def __init__(self, model_path="models/finetuned/rr_tcn_finetuned.pth", fs=20.0):
        self.fs = fs
        self.buf_len = int(fs * 15)
        self.flow_history = []
        self.depth_history = []
        self.abdomen_history = []
        self.thermal_history = []
        self.prev_chest_gray = None
        self.prev_points = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = RespiratoryNet1D(in_channels=1).to(self.device)

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[RR MODULE] Model initialization note: {e}")
        self.model.eval()

    def _extract_sparse_optical_flow(self, chest_rgb):
        if chest_rgb is None or chest_rgb.size == 0:
            return 0.0
        gray = cv2.cvtColor(chest_rgb, cv2.COLOR_BGR2GRAY) if len(chest_rgb.shape) == 3 else chest_rgb
        disp = 0.0

        if self.prev_chest_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_chest_gray, gray, self.prev_points, None,
                winSize=(15, 15), maxLevel=2
            )
            valid = status.flatten() == 1
            if np.sum(valid) >= 2:
                p0 = self.prev_points[valid]
                p1 = next_pts[valid]
                disp = float(np.mean(np.linalg.norm(p1 - p0, axis=1)))
                self.prev_points = p1.reshape(-1, 1, 2)
            else:
                self.prev_points = None
        else:
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=25, qualityLevel=0.01, minDistance=7)
            if corners is not None:
                h, w = gray.shape
                c_x, c_y = w / 2.0, h / 2.0
                diag = np.sqrt(c_x**2 + c_y**2) + 1e-6
                scored = [(pt, 1.0 - (np.sqrt((pt.ravel()[0] - c_x)**2 + (pt.ravel()[1] - c_y)**2) / diag)) for pt in corners]
                scored.sort(key=lambda item: item[1], reverse=True)
                self.prev_points = np.array([item[0] for item in scored[:5]], dtype=np.float32)

        self.prev_chest_gray = gray
        return disp

    def _spectral_peak(self, signal_arr, low=0.1, high=0.7):
        if len(signal_arr) < int(self.fs * 6):
            return None, 0.0
        sig = scipy.signal.detrend(np.array(signal_arr, dtype=np.float32), type='linear')
        freqs, psd = scipy.signal.welch(sig, fs=self.fs, nperseg=len(sig), nfft=8192)
        v_idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(v_idx) < 3:
            return None, 0.0
        peak_idx = np.argmax(psd[v_idx])
        peak_f = freqs[v_idx][peak_idx]
        snr = psd[v_idx][peak_idx] / (np.mean(psd[v_idx]) + 1e-6)
        sqi = float(np.clip(snr / 4.5, 0.0, 1.0))
        return peak_f * 60.0, sqi

    def process(self, roi_bundle: ROIBundle, frame_bundle: FrameBundle) -> Measurement:
        t0 = time.time()

        # 1. Path A: Optical Flow
        flow_sample = self._extract_sparse_optical_flow(roi_bundle.chest_roi)
        self.flow_history.append(flow_sample)
        if len(self.flow_history) > self.buf_len:
            self.flow_history.pop(0)

        # 2. Path B: Depth Disparity
        depth_val = 0.0
        depth_valid_ratio = 0.0
        if roi_bundle.chest_depth is not None and roi_bundle.chest_depth.size > 0:
            valid_pixels = roi_bundle.chest_depth[roi_bundle.chest_depth > 10]
            depth_valid_ratio = len(valid_pixels) / float(roi_bundle.chest_depth.size)
            if depth_valid_ratio >= 0.60:
                disparity = (50.0 * 380.0) / (valid_pixels.astype(np.float32) + 1e-6)
                depth_val = float(np.mean(disparity))

        self.depth_history.append(depth_val)
        if len(self.depth_history) > self.buf_len:
            self.depth_history.pop(0)

        # Abdomen tracking for TAA distress detection
        if roi_bundle.abdomen_depth is not None and roi_bundle.abdomen_depth.size > 0:
            v_ab = roi_bundle.abdomen_depth[roi_bundle.abdomen_depth > 10]
            if len(v_ab) > 20:
                self.abdomen_history.append(float(np.mean(v_ab)))
        if len(self.abdomen_history) > self.buf_len:
            self.abdomen_history.pop(0)

        # 3. Path C: Thermal Airflow (Inference only)
        if frame_bundle.thermal is not None:
            th = frame_bundle.thermal
            h_t, w_t = th.shape[:2]
            patch = th[int(h_t * 0.45):int(h_t * 0.65), int(w_t * 0.40):int(w_t * 0.60)]
            if patch.size > 0:
                self.thermal_history.append(float(np.mean(patch)))
        if len(self.thermal_history) > self.buf_len:
            self.thermal_history.pop(0)

        if len(self.depth_history) < int(self.fs * 6):
            return Measurement(
                value=None, sqi=0.0, ts=frame_bundle.ts_mono,
                source="rr.buffering", latency_ms=(time.time() - t0) * 1000.0,
                meta={"reason": "Buffering"}
            )

        rr_flow, sqi_flow = self._spectral_peak(self.flow_history)
        rr_depth, sqi_depth = self._spectral_peak(self.depth_history)
        if depth_valid_ratio < 0.60:
            sqi_depth = 0.0

        rr_thermal, sqi_thermal = (None, 0.0)
        if len(self.thermal_history) >= int(self.fs * 6):
            rr_thermal, sqi_thermal = self._spectral_peak(self.thermal_history)

        # 4. Path D: Fine-Tuned 1D-ResNet
        rr_deep = None
        if len(self.depth_history) >= 120:
            d_win = np.array(self.depth_history[-120:], dtype=np.float32)
            d_norm = (d_win - np.mean(d_win)) / (np.std(d_win) + 1e-6)
            inp_resamp = scipy.signal.resample(d_norm, 300)
            t_tensor = torch.from_numpy(inp_resamp).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                rr_deep = float(self.model(t_tensor).item())

        candidates = []
        if rr_flow is not None and sqi_flow > 0.2:
            candidates.append((rr_flow, max(0.05, 1.0 - sqi_flow), "flow"))
        if rr_depth is not None and sqi_depth > 0.2:
            candidates.append((rr_depth, max(0.05, 1.0 - sqi_depth), "depth"))
        if rr_thermal is not None and sqi_thermal > 0.3:
            candidates.append((rr_thermal, max(0.05, 1.0 - sqi_thermal), "thermal"))
        if rr_deep is not None and 4.0 <= rr_deep <= 40.0:
            candidates.append((rr_deep, 0.15, "fine_tuned_resnet"))

        if not candidates:
            return Measurement(
                value=None, sqi=0.0, ts=frame_bundle.ts_mono,
                source="rr.failed", latency_ms=(time.time() - t0) * 1000.0,
                meta={"active_paths": 0}
            )

        # Agreement Gate: If any two high-confidence paths differ by > 4 bpm, suppress output
        confident_vals = [c[0] for c in candidates if c[1] < 0.5]
        if len(confident_vals) >= 2 and (max(confident_vals) - min(confident_vals)) > 4.0:
            return Measurement(
                value=None, sqi=0.1, ts=frame_bundle.ts_mono,
                source="rr.agreement_gate", latency_ms=(time.time() - t0) * 1000.0,
                meta={"spread": max(confident_vals) - min(confident_vals)}
            )

        inv_vars = [1.0 / (c[1]**2) for c in candidates]
        total_inv = sum(inv_vars)
        weights = [iv / total_inv for iv in inv_vars]
        rr_fused = sum(w * c[0] for w, c in zip(weights, candidates))
        fused_sqi = float(np.clip(1.0 - np.sqrt(1.0 / total_inv), 0.0, 1.0))

        # Check Thoracoabdominal Asynchrony (TAA)
        taa_flag = False
        if len(self.depth_history) >= 60 and len(self.abdomen_history) >= 60:
            h_chest = scipy.signal.hilbert(self.depth_history[-60:])
            h_ab = scipy.signal.hilbert(self.abdomen_history[-60:])
            diff = np.abs(np.angle(h_chest) - np.angle(h_ab))
            if np.mean(diff) > np.pi / 2.0:
                taa_flag = True

        return Measurement(
            value=float(np.clip(rr_fused, 0.0, 45.0)),
            sqi=fused_sqi,
            ts=frame_bundle.ts_mono,
            source="rr.fused",
            latency_ms=(time.time() - t0) * 1000.0,
            meta={"active_paths": [c[2] for c in candidates], "taa_flag": taa_flag}
        )

_rr_estimator = None

def estimate_rr(roi_bundle: ROIBundle, frame_bundle: FrameBundle) -> Measurement:
    global _rr_estimator
    if _rr_estimator is None:
        _rr_estimator = RespiratoryRateEstimator()
    return _rr_estimator.process(roi_bundle, frame_bundle)
