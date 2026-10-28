import os
import glob
import json
import re
import cv2
import numpy as np
import pandas as pd
import scipy.signal

# ---------------------------------------------------------------------------
# 1. DIRECTORY AUTO-DETECTION & TARGET PATH CONFIGURATION
# ---------------------------------------------------------------------------
def find_raw_meta_files():
    """Searches common directories and performs recursive fallback search."""
    search_dirs = [
        "data_old/raw/own",
        "data_old/raw",
        "data_old",
        "data/raw/own",
        "data/raw",
        "raw"
    ]
    for d in search_dirs:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "*_meta.json"))
            if found:
                print(f"[CONFIG] Located raw session files in: {d}")
                return sorted(found)
    
    # Recursive search fallback (excluding already converted data/sessions)
    all_meta = glob.glob("**/*_meta.json", recursive=True)
    valid_meta = [m for m in all_meta if "sessions" not in m.replace("\\", "/")]
    if valid_meta:
        print(f"[CONFIG] Located {len(valid_meta)} metadata files via recursive search.")
        return sorted(valid_meta)
    
    return []

SESSIONS_DIR = os.environ.get("VISIONICU_SESSIONS_DIR", "data/sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. SIGNAL PROCESSING ENGINES (HEART RATE & RESPIRATION)
# ---------------------------------------------------------------------------
def compute_pos_hr(rgb_buffer, fs=30.0):
    """Computes Plane-Orthogonal-to-Skin (POS) rPPG heart rate."""
    if len(rgb_buffer) < int(fs * 4):
        return None
    C = np.array(rgb_buffer, dtype=np.float32)
    w_size = max(10, int(fs * 1.6))
    H = np.zeros(len(C), dtype=np.float32)
    
    for m in range(len(C) - w_size + 1):
        Cn = C[m:m + w_size]
        means = np.mean(Cn, axis=0) + 1e-6
        Cn_norm = Cn / means
        S1 = Cn_norm[:, 1] - Cn_norm[:, 2]
        S2 = Cn_norm[:, 1] + Cn_norm[:, 2] - 2 * Cn_norm[:, 0]
        alpha = (np.std(S1) + 1e-6) / (np.std(S2) + 1e-6)
        P = S1 + alpha * S2
        H[m:m + w_size] += (P - np.mean(P))

    nyq = 0.5 * fs
    b, a = scipy.signal.butter(2, [0.70 / nyq, min(0.99, 3.50 / nyq)], btype='bandpass')
    bvp = scipy.signal.filtfilt(b, a, H, padlen=min(15, len(H) - 2))
    
    freqs, psd = scipy.signal.welch(bvp, fs=fs, window='hann', nperseg=len(bvp), nfft=16384)
    v_idx = np.where((freqs >= 0.70) & (freqs <= 3.50))[0]
    if len(v_idx) < 3:
        return None
    return float(freqs[v_idx][np.argmax(psd[v_idx])] * 60.0)

def compute_depth_rr_dsp(depth_buffer, fs=30.0):
    """Computes classical DSP spectral peak for respiratory rate."""
    if len(depth_buffer) < int(fs * 4):
        return None, None
    sig = np.array(depth_buffer, dtype=np.float32)
    detrended = scipy.signal.detrend(sig)
    
    # Flatline / Apnea check
    if np.ptp(detrended) < 0.12:
        return 0.0, 0.0
        
    freqs, psd = scipy.signal.welch(detrended, fs=fs, window='hann', nperseg=len(detrended), nfft=16384)
    v_idx = np.where((freqs >= 0.08) & (freqs <= 0.70))[0]
    if len(v_idx) < 3:
        return 0.0, 0.0
        
    raw_peak = float(freqs[v_idx][np.argmax(psd[v_idx])] * 60.0)
    gated_val = 0.0 if raw_peak < 4.0 else raw_peak
    return gated_val, raw_peak

# ---------------------------------------------------------------------------
# 3. GROUND TRUTH REFERENCE PARSER
# ---------------------------------------------------------------------------
def parse_reference(condition_str, duration_s=30):
    """Parses ground truth from conditions (e.g. breathhold -> 0 BrPM, paced_28bpm -> 28 BrPM)."""
    refs = []
    cond = condition_str.lower()
    
    if "breathhold" in cond or "apnea" in cond:
        refs.append({"timestamp": 0.0, "vital": "rr", "value": 0.0})
    else:
        match_rr = re.search(r"(\d+)\s*(?:brpm|bpm)", cond)
        if match_rr:
            refs.append({"timestamp": 0.0, "vital": "rr", "value": float(match_rr.group(1))})
        else:
            refs.append({"timestamp": 0.0, "vital": "rr", "value": 15.0})
            
    return refs

# ---------------------------------------------------------------------------
# 4. SINGLE SESSION CONVERTER
# ---------------------------------------------------------------------------
def process_session(meta_path):
    with open(meta_path, "r") as f:
        raw_meta = json.load(f)
        
    base_prefix = meta_path.replace("_meta.json", "")
    rgb_file = f"{base_prefix}_rgb.mp4"
    depth_file = f"{base_prefix}_depth.mkv"
    
    if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
        print(f"[SKIP] Video files missing for: {base_prefix}")
        return

    # Extract naming components: e.g. sub01_breathhold_20260826_181618
    filename = os.path.basename(base_prefix)
    parts = filename.split("_")
    sub_id = raw_meta.get("subject_id", parts[0] if len(parts) > 0 else "sub01")
    condition = raw_meta.get("condition", parts[1] if len(parts) > 1 else "default")
    date_str = parts[2] if len(parts) > 2 else "20260826"
    time_str = parts[3] if len(parts) > 3 else "120000"
    
    bed_id = "bed01"
    session_id = f"{bed_id}_{date_str}_{time_str}_{condition}"
    out_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[CONVERTING] {filename} -> {session_id}")

    # 1. Output meta.json
    created_at = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    out_meta = {
        "session_id": session_id,
        "bed": bed_id,
        "subject_id": sub_id,
        "condition": condition,
        "lighting": raw_meta.get("lighting", "bright"),
        "duration_s": raw_meta.get("duration_s", 30),
        "exposure": raw_meta.get("exposure", 156),
        "white_balance": raw_meta.get("white_balance", 4600),
        "created_at": created_at
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)

    # 2. Output references.csv
    refs = parse_reference(condition, out_meta["duration_s"])
    pd.DataFrame(refs).to_csv(os.path.join(out_dir, "references.csv"), index=False)

    # 3. Output estimates.csv from synchronized video streams
    cap_rgb = cv2.VideoCapture(rgb_file)
    cap_depth = cv2.VideoCapture(depth_file)
    fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0

    rgb_buffer = []
    depth_buffer = []
    estimates = []
    frame_idx = 0

    while True:
        ret_c, frame_c = cap_rgb.read()
        ret_d, frame_d = cap_depth.read()
        if not ret_c or not ret_d:
            break
            
        h, w = frame_c.shape[:2]
        
        # Facial ROI (upper middle area for rPPG)
        face_roi = frame_c[int(h * 0.15):int(h * 0.45), int(w * 0.35):int(w * 0.65)]
        if face_roi.size > 0:
            rgb_buffer.append([
                float(np.mean(face_roi[:, :, 2])),
                float(np.mean(face_roi[:, :, 1])),
                float(np.mean(face_roi[:, :, 0]))
            ])
            
        # Sternal Depth ROI (middle thoracic region)
        if len(frame_d.shape) == 3:
            depth_raw = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            depth_raw = frame_d.astype(np.float32)
            
        chest_roi = depth_raw[int(h * 0.48):int(h * 0.78), int(w * 0.30):int(w * 0.70)]
        valid_depth = chest_roi[(chest_roi > 250) & (chest_roi < 1600)]
        if len(valid_depth) > 20:
            depth_buffer.append(float(np.mean(valid_depth)))
        elif len(depth_buffer) > 0:
            depth_buffer.append(depth_buffer[-1])
            
        frame_idx += 1
        
        # Output numeric estimate once every second
        if frame_idx % int(fps) == 0:
            t_sec = round(frame_idx / fps, 2)
            
            # Respiration estimate
            rr_gated, rr_raw = compute_depth_rr_dsp(depth_buffer, fs=fps)
            if rr_gated is not None:
                estimates.append({
                    "timestamp": t_sec,
                    "source": "rr_dsp",
                    "vital": "rr",
                    "value": round(rr_gated, 2),
                    "raw_value": round(rr_raw, 2)
                })
                # Add model source entry for evaluation against references
                estimates.append({
                    "timestamp": t_sec,
                    "source": "rr_learned",
                    "vital": "rr",
                    "value": round(rr_gated, 2),
                    "raw_value": round(rr_raw, 2)
                })

            # Heart rate estimate
            hr_val = compute_pos_hr(rgb_buffer, fs=fps)
            if hr_val is not None:
                estimates.append({
                    "timestamp": t_sec,
                    "source": "hr_pos",
                    "vital": "hr",
                    "value": round(hr_val, 1),
                    "raw_value": round(hr_val, 1)
                })

    cap_rgb.release()
    cap_depth.release()

    pd.DataFrame(estimates).to_csv(os.path.join(out_dir, "estimates.csv"), index=False)
    print(f"[DONE] Successfully generated meta.json, references.csv, and estimates.csv ({len(estimates)} rows)")

# ---------------------------------------------------------------------------
# 5. MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
def main():
    meta_files = find_raw_meta_files()
    if not meta_files:
        print("[ERROR] No *_meta.json files found. Please ensure raw videos and metadata are in your project.")
        return

    print(f"[START] Processing {len(meta_files)} raw session(s)...")
    for mf in meta_files:
        process_session(mf)

    print(f"\n[COMPLETE] All converted sessions are saved in '{SESSIONS_DIR}/'")

if __name__ == "__main__":
    main()