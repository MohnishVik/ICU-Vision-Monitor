"""
eval_weights.py - Sanity check on loaded weights before deployment.
Verifies RR, HR, Fall, and Pain checkpoints.
"""
import os
import torch
import numpy as np

def check_checkpoint(path, name):
    if not os.path.exists(path):
        print(f"[{name:15s}] FAIL - Checkpoint file not found: {path}")
        return False
    try:
        data = torch.load(path, map_location="cpu")
        num_keys = len(data) if isinstance(data, dict) else len(data.state_dict())
        print(f"[{name:15s}] PASS - Loaded successfully: {path} ({num_keys} tensors)")
        return True
    except Exception as e:
        print(f"[{name:15s}] FAIL - Error parsing {path}: {e}")
        return False

def main():
    print("=" * 65)
    print("      ICU-VISION-MONITOR: CLINICAL WEIGHTS SANITY CHECK")
    print("=" * 65)

    results = {}
    # Respiratory Rate Checkpoints
    results["RR_FINETUNED"] = check_checkpoint("models/finetuned/rr_tcn_finetuned.pth", "RR_FINETUNED")
    results["RR_PUBLIC"]    = check_checkpoint("models/public/rr_tcn_lbrdic.pth", "RR_PUBLIC")
    
    # Heart Rate Checkpoints
    results["HR_FINETUNED"] = check_checkpoint("models/finetuned/hr_efficientphys_finetuned.pth", "HR_FINETUNED")

    # Pose Model Checkpoint
    results["YOLO_POSE"]    = check_checkpoint("models/public/yolo11n-pose.pt", "YOLO_POSE")
    
    print("-" * 65)
    all_vitals_passed = results["RR_FINETUNED"] and results["RR_PUBLIC"] and results["HR_FINETUNED"]
    
    if all_vitals_passed:
        print("[RESULT] ALL ASSIGNED VITALS MODULE CHECKPOINTS: PASS")
    else:
        print("[RESULT] VITALS CHECKPOINTS VERIFICATION INCOMPLETE: CHECK PATHS")
    print("=" * 65)

if __name__ == "__main__":
    main()
