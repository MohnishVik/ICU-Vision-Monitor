# models/public/

Weights trained on public datasets (plus the pretrained pose model). Git-ignored: copy them in by hand.

| File | Used by | Trained on | Required |
|------|---------|-----------|----------|
| `yolov8n-pose.pt` | Pose: patient tracking, face blur, fall keypoints | COCO (Ultralytics pretrained) | yes |
| `fall_gru_pseudo.pt` | Fall detection, primary GRU | UR-Fall / UP-Fall (pseudo-labelled) | yes |
| `fall_gru_pseudo_e1.pt` | Fall detection, fallback GRU | same | no |
| `pain_mobilenetv3_synpain.pth` | Pain expression (experimental) | SynPAIN part 1 (`SynPain.ipynb`) | no |
| `rr_cnn1d_lbrdic.pth` | Breathing, learned CNN (experimental) | LBRD-IC (`baseline_rr.ipynb`) | no |
| `hr_efficientphys_ubfc.pth` | Heart rate, learned (experimental) | UBFC-rPPG (`baseline_hr.ipynb`) | no |

Rename from the training outputs: `best_resp_model.pth` → `rr_cnn1d_lbrdic.pth`,
`EfficientPhys_Baseline.pth` → `hr_efficientphys_ubfc.pth`,
`pain_synpain_mobilenetv3.pth` / `pain_deepmlp_synpain.pth` → `pain_mobilenetv3_synpain.pth`.

Learned and pain models are shown and logged but never drive alarms until validated
(see the Compare page and `scripts/compare_sessions.py`).
