# models/public/

Weights trained on public datasets, plus the pretrained pose model.
These files are git-ignored: copy them in by hand and never commit them.

```
models/public/
├── yolov8n-pose.pt                # REQUIRED
├── fall_gru_pseudo.pt             # REQUIRED
├── fall_gru_pseudo_e1.pt          # optional
├── pain_mobilenetv3_synpain.pth   # optional (experimental)
├── rr_cnn1d_lbrdic.pth            # optional (experimental)
└── hr_efficientphys_ubfc.pth      # optional (experimental)
```

| File | Used by | Trained on | Required | If it is missing |
|------|---------|------------|----------|------------------|
| `yolov8n-pose.pt` | Pose: patient tracking, face blur, fall keypoints, face crop for pain and EfficientPhys | COCO (Ultralytics pretrained, never retrained) | **yes** | Pose, face blur, fall, pain and EfficientPhys switch off; RR and HR fall back to fixed regions |
| `fall_gru_pseudo.pt` | Fall detection, primary GRU | UR-Fall / UP-Fall (pseudo-labelled) | **yes** | The fallback below is tried; with neither, fall detection switches off |
| `fall_gru_pseudo_e1.pt` | Fall detection, fallback GRU | same | no | Nothing changes while the primary exists |
| `pain_mobilenetv3_synpain.pth` | Pain expression (experimental) | SynPAIN part 1 (`SynPain.ipynb`) | no | Pain panel shows "not loaded" |
| `rr_cnn1d_lbrdic.pth` | Breathing rate, learned 1-D CNN (experimental) | LBRD-IC (`baseline_rr.ipynb`) | no | Only this estimator switches off |
| `hr_efficientphys_ubfc.pth` | Heart rate, learned EfficientPhys (experimental) | UBFC-rPPG (`baseline_hr.ipynb`) | no | Only this estimator switches off |

The fine-tuned breathing model (`rr_cnn1d_d435_finetuned.pth`) goes in `models/finetuned/`; see the README there.

## Renaming training outputs

| Training output | Rename to |
|-----------------|-----------|
| `best_fall_gru.pt` | `fall_gru_pseudo.pt` |
| `best_model.pt` (fall) | `fall_gru_pseudo_e1.pt` |
| `pain_synpain_mobilenetv3.pth` or `pain_deepmlp_synpain.pth` | `pain_mobilenetv3_synpain.pth` |
| `best_resp_model.pth` | `rr_cnn1d_lbrdic.pth` |
| `EfficientPhys_Baseline.pth` | `hr_efficientphys_ubfc.pth` |

Paths can be changed in `config/signal.yaml` (`fall.*`, `pain.weights`, `learned.*.weights`).

## Checking

`python main.py` lists every expected file as `ok`, `MISSING` or `absent` at startup,
and the dashboard's **System** page shows the same table.

## Experimental models

The pain and learned RR/HR models are shown and logged next to the signal-processing values
but never drive the fused readings or alarms. To change that, first validate them on the
**Record & compare** page or with `scripts/compare_sessions.py`, then set `learned.fuse: true`
in `config/signal.yaml`.
