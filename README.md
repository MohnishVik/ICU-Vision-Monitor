# VisionICU

Real-time, non-invasive, camera-based patient monitoring for ICU and IMCU environments.
Estimates Heart Rate, Respiratory Rate, Fall/Chair-Exit events, and Pain proxy
from video alone — no contact sensors attached to the patient.

## Hardware
- Intel RealSense D435 (RGB + Depth + IR)
- ESP32-S3 Thermal Camera (inference-only RR signal — not used in training)
- Jetson Orin Nano Super 8GB (edge deployment)

## Modules
| Module | Weights | Status |
|--------|---------|--------|
| Respiratory Rate | `rr_tcn_finetuned.pth` | Locked |
| Heart Rate | `hr_efficientphys_finetuned.pth` | Locked |
| Fall Detection | `fall_model_finetuned.pt` | Locked |
| Pain Assessment | `pain_deepmlp_synpain.pth` | SynPAIN only; UNBC/BioVid pending |

## Quick Start
```bash
pip install -r requirements.txt
# Place weight files in models/public/ and models/finetuned/ (see README files there)
python scripts/eval_weights.py   # sanity-check all weights
python main.py                   # start monitor + dashboard
```

## Development Sequence
1. `src/types.py` — shared contracts (done — scaffold in place)
2. `config/` — all YAML configs (done)
3. `src/capture/` — D435 + thermal streams
4. `src/scene/` — privacy + scene understanding
5. `src/modules/rr.py` — RR (start here — paste existing code)
6. `src/modules/hr.py` — HR
7. `src/modules/fall.py` — Fall
8. `src/modules/pain.py` — Pain
9. `src/fusion/` + `src/serving/` — EWS + dashboard

## Constraint Notes
- Auto-exposure is locked in `config/camera.yaml`. Never change exposure/WB values.
- Thermal was NOT used in RR training. TCN input is 2-channel [flow, depth] only.
- All subject video weights are locked. Do not retrain.
- Always report MAE + coverage together. Never bare MAE.
