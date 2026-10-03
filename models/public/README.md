# models/public/

Place your public-dataset-trained weight files here:

| File                            | Module | Trained on      | Notes                        |
|---------------------------------|--------|-----------------|------------------------------|
| rr_tcn_lbrdic.pth               | RR     | LBRD-IC         | Reproduces MAE ≤3.84 bpm     |
| hr_efficientphys_ubfc.pth       | HR     | UBFC-rPPG       | MAE ≤3 bpm on PURE           |
| yolo11n-pose.pt                 | Fall   | COCO (Ultralytics pretrained) | Never retrained  |
| fall_model_public.pt            | Fall   | UR Fall+UP-Fall | Your existing .pt            |
| pain_xgboost_synpain.joblib     | Pain   | SynPAIN         | 75.4% acc, AUC 0.841         |
