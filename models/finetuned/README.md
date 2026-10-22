# models/finetuned/

Weights fine-tuned on your own D435 recordings. Git-ignored: copy them in by hand.
Subject videos are not stored or committed; these weights are locked (do not retrain).

| File | Used by | Required |
|------|---------|----------|
| `rr_cnn1d_d435_finetuned.pth` | Breathing, learned CNN on chest depth (experimental) | no |

Rename from the training output: `best_realsense_rr_model.pth` → `rr_cnn1d_d435_finetuned.pth`.
