"""
finetune.py — Stage 2 domain adaptation (LOCKED — do not re-run)
Loads public checkpoint, continues training at 10x lower LR on subject clips.
  RR: lr=1e-4, 30 epochs, 2-channel input [flow, depth]
  HR: lr=9e-4, 10 epochs

Subject videos are not being re-recorded. These weights are final.
Script kept for paper reproducibility only.
"""
# TODO: implement
