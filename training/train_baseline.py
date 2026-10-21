"""
train_baseline.py — Stage 1 training on public datasets (REFERENCE / LOCKED)
  RR:   1D-CNN/TCN on LBRD-IC,     lr=1e-3, 100 epochs, batch=64
  HR:   EfficientPhys on UBFC-rPPG, lr=9e-3, 30 epochs,  batch=32
  Fall: LightGBM on UR Fall+UP-Fall, 100 trees
  Pain: XGBoost on SynPAIN,          500 trees, lr=0.03

All weights from this stage are already trained and locked.
Run this only to reproduce baseline numbers for the paper.
"""
# TODO: implement
