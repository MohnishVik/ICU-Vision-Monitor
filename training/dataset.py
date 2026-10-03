"""
dataset.py — HDF5 dataset loaders for all 4 modules
GroupKFold by subject_id enforced here — NEVER clip or frame split.
All weights are already locked. These loaders support:
  - paper reproducibility (re-running eval on locked test sets)
  - future Pain module training (UNBC/BioVid when access confirmed)
"""
# TODO: implement RRDataset, HRDataset, FallDataset, PainDataset
