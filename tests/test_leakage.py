"""
test_leakage.py — CI blocks build if any subject_id appears in both train and test
Run on every push via .github/workflows/ci.yml
A subject appearing in both splits inflates accuracy by 10-20 points.
"""
import pytest, json, os

SPLIT_FILES = [
    "data/splits/rr_splits.json",
    "data/splits/hr_splits.json",
    "data/splits/fall_splits.json",
    "data/splits/pain_splits.json",
]

def test_no_subject_leakage():
    for split_file in SPLIT_FILES:
        if not os.path.exists(split_file):
            continue
        with open(split_file) as f:
            splits = json.load(f)
        train = set(splits.get("train", []))
        val   = set(splits.get("val",   []))
        test  = set(splits.get("test",  []))
        assert train & test == set(), \
            f"{split_file}: subjects {train & test} in both train and test"
        assert val & test == set(), \
            f"{split_file}: subjects {val & test} in both val and test"
