"""
test_ews_scoring.py — NEWS2 logic with known inputs -> expected EWS score
Scores are read from config/ews_rules.yaml exactly as the running system reads them.
"""
import math

import pytest

from src.fusion.alarm_fsm import EWSScorer, load_rules, parse_range


@pytest.fixture(scope="module")
def scorer():
    return EWSScorer(load_rules("config/ews_rules.yaml"))


@pytest.mark.parametrize("text,lo,hi", [
    ("<=8", -math.inf, 8), ("9-11", 9, 11), (">=25", 25, math.inf), ("111-130", 111, 130),
])
def test_parse_range(text, lo, hi):
    assert parse_range(text) == (lo, hi)


@pytest.mark.parametrize("rr,expected", [
    (6, 3), (8, 3), (9, 1), (11, 1), (12, 0), (16, 0), (20, 0), (21, 2), (24, 2), (25, 3), (35, 3),
])
def test_rr_subscores(scorer, rr, expected):
    assert scorer.sub_score("rr", rr) == expected


@pytest.mark.parametrize("hr,expected", [
    (38, 3), (40, 3), (45, 1), (51, 0), (90, 0), (100, 1), (120, 2), (131, 3),
])
def test_hr_subscores(scorer, hr, expected):
    assert scorer.sub_score("hr", hr) == expected


def test_fractional_values_round_into_whole_number_bands(scorer):
    # 11.4 rounds to 11 (score 1); 11.6 rounds to 12 (score 0) - no gap between bands
    assert scorer.sub_score("rr", 11.4) == 1
    assert scorer.sub_score("rr", 11.6) == 0


def test_levels_follow_ews_rules(scorer):
    assert scorer.score({"rr": 16, "hr": 72}).level == "green"
    assert scorer.score({"rr": 22, "hr": 72}).level == "amber"     # 2
    assert scorer.score({"rr": 26, "hr": 122}).level == "red"      # 3 + 2


def test_missing_vital_is_reported_never_scored_as_normal(scorer):
    r = scorer.score({"rr": None, "hr": 72})
    assert r.missing == ["rr"] and r.partial and r.total == 0
    none = scorer.score({"rr": None, "hr": None})
    assert none.total is None and none.level == "unknown"
