"""test_rr_regions.py — rr.depth combines the chest and abdomen spectra."""
import types

import numpy as np
import yaml

from src.modules.rr import RRConfig, RRModule
from src.types import FrameBundle

FS = 30.0


def _rr(regions=("chest", "abdomen")):
    sig = yaml.safe_load(open("config/signal.yaml", encoding="utf-8"))
    sig["rr"]["depth"]["regions"] = list(regions)
    return RRModule(RRConfig.from_dict(sig), depth_fs=FS)


def _run(chest_at, abd_at, seconds=60, regions=("chest", "abdomen"), seed=0):
    """chest_at/abd_at(ts, rng) give each region's depth in mm (None = no crop)."""
    rr, rng = _rr(regions), np.random.default_rng(seed)
    c, a = np.zeros((40, 40), np.uint16), np.zeros((40, 40), np.uint16)
    rows = []
    for i in range(int(seconds * FS)):
        ts = i / FS
        zc, za = chest_at(ts, rng), abd_at(ts, rng)
        c[:] = int(round(zc))
        if za is not None:
            a[:] = int(round(za))
        roi = types.SimpleNamespace(chest_depth=c, abdomen_depth=a if za is not None else None,
                                    thermal_roi=None)
        m = rr.process(FrameBundle(rgb=None, depth=c, ir=None, thermal=None, ts_mono=ts), roi).get("rr.depth")
        if m is not None and m.meta.get("reason") != "motion":
            rows.append((ts, m))
    return rows


def _breath(rate_bpm, amp_mm):
    return lambda ts: amp_mm * np.sin(2 * np.pi * rate_bpm / 60 * ts)


def test_belly_breather_is_read_from_the_abdomen():
    # the chest barely moves (1 mm buried in 4 mm noise); the belly moves 6 mm at 12 /min
    b = _breath(12, 6)
    rows = _run(lambda ts, r: 1200 + 0.2 * b(ts) + r.normal(0, 4), lambda ts, r: 1250 + b(ts) + r.normal(0, 0.5))
    vals = [m.value for ts, m in rows if ts > 30 and m.value is not None]
    assert len(vals) >= 0.9 * len([1 for ts, _ in rows if ts > 30])
    assert abs(np.median(vals) - 12.0) < 1.0
    assert "abdomen" in rows[-1][1].meta["regions_used"]


def test_two_noisy_regions_beat_one():
    # the same 15 /min breath on both surfaces, each with its own heavy noise
    b = _breath(15, 3)
    err = {}
    for regions in (("chest",), ("chest", "abdomen")):
        e = []
        for seed in range(4):
            rows = _run(lambda ts, r: 1200 + b(ts) + r.normal(0, 6),
                        lambda ts, r: 1250 + b(ts) + r.normal(0, 6), regions=regions, seed=seed)
            e += [abs(m.value - 15.0) if m.value is not None else 5.0 for ts, m in rows if ts > 30]
        err[regions] = float(np.mean(e))
    assert err[("chest", "abdomen")] <= err[("chest",)]


def test_without_an_abdomen_crop_it_is_the_chest_estimate():
    b = _breath(15, 5)
    rows = _run(lambda ts, r: 1200 + b(ts), lambda ts, r: None)
    m = rows[-1][1]
    assert abs(m.value - 15.0) < 0.5 and m.meta["regions_used"] == ["chest"]
    assert set(m.meta["regions"]) == {"chest"}


def test_apnoea_needs_both_surfaces_still():
    b = _breath(15, 5)
    # chest goes still at 40 s but the belly keeps breathing: not an apnoea
    rows = _run(lambda ts, r: 1200 + (b(ts) if ts < 40 else 0) + r.normal(0, 0.2),
                lambda ts, r: 1250 + b(ts), seconds=70)
    assert not any(m.meta["apnea"] for ts, m in rows)
    # both go still: apnoea
    rows = _run(lambda ts, r: 1200 + (b(ts) if ts < 40 else 0) + r.normal(0, 0.2),
                lambda ts, r: 1250 + (b(ts) if ts < 40 else 0) + r.normal(0, 0.2), seconds=70)
    assert any(m.meta["apnea"] for ts, m in rows if ts > 50)
