"""
inv_variance.py — inverse-variance weighted fusion of Measurement objects
w_i = (1/sigma_i^2) / sum(1/sigma_j^2)
SQI gate: sqi < threshold -> suppress (never impute as normal)
value=None propagates cleanly — never treated as a normal reading.
Staleness watchdog: no output > 5s -> mark stale in UI.

Each estimator (e.g. rr.depth and rr.thermal) reports a value plus a signal
quality index. A path's uncertainty sigma is taken from meta["sigma_bpm"] when
the module provides one, otherwise derived from its SQI: a perfect signal
(sqi=1) gets sigma_min, a barely-passing one gets close to sigma_max.

Agreement gate: if two surviving paths disagree by more than
`agreement_gate_bpm` (signal.yaml rr/hr.agreement_gate_bpm), we do not average
them into a plausible-looking middle value — the fused output is None with
meta.reason = "paths_disagree".

Secondary paths (the learned networks, when fused) are advisory: when one is
further than the gate from the signal-processing paths it is dropped for that
update (meta.rejected = "disagrees_with_primary") instead of blanking the vital.
With no primary path available, secondary paths fuse on their own.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from src.types import Measurement


@dataclass
class FusionConfig:
    sqi_min: float = 0.30
    sigma_min: float = 0.5             # bpm, for sqi = 1.0
    sigma_max: float = 6.0             # bpm, for sqi = sqi_min
    agreement_gate_bpm: Optional[float] = None
    secondary: frozenset = frozenset()  # sources dropped first when paths disagree


def _sigma(m: Measurement, cfg: FusionConfig) -> float:
    s = m.meta.get("sigma_bpm") if isinstance(m.meta, dict) else None
    if isinstance(s, (int, float)) and math.isfinite(s) and s > 0:
        return float(s)
    q = min(1.0, max(cfg.sqi_min, float(m.sqi)))
    frac = (1.0 - q) / max(1e-6, 1.0 - cfg.sqi_min)       # 0 at sqi=1, 1 at sqi=sqi_min
    return cfg.sigma_min + frac * (cfg.sigma_max - cfg.sigma_min)


def fuse(measurements: Iterable[Optional[Measurement]], source: str,
         cfg: Optional[FusionConfig] = None) -> Measurement:
    """Fuse same-quantity Measurements into one. Always returns a Measurement;
    value is None whenever no trustworthy estimate exists."""
    cfg = cfg or FusionConfig()
    t0 = time.perf_counter()
    candidates = [m for m in measurements if m is not None]
    used: List[Measurement] = []
    rejected: Dict[str, str] = {}
    for m in candidates:
        if m.value is None or not math.isfinite(float(m.value)):
            rejected[m.source] = "no_value"
        elif m.sqi < cfg.sqi_min:
            rejected[m.source] = f"sqi<{cfg.sqi_min:.2f}"
        else:
            used.append(m)

    ts = max((m.ts for m in candidates), default=time.time())
    base_meta = {"inputs": [m.source for m in used], "rejected": rejected}

    if not used:
        # pass the most actionable cause through ("Patient moving" beats "Signal too weak")
        causes = {m.meta.get("reason") for m in candidates if isinstance(m.meta, dict)}
        reason = next((r for r in ("motion", "low_light") if r in causes), "no_valid_input")
        return Measurement(None, 0.0, ts, source, (time.perf_counter() - t0) * 1e3,
                           {**base_meta, "reason": reason})

    gate = cfg.agreement_gate_bpm
    primary = sorted(float(m.value) for m in used if m.source not in cfg.secondary)
    if gate is not None and primary and len(primary) < len(used):
        ref = primary[len(primary) // 2]                   # median of the primary paths
        for m in [m for m in used if m.source in cfg.secondary and abs(float(m.value) - ref) > gate]:
            used.remove(m)
            rejected[m.source] = "disagrees_with_primary"
        base_meta["inputs"] = [m.source for m in used]

    vals = [float(m.value) for m in used]
    if cfg.agreement_gate_bpm is not None and len(vals) > 1 \
            and (max(vals) - min(vals)) > cfg.agreement_gate_bpm:
        return Measurement(None, 0.0, ts, source, (time.perf_counter() - t0) * 1e3,
                           {**base_meta, "reason": "paths_disagree",
                            "spread_bpm": max(vals) - min(vals)})

    inv = [1.0 / _sigma(m, cfg) ** 2 for m in used]
    total = sum(inv)
    weights = [w / total for w in inv]
    value = sum(w * v for w, v in zip(weights, vals))
    sigma = math.sqrt(1.0 / total)
    sqi = sum(w * m.sqi for w, m in zip(weights, used))
    return Measurement(value, float(min(1.0, sqi)), ts, source, (time.perf_counter() - t0) * 1e3,
                       {**base_meta, "weights": {m.source: w for m, w in zip(used, weights)},
                        "sigma_bpm": sigma})


class StalenessWatchdog:
    """Tracks when each source last produced a real value."""

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = float(timeout_s)
        self._last_valid: Dict[str, float] = {}

    def observe(self, m: Optional[Measurement], now: float) -> None:
        if m is not None and m.value is not None:
            self._last_valid[m.source] = now

    def age(self, source: str, now: float) -> Optional[float]:
        t = self._last_valid.get(source)
        return None if t is None else now - t

    def is_stale(self, source: str, now: float) -> bool:
        a = self.age(source, now)
        return a is None or a > self.timeout_s
