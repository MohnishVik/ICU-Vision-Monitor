"""
trend.py — rolling trend statistics per vital sign
Windows: 5min, 15min, 60min, 240min
Outputs: slope (bpm/min), rolling_mean, variability (std)
Used by alarm_fsm for deterioration scoring.

Only real readings enter the buffer: a value of None ("cannot measure right
now") is skipped, never treated as a normal reading. A window with too few
samples reports n and leaves slope/mean/std as None rather than guessing.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

DEFAULT_WINDOWS_S: Tuple[int, ...] = (300, 900, 3600, 14400)   # 5, 15, 60, 240 min


@dataclass(frozen=True)
class TrendStats:
    window_s: int
    n: int
    mean: Optional[float]
    std: Optional[float]
    slope_per_min: Optional[float]      # bpm per minute (least-squares)
    first: Optional[float]              # mean over the oldest edge (5 % of span, <= 30 s)
    last: Optional[float]               # mean over the newest edge (5 % of span, <= 30 s)
    span_s: float                       # time actually covered by the samples

    def to_dict(self) -> dict:
        return {
            "window_s": self.window_s, "n": self.n, "mean": self.mean, "std": self.std,
            "slope_per_min": self.slope_per_min, "first": self.first, "last": self.last,
            "span_s": self.span_s,
        }


class TrendTracker:
    """Keeps (ts, value) samples for the longest window and summarises any window."""

    def __init__(self, windows_s: Sequence[int] = DEFAULT_WINDOWS_S, min_samples: int = 5):
        self.windows_s = tuple(sorted(int(w) for w in windows_s))
        self.min_samples = int(min_samples)
        self._buf: Deque[Tuple[float, float]] = deque()

    def add(self, ts: float, value: Optional[float]) -> None:
        if value is None or not math.isfinite(value):
            return
        self._buf.append((float(ts), float(value)))
        horizon = ts - self.windows_s[-1]
        while self._buf and self._buf[0][0] < horizon:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def window(self, now: float, window_s: int) -> TrendStats:
        pts = [(t, v) for (t, v) in self._buf if t >= now - window_s]
        n = len(pts)
        if n < self.min_samples:
            return TrendStats(window_s, n, None, None, None, None, None, 0.0)
        ts = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        mean = sum(vs) / n
        var = sum((v - mean) ** 2 for v in vs) / max(1, n - 1)
        t0 = ts[0]
        xs = [(t - t0) / 60.0 for t in ts]                 # minutes
        xm = sum(xs) / n
        sxx = sum((x - xm) ** 2 for x in xs)
        slope = (sum((x - xm) * (v - mean) for x, v in zip(xs, vs)) / sxx) if sxx > 1e-9 else 0.0
        # endpoints: average a short edge (5 % of the span, at most 30 s) so one noisy
        # window can't distort them, yet "last" stays within seconds of the live value
        edge = min(30.0, 0.05 * (ts[-1] - ts[0]))
        head = [v for t, v in pts if t <= ts[0] + edge]
        tail = [v for t, v in pts if t >= ts[-1] - edge]
        first = sum(head) / len(head)
        last = sum(tail) / len(tail)
        return TrendStats(window_s, n, mean, math.sqrt(var), slope, first, last, ts[-1] - ts[0])

    def stats(self, now: float) -> Dict[int, TrendStats]:
        return {w: self.window(now, w) for w in self.windows_s}
