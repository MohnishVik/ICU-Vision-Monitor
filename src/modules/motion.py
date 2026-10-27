"""
motion.py — detect patient movement in a vital-sign trace so the window can restart.

A breath moves the chest a few millimetres over a couple of seconds and a heartbeat
changes skin colour by a fraction of a percent. Turning, talking or settling into the
chair moves the same traces by tens of millimetres or several percent in a fraction
of a second. One such step is larger than everything else in a 20-30 s window put
together, so the spectral peak (and the quality score) then describes the movement,
not the vital sign. Rather than let it poison the next 30 s, the module drops what it
has collected and starts again once the patient is still.

The test compares the mean of the last `span_s` with the mean of the `span_s` before
it: averaging over half a second keeps frame noise and the vital sign itself well
below the threshold, while real movement stands out.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class MotionGate:
    def __init__(self, fs: float, abs_thresh: Optional[float] = None,
                 rel_thresh: Optional[float] = None, span_s: float = 0.5, settle_s: float = 1.0):
        self.n = max(2, int(round(span_s * fs)))
        self.abs_thresh, self.rel_thresh = abs_thresh, rel_thresh
        self.settle_s = float(settle_s)
        self._v: Deque[float] = deque(maxlen=2 * self.n)
        self._quiet_at: Optional[float] = None
        self.last_motion_ts: Optional[float] = None

    def update(self, value: float, ts: float) -> bool:
        """Feed one sample; True while movement is happening or settling."""
        self._v.append(float(value))
        if len(self._v) == self._v.maxlen:
            vals = list(self._v)
            before = sum(vals[:self.n]) / self.n
            after = sum(vals[self.n:]) / self.n
            step = abs(after - before)
            moved = ((self.abs_thresh is not None and step > self.abs_thresh) or
                     (self.rel_thresh is not None and step > self.rel_thresh * max(abs(before), 1e-6)))
            if moved:
                self._quiet_at = ts + self.settle_s
                self.last_motion_ts = ts
                self._v.clear()                  # compare fresh samples once it stops
        return self._quiet_at is not None and ts < self._quiet_at

    def recent(self, ts: float, within_s: float) -> bool:
        """Whether movement happened in the last `within_s` seconds."""
        return self.last_motion_ts is not None and ts - self.last_motion_ts <= within_s
