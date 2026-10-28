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

import numpy as np


class MotionGate:
    def __init__(self, fs: float, abs_thresh: Optional[float] = None,
                 rel_thresh: Optional[float] = None, span_s: float = 0.5, settle_s: float = 1.5):
        # settle_s must exceed the 2 x span_s refill after a detection, or continuous
        # movement would look still for an instant between detections.
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


class MotionBridge:
    """
    Carry a vital-sign trace through a brief movement instead of discarding the window.

    Restarting the analysis window on every movement means a patient who shifts every
    20-30 s never gets a reading. A short movement is a step in level plus a second of
    junk, while the rhythm (breaths, beats) carries on, so instead:
      - while the patient moves, the last good value is held;
      - once still, new samples are shifted (mode "add") or scaled (mode "mul", for colour
        channels whose brightness changed) so the trace continues where it left off;
      - only movement lasting longer than max_hold_s asks the caller to restart.

    The gate confirms a step only once a few post-step samples are in, so output is
    delayed by the gate's half-second span: when movement is confirmed, the samples still
    waiting (which already contain the step) are replaced by the pre-step value. Half a
    second of delay is nothing for 20-30 s analysis windows.

    feed(value, ts) returns (items, restart): items is a list of (ts, value) to append, in
    order (usually one, none while warming up); restart means drop the window.
    """

    def __init__(self, gate: MotionGate, mode: str = "add", max_hold_s: float = 4.0,
                 trigger=None):
        self.gate, self.mode, self.max_hold_s = gate, mode, float(max_hold_s)
        self.trigger = trigger or (lambda v: v)     # scalar the gate watches (e.g. luma of RGB)
        self.restarts = 0
        self.reset()

    def reset(self) -> None:
        self._moving_since: Optional[float] = None
        self._held = None                           # last value emitted before the movement
        self._last_out = None
        self._shift = None                          # offset ("add") or ratio ("mul")
        self._q: Deque = deque()                    # (ts, value) not emitted yet

    def feed(self, value, ts: float):
        moving = self.gate.update(self.trigger(value), ts)
        if moving:
            if self._moving_since is None:          # movement confirmed: the queued samples
                self._moving_since = ts             # already hold the step, so hold over them
                self._held = self._last_out
                pending = [t for t, _ in self._q]
                self._q.clear()
            else:
                pending = []
            if ts - self._moving_since > self.max_hold_s:
                self.reset()
                self.restarts += 1
                return [], True                     # sustained movement: start again
            if self._held is None:
                return [], False
            return [(t, self._held) for t in pending + [ts]], False
        v = np.asarray(value, dtype=np.float64)
        if self._moving_since is not None:          # just became still: re-baseline so the
            self._moving_since = None               # first still sample equals the held one
            if self._held is not None:
                held = np.asarray(self._held, dtype=np.float64)
                if self.mode == "mul":
                    safe = np.where(np.abs(v) > 1e-9, v, 1.0)
                    self._shift = np.where(np.abs(v) > 1e-9, held / safe, 1.0)
                else:
                    self._shift = held - v
        if self._shift is not None:
            v = v * self._shift if self.mode == "mul" else v + self._shift
        self._q.append((ts, float(v) if v.ndim == 0 else v))
        out = []
        while len(self._q) > self.gate.n:
            t_out, v_out = self._q.popleft()
            self._last_out = v_out
            out.append((t_out, v_out))
        return out, False