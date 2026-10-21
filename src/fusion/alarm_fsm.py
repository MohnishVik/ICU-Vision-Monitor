"""
alarm_fsm.py — EWS scoring + alarm state machine
NEWS2/RRT-18 scoring from camera-derived vitals.
4-stage escalation: log -> dashboard -> chime -> phone push.
Hysteresis + dwell to prevent rapid state flipping.
Explainability: "RR 16->26 over 40min. HR 88->112. Suggest assessment."
Target: <= 3 actionable alerts per nurse per shift.

Everything is driven by config/ews_rules.yaml:
  vitals.<rr|hr>.score_table  -> sub-scores ("<=8", "9-11", ">=25" ranges)
  ews_levels                  -> green / amber / red bands on the total
  alarm.dwell_s               -> a condition must persist this long to alarm
  alarm.escalation_stages     -> delay_s -> action
  alarm.staff_suppress        -> staff in the room caps escalation at log_only

Scoring is PARTIAL by design: the camera measures RR and HR only (no SpO2,
temperature, BP or consciousness), so this is a respiratory-cardiac early
warning sub-score, not full NEWS2. A missing vital is reported as missing —
it is never scored as if it were normal.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from src.fusion.trend import TrendStats

# --------------------------------------------------------------------------- #
# Defaults (identical to config/ews_rules.yaml, used if the file is absent)
# --------------------------------------------------------------------------- #
_DEFAULT_RULES: dict = {
    "vitals": {
        "rr": {"score_table": [
            {"range": "<=8", "score": 3}, {"range": "9-11", "score": 1},
            {"range": "12-20", "score": 0}, {"range": "21-24", "score": 2},
            {"range": ">=25", "score": 3}]},
        "hr": {"score_table": [
            {"range": "<=40", "score": 3}, {"range": "41-50", "score": 1},
            {"range": "51-90", "score": 0}, {"range": "91-110", "score": 1},
            {"range": "111-130", "score": 2}, {"range": ">=131", "score": 3}]},
    },
    "ews_levels": {
        "green": {"score_range": [0, 1], "action": "routine_monitoring"},
        "amber": {"score_range": [2, 4], "action": "increased_monitoring"},
        "red": {"score_range": [5, 999], "action": "urgent_clinical_review"},
    },
    "alarm": {
        "dwell_s": 1.5, "staff_suppress": True, "target_alerts_per_shift": 3,
        "escalation_stages": [
            {"delay_s": 0, "action": "log_only"}, {"delay_s": 10, "action": "dashboard_highlight"},
            {"delay_s": 30, "action": "chime"}, {"delay_s": 60, "action": "phone_push"}],
    },
}

ACTION_TEXT = {
    "routine_monitoring": "Continue routine monitoring.",
    "increased_monitoring": "Increase observation frequency.",
    "urgent_clinical_review": "Suggest urgent clinical assessment.",
}
LEVEL_RANK = {"unknown": -1, "green": 0, "amber": 1, "red": 2}


def load_rules(path: str | Path = "config/ews_rules.yaml") -> dict:
    p = Path(path)
    if yaml is not None and p.exists():
        try:
            raw = yaml.safe_load(p.read_text()) or {}
            merged = {**_DEFAULT_RULES, **raw}
            merged["alarm"] = {**_DEFAULT_RULES["alarm"], **(raw.get("alarm") or {})}
            return merged
        except Exception:  # noqa: BLE001 - fall back to defaults, never crash
            pass
    return dict(_DEFAULT_RULES)


# --------------------------------------------------------------------------- #
# Range parsing + EWS scoring
# --------------------------------------------------------------------------- #
def parse_range(text: str) -> Tuple[float, float]:
    """'<=8' -> (-inf, 8); '9-11' -> (9, 11); '>=25' -> (25, inf). Inclusive."""
    t = str(text).replace(" ", "")
    if t.startswith("<="):
        return (-math.inf, float(t[2:]))
    if t.startswith(">="):
        return (float(t[2:]), math.inf)
    if t.startswith("<"):
        return (-math.inf, float(t[1:]) - 1)
    if t.startswith(">"):
        return (float(t[1:]) + 1, math.inf)
    lo, hi = t.split("-", 1)
    return (float(lo), float(hi))


@dataclass
class EWSResult:
    total: Optional[int]
    level: str                               # green | amber | red | unknown
    subscores: Dict[str, Optional[int]]
    missing: List[str]
    action: str
    partial: bool

    def to_dict(self) -> dict:
        return {"total": self.total, "level": self.level, "subscores": self.subscores,
                "missing": self.missing, "action": self.action,
                "action_text": ACTION_TEXT.get(self.action, ""), "partial": self.partial}


class EWSScorer:
    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or dict(_DEFAULT_RULES)
        self._tables: Dict[str, List[Tuple[float, float, int]]] = {}
        for vital, spec in (self.rules.get("vitals") or {}).items():
            rows = []
            for row in spec.get("score_table", []):
                lo, hi = parse_range(row["range"])
                rows.append((lo, hi, int(row["score"])))
            self._tables[vital] = rows
        self._levels = sorted(
            ((name, int(spec["score_range"][0]), int(spec["score_range"][1]), spec.get("action", ""))
             for name, spec in (self.rules.get("ews_levels") or {}).items()),
            key=lambda r: r[1])

    def sub_score(self, vital: str, value: Optional[float]) -> Optional[int]:
        if value is None or not math.isfinite(value) or vital not in self._tables:
            return None
        v = round(value)                         # NEWS2 bands are whole numbers
        for lo, hi, s in self._tables[vital]:
            if lo <= v <= hi:
                return s
        return None

    def level_for(self, total: int) -> Tuple[str, str]:
        for name, lo, hi, action in self._levels:
            if lo <= total <= hi:
                return name, action
        return (self._levels[-1][0], self._levels[-1][3]) if self._levels else ("unknown", "")

    def score(self, values: Dict[str, Optional[float]]) -> EWSResult:
        subs = {k: self.sub_score(k, values.get(k)) for k in self._tables}
        present = {k: s for k, s in subs.items() if s is not None}
        missing = [k for k, s in subs.items() if s is None]
        if not present:
            return EWSResult(None, "unknown", subs, missing, "", True)
        total = sum(present.values())
        level, action = self.level_for(total)
        return EWSResult(total, level, subs, missing, action, bool(missing))


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
_CHANGE_THRESH = {"rr": 3.0, "hr": 10.0}
_NAMES = {"rr": "RR", "hr": "HR"}


def explain(trends: Dict[str, Optional[TrendStats]], ews: EWSResult,
            current: Optional[Dict[str, Optional[float]]] = None) -> str:
    """'RR 16->26 over 40 min. HR 88->112 over 40 min. Suggest urgent clinical assessment.'

    `current` (the live values shown on screen) is used as the end point when given,
    so the sentence never quotes a different number than the readout beside it."""
    parts: List[str] = []
    for vital in ("rr", "hr"):
        st = trends.get(vital)
        if st is None or st.mean is None or st.first is None or st.last is None:
            if vital in ews.missing:
                parts.append(f"{_NAMES[vital]} unavailable")
            continue
        live = (current or {}).get(vital)
        if live is not None and math.isfinite(live):
            st = TrendStats(st.window_s, st.n, st.mean, st.std, st.slope_per_min, st.first, live, st.span_s)
        mins = max(1, int(round(st.span_s / 60.0)))
        delta = st.last - st.first
        sub = ews.subscores.get(vital) or 0
        if abs(delta) >= _CHANGE_THRESH[vital]:          # a real trend: show where it came from
            parts.append(f"{_NAMES[vital]} {st.first:.0f}→{st.last:.0f} over {mins} min")
        elif sub >= 1:                                    # abnormal but flat: say so, no fake arrow
            parts.append(f"{_NAMES[vital]} {st.last:.0f}, steady (scores +{sub})")
        else:
            parts.append(f"{_NAMES[vital]} {st.last:.0f} stable")
    msg = ". ".join(parts)
    action = ACTION_TEXT.get(ews.action, "")
    if action and ews.level in ("amber", "red"):
        msg = f"{msg}. {action}" if msg else action
    return (msg + ".") if msg and not msg.endswith(".") else msg


# --------------------------------------------------------------------------- #
# Alarm state machine
# --------------------------------------------------------------------------- #
_ids = itertools.count(1)


@dataclass
class Alarm:
    id: int
    kind: str                       # ews | fall | apnea | taa
    severity: str                   # amber | red
    title: str
    message: str
    first_seen: float               # wall clock
    state: str = "pending"          # pending | active | acknowledged | cleared
    active_since: Optional[float] = None
    last_seen: float = 0.0
    stage: int = 0
    action: str = "log_only"
    suppressed: bool = False
    acked_at: Optional[float] = None
    cleared_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "severity": self.severity, "title": self.title,
            "message": self.message, "state": self.state, "first_seen": self.first_seen,
            "active_since": self.active_since, "last_seen": self.last_seen, "stage": self.stage,
            "action": self.action, "suppressed": self.suppressed, "acked_at": self.acked_at,
            "cleared_at": self.cleared_at,
        }


@dataclass
class Condition:
    severity: str
    title: str
    message: str


class AlarmManager:
    """One instance per patient. Feed it the current conditions every tick."""

    LATCHING = frozenset({"fall"})             # stay until a nurse acknowledges
    EVENT_DWELL_ZERO = frozenset({"fall", "apnea", "taa"})  # modules already sustain these

    def __init__(self, rules: Optional[dict] = None, clear_hold_s: float = 5.0,
                 staff_threshold: int = 2, history_len: int = 50):
        alarm = (rules or _DEFAULT_RULES).get("alarm") or _DEFAULT_RULES["alarm"]
        self.dwell_s = float(alarm.get("dwell_s", 1.5))
        self.staff_suppress = bool(alarm.get("staff_suppress", True))
        self.target_per_shift = int(alarm.get("target_alerts_per_shift", 3))
        self.stages = sorted(((float(s["delay_s"]), str(s["action"]))
                              for s in alarm.get("escalation_stages", [])), key=lambda s: s[0]) \
            or [(0.0, "log_only")]
        self.clear_hold_s = float(clear_hold_s)
        self.staff_threshold = int(staff_threshold)
        self.history_len = int(history_len)
        self._open: Dict[str, Alarm] = {}
        self.history: List[Alarm] = []
        self._actionable_ts: List[float] = []

    # -- public ------------------------------------------------------------- #
    def update(self, now: float, conditions: Dict[str, Optional[Condition]],
               staff_count: int = 0) -> List[Alarm]:
        suppress = self.staff_suppress and staff_count >= self.staff_threshold
        for kind, cond in conditions.items():
            alarm = self._open.get(kind)
            if cond is not None:
                if alarm is None:
                    alarm = Alarm(next(_ids), kind, cond.severity, cond.title, cond.message, now,
                                  last_seen=now)
                    self._open[kind] = alarm
                self._present(alarm, cond, now, suppress)
            elif alarm is not None:
                self._absent(kind, alarm, now)
        # kinds not mentioned this tick count as absent
        for kind in [k for k in self._open if k not in conditions]:
            self._absent(kind, self._open[kind], now)
        return self.open_alarms()

    def acknowledge(self, alarm_id: Optional[int] = None, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        n = 0
        for kind, a in list(self._open.items()):
            if a.state in ("active", "pending") and (alarm_id is None or a.id == alarm_id):
                a.state, a.acked_at = "acknowledged", now
                n += 1
                if kind in self.LATCHING and (now - a.last_seen) >= self.clear_hold_s:
                    self._close(kind, a, now)
        return n

    def open_alarms(self) -> List[Alarm]:
        return sorted((a for a in self._open.values() if a.state != "pending"),
                      key=lambda a: (-LEVEL_RANK.get(a.severity, 0), a.first_seen))

    def actionable_this_shift(self, now: float, shift_s: float = 12 * 3600) -> int:
        self._actionable_ts = [t for t in self._actionable_ts if now - t <= shift_s]
        return len(self._actionable_ts)

    # -- internals ---------------------------------------------------------- #
    def _present(self, a: Alarm, cond: Condition, now: float, suppress: bool) -> None:
        escalated = LEVEL_RANK.get(cond.severity, 0) > LEVEL_RANK.get(a.severity, 0)
        a.severity, a.title, a.message, a.last_seen = cond.severity, cond.title, cond.message, now
        if a.state == "acknowledged" and escalated:
            a.state, a.active_since, a.acked_at = "active", now, None   # worse than when acked
        if a.state == "pending":
            dwell = 0.0 if a.kind in self.EVENT_DWELL_ZERO else self.dwell_s
            if now - a.first_seen >= dwell:
                a.state, a.active_since = "active", now
        if a.state == "active":
            elapsed = now - (a.active_since or now)
            stage = 0
            for i, (delay, _action) in enumerate(self.stages):
                if elapsed >= delay:
                    stage = i
            prev_rank = a.stage
            a.stage = stage
            a.suppressed = suppress
            a.action = self.stages[0][1] if suppress else self.stages[stage][1]
            chime_idx = next((i for i, (_, act) in enumerate(self.stages) if act == "chime"), None)
            if (not suppress and chime_idx is not None and stage >= chime_idx > prev_rank):
                self._actionable_ts.append(now)

    def _absent(self, kind: str, a: Alarm, now: float) -> None:
        if a.state == "pending":
            del self._open[kind]                     # never persisted past dwell
            return
        if kind in self.LATCHING and a.state != "acknowledged":
            return                                   # a fall stays up until acknowledged
        if now - a.last_seen >= self.clear_hold_s:
            self._close(kind, a, now)

    def _close(self, kind: str, a: Alarm, now: float) -> None:
        a.state, a.cleared_at = "cleared", now
        self._open.pop(kind, None)
        self.history.insert(0, a)
        del self.history[self.history_len:]
