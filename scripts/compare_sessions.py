"""
compare_sessions.py — score recorded sessions: every RR/HR source against your references.

    python scripts/compare_sessions.py                       # all sessions in data/sessions
    python scripts/compare_sessions.py data/sessions/bed-1_2026*  # pick sessions
    python scripts/compare_sessions.py --by condition --out reports/compare.md

For each source it prints MAE and coverage side by side (never MAE alone), plus bias,
RMSE, share within tolerance (+-2 BrPM / +-5 BPM) and Pearson r. Seconds in the warm-up
and right after a reference change are skipped (see src/evaluation/compare.py).
Simulated (demo) sessions are skipped unless --include-simulated.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.compare import aggregate, load_session, score_session  # noqa: E402

COLS = ("source", "n_scored", "coverage", "mae", "bias", "rmse", "within_tol", "pearson_r",
        "mae_unfiltered", "coverage_unfiltered")


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:.2f}"


def _table(title: str, rep: dict) -> str:
    lines = [f"### {title}", ""]
    for vital, unit in (("rr", "BrPM"), ("hr", "BPM")):
        stats = rep.get(vital) or {}
        if not any(s["n_scored"] for s in stats.values()):
            continue
        lines += [f"**{vital.upper()}** ({unit}; within = +-{stats[next(iter(stats))]['tolerance']:g})", "",
                  "| source | seconds | coverage | MAE | bias | RMSE | within | r | MAE (unfiltered) | cov. (unf.) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, s in sorted(stats.items(), key=lambda kv: (not kv[0].endswith(".fused"), kv[0])):
            lines.append(f"| {name} | {s['n_scored']} | {_fmt(s['coverage'], True)} | {_fmt(s['mae'])} | "
                         f"{_fmt(s['bias'])} | {_fmt(s['rmse'])} | {_fmt(s['within_tol'], True)} | "
                         f"{_fmt(s['pearson_r'])} | {_fmt(s['mae_unfiltered'])} | "
                         f"{_fmt(s['coverage_unfiltered'], True)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session folders (default: every folder in --root)")
    ap.add_argument("--root", default="data/sessions")
    ap.add_argument("--warmup", type=float, default=30.0, help="seconds skipped at session start")
    ap.add_argument("--settle", type=float, default=20.0, help="seconds skipped after a reference change")
    ap.add_argument("--by", choices=("session", "condition", "label"), default="condition",
                    help="how to group the pooled tables")
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("--out", help="also write the report as Markdown here (and a .csv next to it)")
    args = ap.parse_args()

    dirs = [Path(p) for p in args.sessions] or sorted(p for p in Path(args.root).glob("*") if p.is_dir())
    reports = []
    for d in dirs:
        try:
            sess = load_session(d)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {d}: {exc}", file=sys.stderr)
            continue
        if sess["meta"].get("simulated") and not args.include_simulated:
            continue
        reports.append(score_session(sess, warmup_s=args.warmup, settle_s=args.settle))
    if not reports:
        print("No sessions to score. Record one from the dashboard (bed page -> Record).")
        return 1

    groups = defaultdict(list)
    for r in reports:
        key = r["session"]["id"] if args.by == "session" else (r["session"].get(args.by) or "(none)")
        groups[key].append(r)

    parts = [f"# VisionICU source comparison\n\n{len(reports)} session(s); warm-up {args.warmup:g} s, "
             f"settle {args.settle:g} s after each reference change.\n",
             _table("All sessions pooled", aggregate(reports))]
    if len(groups) > 1 or args.by == "session":
        for key, reps in sorted(groups.items()):
            parts.append(_table(f"{args.by}: {key} ({len(reps)} session(s))", aggregate(reps)))
    text = "\n".join(parts)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        with open(out.with_suffix(".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "vital", *COLS])
            for key, reps in [("ALL", reports), *sorted(groups.items())]:
                agg = aggregate(reps)
                for vital in ("rr", "hr"):
                    for name, s in (agg.get(vital) or {}).items():
                        w.writerow([key, vital, name, *[s.get(c) for c in COLS[1:]]])
        print(f"\nWrote {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
