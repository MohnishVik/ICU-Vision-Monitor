#!/usr/bin/env python3
"""
finetune.py — fine-tune a learned VisionICU model on your own training-capture sessions.

  python training/finetune.py --module rr_d435 --sessions data/sessions
  python training/finetune.py --module rr_lbrd --sessions data/sessions --epochs 40
  python training/finetune.py --module fall    --sessions data/sessions

What it does
  1. Builds windows from every session recorded with "Capture training data"
     (training/dataset.py), skipping simulated (demo) sessions unless --include-simulated.
  2. Splits PEOPLE, not windows, into train / val / test and writes the split to
     data/splits/<module>_splits.json (tests/test_leakage.py checks it).
  3. Starts from the weights the monitor uses now (config/signal.yaml) and trains at a low
     learning rate with augmentation, keeping the epoch with the best validation score.
  4. Scores the starting weights and the new ones on the held-out test people.
  5. Saves models/finetuned/<module>_<date>.pth (same format the monitor loads) and a
     <same name>.json report. Public weights are never overwritten.

To use the result, point the weights path in config/signal.yaml at the new file
(learned.rr_d435.weights, learned.rr_lbrd.weights, or fall.gru_weights_primary), restart,
and record validation sessions on people the model has never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.augmentations import augment_fall, augment_rr   # noqa: E402
from training.dataset import Windows, build, split_subjects    # noqa: E402

MODULES = ("rr_d435", "rr_lbrd", "fall")


def load_signal_cfg(path: str = "config/signal.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def current_weights(module: str, sig: dict) -> Path:
    from src.modules.fall import FallConfig
    from src.modules.learned import LearnedConfig
    if module == "fall":
        return FallConfig.from_dict(sig).gru_weights_primary
    lc = LearnedConfig.from_dict(sig)
    return (lc.rr_d435 if module == "rr_d435" else lc.rr_lbrd).weights


def load_model(module: str, path: Path):
    """(model, extra checkpoint fields to carry into the saved file)."""
    if module == "fall":
        from src.modules.fall import _load_gru
        model, _ = _load_gru(Path(path), Path(path), torch.device("cpu"))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return model, {"model_config": ck.get("model_config", {
            "input_size": model.input_size, "hidden_size": model.hidden_size})}
    from src.modules.learned import Resp1DCNN
    model = Resp1DCNN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    return model, {}


def predict(module: str, model: nn.Module, w: Windows, batch: int = 256) -> np.ndarray:
    """RR: rate in br/min. Fall: probability of a fall."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(w), batch):
            y = model(torch.from_numpy(w.X[i:i + batch]).float()).reshape(-1)
            out.append(torch.sigmoid(y).numpy() if module == "fall" else y.numpy())
    return np.concatenate(out) if out else np.zeros(0)


def score(module: str, pred: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Optional[float]]:
    if len(y) == 0:
        return {"n": 0}
    if module == "fall":
        p = pred >= threshold
        t = y > 0.5
        tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
        sens = tp / (tp + fn) if tp + fn else None
        spec = tn / (tn + fp) if tn + fp else None
        prec = tp / (tp + fp) if tp + fp else None
        f1 = 2 * prec * sens / (prec + sens) if prec and sens else None
        return {"n": len(y), "positives": int(t.sum()), "sensitivity": sens, "specificity": spec,
                "precision": prec, "f1": f1, "threshold": threshold}
    err = pred - y
    return {"n": len(y), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)), "within_2": float(np.mean(np.abs(err) <= 2.0))}


def _val_loss(module: str, s: dict) -> float:
    """Lower is better: MAE for RR, 1 - F1 (or 1 - specificity with no falls) for falls."""
    if s.get("n", 0) == 0:
        return float("inf")
    if module == "fall":
        return 1.0 - (s["f1"] if s.get("f1") is not None else (s.get("specificity") or 0.0) * 0.5)
    return s["mae"]


def train(module: str, model: nn.Module, tr: Windows, va: Windows, epochs: int, lr: float,
          batch: int, seed: int, patience: int, log=print) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if module == "fall":
        pos = float(tr.y.sum())
        pw = torch.tensor([(len(tr.y) - pos) / pos if pos else 1.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.SmoothL1Loss()
    aug = augment_fall if module == "fall" else augment_rr
    best = {"loss": float("inf"), "epoch": 0, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue                                       # BatchNorm needs > 1 sample
            xb = torch.from_numpy(np.stack([aug(tr.X[j], rng) for j in idx])).float()
            yb = torch.from_numpy(tr.y[idx]).float()
            opt.zero_grad()
            loss = loss_fn(model(xb).reshape(-1), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        vs = score(module, predict(module, model, va), va.y) if len(va) else {"n": 0}
        vl = _val_loss(module, vs) if len(va) else total / max(1, len(tr))
        history.append({"epoch": ep, "train_loss": total / max(1, len(tr)), "val": vs})
        log(f"epoch {ep:3d}  train loss {total / max(1, len(tr)):.4f}  val {json.dumps(vs)}")
        if vl < best["loss"] - 1e-6:
            best = {"loss": vl, "epoch": ep, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif ep - best["epoch"] >= patience:
            log(f"stopping: no validation improvement for {patience} epochs")
            break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "history": history}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, choices=MODULES)
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--init", help="starting weights (default: the file config/signal.yaml points at)")
    ap.add_argument("--out", help="output .pth (default models/finetuned/<module>_<date>.pth)")
    ap.add_argument("--signal-yaml", default="config/signal.yaml")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-subjects", type=int, default=3,
                    help="refuse to train with fewer people (results would not generalise)")
    ap.add_argument("--include-simulated", action="store_true", help="also use demo sessions (testing only)")
    a = ap.parse_args(argv)

    sig = load_signal_cfg(a.signal_yaml)
    data = build(a.module, a.sessions, signal_cfg=sig, include_simulated=a.include_simulated)
    subjects = sorted(set(data.subject))
    print(f"{a.module}: {len(data)} windows from {len(subjects)} subjects ({', '.join(subjects) or 'none'})")
    if len(data) == 0:
        print("No labelled windows. Record sessions with 'Capture training data' on and enter references "
              "(or mark fall events), then try again.", file=sys.stderr)
        return 1
    if len(subjects) < a.min_subjects:
        print(f"Only {len(subjects)} subject(s); need at least {a.min_subjects} so testing uses people "
              "the model never saw. Lower --min-subjects only for a smoke test.", file=sys.stderr)
        return 1

    split = split_subjects(subjects, a.val_frac, a.test_frac, a.seed)
    Path(a.splits_dir).mkdir(parents=True, exist_ok=True)
    split_path = Path(a.splits_dir) / f"{a.module}_splits.json"
    split_path.write_text(json.dumps(split, indent=2))
    tr, va, te = (data.select(split[k]) for k in ("train", "val", "test"))
    print(f"split by subject -> train {len(tr)} / val {len(va)} / test {len(te)} windows  ({split_path})")
    if len(tr) < 2:
        print("Not enough training windows.", file=sys.stderr)
        return 1

    init = Path(a.init) if a.init else current_weights(a.module, sig)
    model, extra = load_model(a.module, init)
    before = score(a.module, predict(a.module, model, te), te.y)
    print(f"test before ({init}): {json.dumps(before)}")
    t0 = time.time()
    hist = train(a.module, model, tr, va, a.epochs, a.lr, a.batch, a.seed, a.patience)
    after = score(a.module, predict(a.module, model, te), te.y)
    print(f"test after: {json.dumps(after)}")

    out = Path(a.out) if a.out else Path("models/finetuned") / f"{a.module}_{time.strftime('%Y%m%d_%H%M')}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.module == "fall":
        torch.save({"model_state_dict": model.state_dict(), **extra}, out)
    else:
        torch.save(model.state_dict(), out)
    report = {"module": a.module, "init": str(init), "out": str(out), "sessions": str(a.sessions),
              "split": split, "windows": {"train": len(tr), "val": len(va), "test": len(te)},
              "test_before": before, "test_after": after, "best_epoch": hist["best_epoch"],
              "epochs_run": len(hist["history"]), "seconds": round(time.time() - t0, 1),
              "settings": {k: getattr(a, k) for k in ("epochs", "lr", "batch", "patience", "seed")}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    key = {"rr_d435": "learned.rr_d435.weights", "rr_lbrd": "learned.rr_lbrd.weights",
           "fall": "fall.gru_weights_primary"}[a.module]
    print(f"saved {out}\nTo use it: set {key}: \"{out.as_posix()}\" in {a.signal_yaml} and restart.")
    if not len(te):
        print("WARNING: no test subjects, so these weights are unscored. Record more people.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
