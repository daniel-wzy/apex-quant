"""
quant/train_walkforward.py — Phase 3.1: walk-forward (out-of-sample) validation.

Financial data has time dependency, so random splits leak the future. This
trains on a rolling 6-month window and validates on the next 1 month, stepping
forward 1 month at a time (plan §3.1). It reports per-fold and aggregate AUC /
precision@threshold for the ENTRY model — the headline "does it generalize?"
check before trusting train.py's full-data artifacts.

Usage:
    ./venv/bin/python quant/train_walkforward.py
"""
from __future__ import annotations

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quant import qconfig, feature_store as fs  # noqa: E402
from quant.train import load_labeled, buy_signal_mask, FEATURES  # noqa: E402


def _fit_fold(Xtr, ytr, Xva):
    scaler = StandardScaler().fit(Xtr)
    pos = max(1, int(ytr.sum()))
    neg = max(1, int((ytr == 0).sum()))
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        scale_pos_weight=neg / pos, n_jobs=-1, random_state=42,
    )
    clf.fit(scaler.transform(Xtr), ytr, verbose=False)
    return clf.predict_proba(scaler.transform(Xva))[:, 1]


def walk_forward(data: pd.DataFrame, target: str, threshold: float) -> pd.DataFrame:
    buys = data[buy_signal_mask(data)].dropna(subset=[target]).copy()
    buys["time"] = pd.to_datetime(buys["time"])
    if buys.empty:
        raise SystemExit("No buy-signal bars with labels to validate.")

    start = buys["time"].min().normalize().replace(day=1)
    end = buys["time"].max()

    folds = []
    fold_start = start
    fold_n = 0
    while True:
        train_lo = fold_start
        train_hi = train_lo + relativedelta(months=qconfig.WF_TRAIN_MONTHS)
        valid_hi = train_hi + relativedelta(months=qconfig.WF_VALID_MONTHS)
        if train_hi >= end:
            break

        tr = buys[(buys["time"] >= train_lo) & (buys["time"] < train_hi)]
        va = buys[(buys["time"] >= train_hi) & (buys["time"] < valid_hi)]
        fold_start = fold_start + relativedelta(months=qconfig.WF_STEP_MONTHS)

        if len(tr) < 100 or len(va) < 10:
            continue
        if tr[target].nunique() < 2:
            continue

        Xtr = tr[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        Xva = va[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
        ytr = tr[target].astype(int).to_numpy()
        yva = va[target].astype(int).to_numpy()

        proba = _fit_fold(Xtr, ytr, Xva)
        pred = (proba >= threshold).astype(int)
        fold_n += 1
        row = {
            "fold": fold_n,
            "train": f"{train_lo:%Y-%m}→{train_hi:%Y-%m}",
            "valid": f"{train_hi:%Y-%m}",
            "n_train": len(tr), "n_valid": len(va),
            "valid_pos_rate": float(yva.mean()),
            "auc": float(roc_auc_score(yva, proba)) if len(np.unique(yva)) > 1 else np.nan,
            "precision@thr": float(precision_score(yva, pred, zero_division=0)),
            "recall@thr": float(recall_score(yva, pred, zero_division=0)),
            "n_signals@thr": int(pred.sum()),
        }
        folds.append(row)

    return pd.DataFrame(folds)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="+", default=None, dest="timeframes",
                    help="restrict to these timeframes, e.g. --tf daily 4h")
    args = ap.parse_args(argv)

    data = load_labeled()
    if args.timeframes:
        data = data[data["timeframe"].isin(args.timeframes)]
        print(f"Restricted to timeframes {args.timeframes}")
    print(f"Walk-forward on {len(data)} bars, target={qconfig.ENTRY_TARGET}, "
          f"entry threshold={qconfig.ENTRY_THRESHOLD}\n")
    folds = walk_forward(data, qconfig.ENTRY_TARGET, qconfig.ENTRY_THRESHOLD)
    if folds.empty:
        print("No valid folds (need more history per window).")
        return

    pd.set_option("display.width", 160)
    print(folds.to_string(index=False,
          formatters={"auc": "{:.3f}".format,
                      "precision@thr": "{:.3f}".format,
                      "recall@thr": "{:.3f}".format,
                      "valid_pos_rate": "{:.3f}".format}))

    print("\n─── Aggregate (out-of-sample) ───")
    print(f"  Folds:             {len(folds)}")
    print(f"  Mean AUC:          {folds['auc'].mean():.3f}")
    print(f"  Mean precision@{qconfig.ENTRY_THRESHOLD:.2f}: {folds['precision@thr'].mean():.3f}  "
          f"(baseline pos rate {folds['valid_pos_rate'].mean():.3f})")
    print(f"  Mean recall@{qconfig.ENTRY_THRESHOLD:.2f}:    {folds['recall@thr'].mean():.3f}")

    out = os.path.join(qconfig.MODELS_DIR, "walkforward_report.csv")
    folds.to_csv(out, index=False)
    print(f"\n  Fold table → {out}")


if __name__ == "__main__":
    main()
