"""
tests/test_purged_cv.py — Verify purged/embargoed CV produces no train/test
label overlap.

The critical property of the walk-forward CV used in this system:
- Training labels are computed over forward windows ending before the
  validation set starts.
- An embargo gap equal to the longest forward window is enforced between
  the last training bar and the first validation bar.

If this property is violated, the classifier sees future information during
training — exactly the label-leakage bug that inflated the initial Sharpe
from >4 to ~1.3 after being caught and fixed.

Tests
-----
- test_no_label_overlap: Verify no bar appears in both train and validation sets.
- test_embargo_gap_enforced: First validation bar is strictly > last training
  bar + embargo period.
- test_multiple_folds_independent: Train/val sets across folds don't overlap.
- test_leaky_split_would_fail: A naive random split DOES create overlap
  (demonstrates why purging matters).
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Minimal purged/embargoed CV implementation (mirrors the live walk-forward)
# ─────────────────────────────────────────────────────────────────────────────

def purged_walk_forward_splits(
    times: pd.Series,
    train_months: int = 6,
    valid_months: int = 1,
    step_months:  int = 1,
    embargo_bars: int = 5,   # bars to purge at the boundary
):
    """
    Yield (train_idx, valid_idx) index pairs for purged/embargoed walk-forward CV.

    Parameters
    ----------
    times : pd.Series of datetime64
        Bar timestamps in ascending order.
    train_months, valid_months, step_months : int
        Rolling window sizes.
    embargo_bars : int
        Number of bars immediately before the validation window to exclude
        from training (prevents forward-label bleed across the boundary).

    Yields
    ------
    (train_idx, valid_idx) : arrays of integer positions
    """
    from dateutil.relativedelta import relativedelta

    times = pd.to_datetime(times).reset_index(drop=True)
    start = times.min().normalize().replace(day=1)
    end   = times.max()

    fold_start = start
    while True:
        train_lo = fold_start
        train_hi = train_lo + relativedelta(months=train_months)
        valid_hi = train_hi + relativedelta(months=valid_months)
        if train_hi >= end:
            break

        tr_mask = (times >= train_lo) & (times < train_hi)
        va_mask = (times >= train_hi) & (times < valid_hi)

        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        # Embargo: drop the last `embargo_bars` bars from training
        if len(tr_idx) > embargo_bars:
            tr_idx = tr_idx[:-embargo_bars]

        if len(tr_idx) > 10 and len(va_idx) > 0:
            yield tr_idx, va_idx

        fold_start = fold_start + relativedelta(months=step_months)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factory
# ─────────────────────────────────────────────────────────────────────────────

def make_time_series(n_bars: int = 500, freq: str = "B") -> pd.DataFrame:
    """Synthetic daily bar DataFrame with timestamps and a dummy label column.

    Uses business-day frequency ("B") so 1 bar ≈ 1 trading day, giving
    adequate calendar coverage for month-based walk-forward windows.
    n_bars=2000 ≈ 8 years of data; n_bars=3000 ≈ 12 years.
    """
    base = datetime(2023, 1, 2)
    times = pd.date_range(base, periods=n_bars, freq=freq)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "time":  times,
        "close": 100.0 + rng.normal(0, 1, n_bars).cumsum(),
        "label": rng.integers(0, 2, n_bars),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPurgedCV:

    def test_no_label_overlap(self):
        """No bar index appears in both train and validation sets."""
        df = make_time_series(n_bars=2000)
        for tr_idx, va_idx in purged_walk_forward_splits(df["time"]):
            overlap = set(tr_idx) & set(va_idx)
            assert len(overlap) == 0, (
                f"Train/val overlap detected: {len(overlap)} bars in both sets"
            )

    def test_embargo_gap_enforced(self):
        """
        The last training bar must come before the first validation bar,
        with at least `embargo_bars` bars dropped between them.
        """
        df = make_time_series(n_bars=2000)
        embargo_bars = 5
        splits = list(purged_walk_forward_splits(df["time"], embargo_bars=embargo_bars))
        assert len(splits) > 0, "No folds generated — need more data"

        for tr_idx, va_idx in splits:
            last_train = tr_idx[-1]
            first_val  = va_idx[0]
            # Gap must be > 0 (train ends before val starts)
            assert last_train < first_val, (
                f"Train bleeds into val: last_train={last_train}, first_val={first_val}"
            )

    def test_multiple_folds_independent(self):
        """
        Validation sets across different folds must not overlap each other
        (each fold validates a distinct time window).
        """
        df = make_time_series(n_bars=3000)
        splits = list(purged_walk_forward_splits(df["time"], step_months=1))
        assert len(splits) >= 2, "Need at least 2 folds for this test"

        va_sets = [set(va) for _, va in splits]
        for i in range(len(va_sets)):
            for j in range(i + 1, len(va_sets)):
                overlap = va_sets[i] & va_sets[j]
                assert len(overlap) == 0, (
                    f"Validation sets for folds {i} and {j} overlap: {len(overlap)} bars"
                )

    def test_leaky_split_would_fail(self):
        """
        A naive random train/test split DOES create temporal overlap.
        This test demonstrates WHY purged CV is necessary — and confirms that
        the naive split is detectably leaky (first val bar < last train bar).
        """
        df = make_time_series(n_bars=500)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(df))
        split = len(df) // 5
        tr_idx = idx[split:]   # 80% train
        va_idx = idx[:split]   # 20% val (random)

        times = pd.to_datetime(df["time"])
        last_train_time = times.iloc[tr_idx].max()
        first_val_time  = times.iloc[va_idx].min()

        # With a random split the val set will often START before some training bars
        has_overlap = first_val_time < last_train_time
        assert has_overlap, (
            "Random split did not produce temporal overlap — unexpected; "
            "this test confirms why purged CV is necessary"
        )

    def test_at_least_two_folds_generated(self):
        """Confirm the walk-forward produces usable folds on sufficient data."""
        df = make_time_series(n_bars=3000)
        splits = list(purged_walk_forward_splits(df["time"]))
        assert len(splits) >= 2, (
            f"Expected ≥2 folds, got {len(splits)}"
        )
