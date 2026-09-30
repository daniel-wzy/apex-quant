"""
quant/feature_store.py — Shared feature vector builder.

This is the SINGLE source of truth for turning OHLCV bars into the model's
feature vector. It is imported by both the training pipeline and the live
decision path. Sharing this module is what guarantees train/live parity —
the #1 risk in an indicator-driven quant system.

Design
------
- compute_indicator_columns(df): run every indicator's compute() over the bar
  series and return a frame with all boolean signal columns coerced to 0/1.
- engineer_features(df, ...): add engineered context features.
- build_feature_frame(...): the full per-bar feature table (one row per bar).
- FEATURE_COLUMNS: ordered model input columns (excludes leak-prone fields
  like is_researched, raw price, and forward labels).

Lookahead discipline
--------------------
Every column here is computed from data available AT the bar's close.
Forward-looking labels live in a separate label module, never here.

Indicator stubs
---------------
This showcase ships the abstract interface (indicators/interface.py) and a
demonstrable moving-average crossover example. In the live system, each entry
in INDICATOR_MODULES is a proprietary module implementing the same interface.
The SIGNAL_COLUMNS registry and all feature-engineering logic below are real.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# Make the repo root importable whether run as a script or a module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import indicators.example_indicator as macross  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Confluence scoring weights
# These mirror the live system's SIGNAL_POINTS / CONTEXT_POINTS dictionaries.
# The exact numeric weights are not published; these are representative values
# illustrating the pattern (buy points > 0, sell points > 0, context bonuses).
# ─────────────────────────────────────────────────────────────────────────────
SIGNAL_POINTS: dict[str, int] = {
    # example_indicator signals — shown here so the scoring loop compiles
    "MACROSS_BUY":  2,
    "MACROSS_SELL": 2,
}

CONTEXT_POINTS: dict[str, int] = {
    # Trend-state bonuses applied in _per_bar_confluence
    "HMA_RED":    2,    # bullish HMA state adds to buy points
    "HMA_GREEN": -2,    # bearish HMA state adds to sell points
    "ABOVE_180":  1,    # price above 180-bar MA adds to buy points
    "BELOW_180": -1,    # price below 180-bar MA adds to sell points
}

# ─────────────────────────────────────────────────────────────────────────────
# Indicator registry
# ─────────────────────────────────────────────────────────────────────────────
# Each indicator module exposes compute(df) → df with boolean signal columns.
# In the live system this dict contains 8 proprietary indicator modules.
# Here we register the example indicator to keep the pipeline runnable.
INDICATOR_MODULES = {
    "MACROSS": macross,
}

# Canonical boolean signal columns per indicator (the model's raw inputs).
# Direction + owning indicator are tracked so confluence can be computed the
# same way the live system does, without re-running the heavy scoring per bar.
#
# In the live system this dict has ~40 entries across 8 indicator families.
# The structure (col -> (indicator_name, direction)) is identical.
SIGNAL_COLUMNS: dict[str, tuple[str, str]] = {
    # MACROSS (example — not used in live strategy)
    "MACROSS_BUY":  ("MACROSS", "BUY"),
    "MACROSS_SELL": ("MACROSS", "SELL"),
}

# Numeric oscillator/context columns worth feeding the model (state, not events).
# Live system feeds ~12 oscillator values from indicator internals.
NUMERIC_INDICATOR_COLUMNS: list[str] = [
    "hma_slope",
    "vol_ratio",
]

# Engineered context features.
ENGINEERED_COLUMNS: list[str] = [
    "confluence_score",
    "num_buy_indicators",
    "num_sell_indicators",
    "buy_tfs_count",
    "sell_tfs_count",
    "hma_state",          # 0=bearish, 1=neutral, 2=bullish
    "above_180",
    "vol_ratio",
    "price_vs_stop",
    "stock_change_pct",
    "spy_change_pct",
    "relative_strength",
    "time_of_day",
    "day_of_week",
]

# Buy-only and sell-only signal columns (used for entry candidate selection).
BUY_SIGNAL_COLUMNS  = [c for c, (_, d) in SIGNAL_COLUMNS.items() if d == "BUY"]
SELL_SIGNAL_COLUMNS = [c for c, (_, d) in SIGNAL_COLUMNS.items() if d == "SELL"]

# The model's ordered input columns. Deliberately EXCLUDES:
#   - is_researched (future knowledge — analyst bias)
#   - raw price/volume (non-stationary — only ratios/deltas kept)
#   - forward labels (computed downstream by the labeling module)
FEATURE_COLUMNS = list(SIGNAL_COLUMNS.keys()) + NUMERIC_INDICATOR_COLUMNS + ENGINEERED_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Indicator computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every indicator's compute() and merge the signal columns into one frame.

    Returns the original OHLCV columns plus every column listed in
    SIGNAL_COLUMNS (as 0/1 int) and NUMERIC_INDICATOR_COLUMNS (as float).
    Missing columns (an indicator that did not emit one) are filled with 0.
    """
    base = df.reset_index(drop=True).copy()
    merged = base.copy()

    for name, module in INDICATOR_MODULES.items():
        try:
            computed = module.compute(base)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"  ⚠️  {name}.compute failed: {exc}")
            continue
        for col in computed.columns:
            if col in base.columns:
                continue
            if col in SIGNAL_COLUMNS or col in NUMERIC_INDICATOR_COLUMNS:
                merged[col] = computed[col].values

    # Coerce signal columns to 0/1 ints; numeric to float; fill gaps.
    for col in SIGNAL_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(bool).astype(int)
        else:
            merged[col] = 0
    for col in NUMERIC_INDICATOR_COLUMNS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(float)
        else:
            merged[col] = 0.0

    return merged


def _per_bar_confluence(row: pd.Series) -> float:
    """
    Lightweight reproduction of the live system's net confluence score for a
    single bar. Uses SIGNAL_POINTS + CONTEXT_POINTS. Net = buy_pts - sell_pts.
    """
    buy_pts = 0
    sell_pts = 0
    for col, (_, direction) in SIGNAL_COLUMNS.items():
        if direction == "HOLD":
            continue
        if row.get(col, 0):
            pts = SIGNAL_POINTS.get(col, 1)
            if direction == "BUY":
                buy_pts += pts
            else:
                sell_pts += pts
    # Context bonuses.
    if row.get("HMA_RED_state", 0):
        buy_pts += CONTEXT_POINTS["HMA_RED"]
    elif row.get("HMA_GREEN_state", 0):
        sell_pts += abs(CONTEXT_POINTS["HMA_GREEN"])
    if row.get("above_180", 0):
        buy_pts += CONTEXT_POINTS["ABOVE_180"]
    else:
        sell_pts += abs(CONTEXT_POINTS["BELOW_180"])
    return float(buy_pts - sell_pts)


def _unique_indicator_count(row: pd.Series, columns: list[str]) -> int:
    """Count distinct indicator families firing among `columns` for this bar."""
    fams: set[str] = set()
    for col in columns:
        if row.get(col, 0):
            indicator, _ = SIGNAL_COLUMNS[col]
            fams.add(indicator)
    return len(fams)


def buy_families(row: pd.Series) -> set[str]:
    """Set of distinct buy-indicator families firing on a bar."""
    fams: set[str] = set()
    for col in BUY_SIGNAL_COLUMNS:
        if row.get(col, 0):
            indicator, _ = SIGNAL_COLUMNS[col]
            fams.add(indicator)
    return fams


def is_weak_entry_combo(row: pd.Series, block_alone: set[str] | None = None) -> bool:
    """
    True if the entry should be blocked because its ONLY firing buy family is a
    known weak/loser family (see qconfig.BLOCK_ALONE_FAMILIES). Confirmation
    from any second family clears the gate.
    """
    if block_alone is None:
        from quant.qconfig import BLOCK_ALONE_FAMILIES as block_alone
    fams = buy_families(row)
    return len(fams) == 1 and next(iter(fams)) in block_alone


# ─────────────────────────────────────────────────────────────────────────────
# Engineered features
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    researched: bool = False,
) -> pd.DataFrame:
    """
    Add engineered context features to a frame that already has indicator
    columns. `df` must contain OHLCV + helper columns (HMA_RED, HMA_GREEN,
    ABOVE_180, STOP_LONG) produced by the primary trend indicator.
    """
    out = df.copy()
    close = out["close"].astype(float)

    # HMA state encoding (0=bearish, 1=neutral, 2=bullish).
    hma_red   = out.get("HMA_RED",   pd.Series(False, index=out.index)).fillna(False).astype(bool)
    hma_green = out.get("HMA_GREEN", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["HMA_RED_state"]   = hma_red.astype(int)
    out["HMA_GREEN_state"] = hma_green.astype(int)
    out["hma_state"] = np.where(hma_red, 2, np.where(hma_green, 0, 1)).astype(int)

    # Above 180-bar MA.
    if "ABOVE_180" in out.columns:
        out["above_180"] = out["ABOVE_180"].fillna(False).astype(int)
    else:
        out["above_180"] = 0

    # Volume ratio vs 5-bar trailing average.
    vol = out["volume"].astype(float)
    vol5 = vol.rolling(5).mean()
    out["vol_ratio"] = (vol / vol5.replace(0, np.nan)).fillna(1.0)

    # Price distance to the smart stop (prior 5-bar low).
    if "STOP_LONG" in out.columns:
        stop = pd.to_numeric(out["STOP_LONG"], errors="coerce")
        out["price_vs_stop"] = ((close - stop) / close.replace(0, np.nan)).fillna(0.0)
    else:
        out["price_vs_stop"] = 0.0

    # Intraday % change (close vs previous close).
    out["stock_change_pct"] = (close.pct_change() * 100).fillna(0.0)

    # SPY market context (aligned by timestamp, merge-asof to avoid lookahead).
    spy_change = pd.Series(0.0, index=out.index)
    if spy_df is not None and not spy_df.empty:
        spy = spy_df[["time", "close"]].copy()
        spy["time"] = pd.to_datetime(spy["time"])
        spy = spy.sort_values("time")
        spy["spy_change_pct"] = spy["close"].pct_change() * 100
        left = out[["time"]].copy()
        left["time"] = pd.to_datetime(left["time"])
        left = left.reset_index().sort_values("time")
        merged = pd.merge_asof(
            left, spy[["time", "spy_change_pct"]], on="time", direction="backward"
        ).sort_values("index")
        spy_change = pd.Series(merged["spy_change_pct"].fillna(0.0).values, index=out.index)
    out["spy_change_pct"]   = spy_change.values
    out["relative_strength"] = out["stock_change_pct"] - out["spy_change_pct"]

    # Time-of-day (minutes since 09:30 open) and day-of-week.
    t = pd.to_datetime(out["time"])
    out["time_of_day"] = ((t.dt.hour * 60 + t.dt.minute) - (9 * 60 + 30)).clip(lower=0, upper=390)
    out["day_of_week"]  = t.dt.dayofweek

    # Per-bar confluence + unique indicator counts.
    out["confluence_score"]    = out.apply(_per_bar_confluence, axis=1)
    out["num_buy_indicators"]  = out.apply(
        lambda r: _unique_indicator_count(r, BUY_SIGNAL_COLUMNS), axis=1
    )
    out["num_sell_indicators"] = out.apply(
        lambda r: _unique_indicator_count(r, SELL_SIGNAL_COLUMNS), axis=1
    )

    # Cross-timeframe counts are filled in build_features (needs all TFs); default 0.
    if "buy_tfs_count" not in out.columns:
        out["buy_tfs_count"] = 0
    if "sell_tfs_count" not in out.columns:
        out["sell_tfs_count"] = 0

    # is_researched kept as a column for analysis but excluded from FEATURE_COLUMNS.
    out["is_researched"] = int(bool(researched))

    return out


def build_feature_frame(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    spy_df: pd.DataFrame | None = None,
    researched: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline for one symbol/timeframe: OHLCV → indicators → engineered.

    Returns a per-bar frame containing `time`, OHLCV, all FEATURE_COLUMNS,
    plus bookkeeping columns (symbol, timeframe, is_researched).
    """
    if df is None or len(df) < 30:
        return pd.DataFrame()

    enriched = compute_indicator_columns(df.reset_index(drop=True).copy())
    feat = engineer_features(enriched, spy_df=spy_df, researched=researched)
    feat["symbol"]    = symbol
    feat["timeframe"] = timeframe
    return feat


def build_live_vector(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    spy_df: pd.DataFrame | None = None,
    completed_bar: bool = True,
) -> pd.DataFrame | None:
    """
    Live path: build a single-row feature vector for the most recent COMPLETED
    bar (iloc[-2], matching the indicators' get_signals convention). Returns a
    1-row DataFrame with exactly FEATURE_COLUMNS, ready for model.predict().
    """
    feat = build_feature_frame(df, symbol, timeframe, spy_df=spy_df)
    if feat.empty or len(feat) < 2:
        return None
    idx = -2 if completed_bar else -1
    row = feat.iloc[[idx]][FEATURE_COLUMNS].copy()
    return row.fillna(0.0)
