"""
quant/qconfig.py — Paths, timeframes, and model thresholds for the quant layer.

All credentials are loaded from a .env file at the repo root.
No API keys or secrets are hardcoded here.
"""
import os
from pathlib import Path

# ─── Directory layout ────────────────────────────────────────────────
QUANT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(QUANT_DIR)
DATA_DIR     = os.path.join(QUANT_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
LABELS_DIR   = os.path.join(DATA_DIR, "labels")
MODELS_DIR   = os.path.join(QUANT_DIR, "models")

for _d in (RAW_DIR, FEATURES_DIR, LABELS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Timeframes the pipeline trains on ───────────────────────────────
TIMEFRAMES = ["30m", "1h", "4h", "daily"]

# SPY used for market-context features; fetched alongside the universe.
MARKET_SYMBOL = "US.SPY"

# How far back to request (data provider paginates; this is the target ceiling).
HISTORY_START = "2023-01-01 00:00:00"

# ─── Label parameters ─────────────────────────────────────────────────
TP_LEVELS = {           # label_name -> (take_profit_pct, stop_pct, horizon_bars)
    "tp5":  (5.0,  5.0,  5),
    "tp10": (10.0, 5.0, 10),
    "tp15": (15.0, 5.0, 20),
}
PNL_HORIZONS = [5, 10, 20]   # forward bars for regression labels
EXIT_HOLD_HORIZON = 5        # bars used to decide label_hold (exit model)

# ─── Model thresholds ─────────────────────────────────────────────────
# 0.75 chosen from the entry threshold sweep: dominates 0.60 on win rate,
# profit factor, and drawdown with roughly flat trade count.
ENTRY_THRESHOLD = 0.75
EXIT_THRESHOLD  = 0.65
ENTRY_TARGET    = "label_tp15"
EXIT_TARGET     = "label_hold"
SIZING_TARGET   = "label_pnl_20"

# Entry indicator-combo gate. An entry whose ONLY firing buy-indicator family
# is one of these is blocked — these were net losers / break-even in backtest.
BLOCK_ALONE_FAMILIES = {"IND_A", "IND_B"}  # illustrative

# Models are trained PER TIMEFRAME (the +15% barrier is unreachable intraday,
# so a pooled model just learns time_of_day as a timeframe proxy).
PER_TIMEFRAME_MODELS = True


def model_path(role: str, timeframe: str | None = None) -> str:
    """quant/models/<role>_model[_<tf>].pkl  (tf=None → pooled fallback)."""
    name = f"{role}_model" + (f"_{timeframe}" if timeframe else "") + ".pkl"
    return os.path.join(MODELS_DIR, name)


# ─── Walk-forward parameters ──────────────────────────────────────────
WF_TRAIN_MONTHS = 6
WF_VALID_MONTHS = 1
WF_STEP_MONTHS  = 1
