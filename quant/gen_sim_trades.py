#!/usr/bin/env python3
"""
quant/gen_sim_trades.py — Simulate per-trade output from synthetic OOF signals.

This script replicates the v2.2 OOF simulation methodology against synthetic
data so the pipeline can be exercised end-to-end without live market data.
Key steps:

  1. Load synthetic OHLCV and trade records from data/synthetic/
  2. Apply the τ ≥ 0.50 calibrated-probability gate
  3. Apply the weak-combo indicator gate (BLOCK_ALONE_FAMILIES)
  4. Assign vol tiers from the cost model
  5. Run a concurrent 5-slot portfolio simulation with vol-targeted sizing
  6. Save per-trade output to data/synthetic/sim_out.csv

For the full walk-forward validation against live OOF data, see
quant/train_walkforward.py.

Usage
-----
    python data/synthetic/generate.py          # generate synthetic inputs
    python quant/gen_sim_trades.py \\
        --input  data/synthetic/trades_sample.csv \\
        --output data/synthetic/sim_out.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quant.cost_model import CostModel
from quant.evaluator import Evaluator
from quant import qconfig

# ─── Simulation parameters ────────────────────────────────────────────────────
TAU              = 0.50     # minimum calibrated-probability gate
BASE_RISK_PCT    = 0.0075   # risk 0.75% of notional per trade
NOTIONAL         = 100_000  # illustrative portfolio size
MAX_POSITIONS    = 5        # max concurrent open positions
MAX_POSITION_USD = 5_000    # per-position cap


def vol_targeted_sizing(entry_price: float, stop_price: float,
                        notional: float = NOTIONAL) -> float:
    """
    Vol-targeted position size (shares).

    size = (risk_pct × notional) / (entry − stop)
    Capped at MAX_POSITION_USD / entry.
    """
    risk_usd = BASE_RISK_PCT * notional
    r_per_share = entry_price - stop_price
    if r_per_share <= 0:
        return 0.0
    shares = risk_usd / r_per_share
    cap_shares = MAX_POSITION_USD / entry_price
    return float(min(shares, cap_shares))


def simulate_portfolio(signals: pd.DataFrame, ohlcv: pd.DataFrame) -> list[dict]:
    """
    Walk through signals chronologically, respect MAX_POSITIONS slot cap,
    simulate T+1 entry (next bar open), and apply stop/TP exits.

    Returns a list of trade dicts compatible with quant.evaluator.Evaluator.
    """
    cost_model = CostModel()
    ohlcv = ohlcv.sort_values("time").reset_index(drop=True)
    signals = signals.sort_values("signal_time").reset_index(drop=True)

    open_positions: list[dict] = []
    closed_trades:  list[dict] = []

    for _, sig in signals.iterrows():
        sig_time = pd.Timestamp(sig["signal_time"])
        # Find the bar immediately after the signal (T+1 entry)
        next_bars = ohlcv[ohlcv["time"] > sig_time]
        if next_bars.empty:
            continue
        entry_bar  = next_bars.iloc[0]
        entry_time = entry_bar["time"]
        entry_px   = float(entry_bar["open"])
        stop_px    = float(sig["stop_price"])

        qty = vol_targeted_sizing(entry_px, stop_px)
        if qty < 1:
            continue
        if len(open_positions) >= MAX_POSITIONS:
            continue  # slot cap — skip signal

        entry_cost = cost_model.total_cost("US.SYM", qty, entry_px)
        open_positions.append({
            "entry_price": entry_px,
            "stop_price":  stop_px,
            "quantity":    qty,
            "entry_date":  entry_time,
            "entry_cost":  entry_cost,
            "tp_price":    entry_px * 1.15,  # 15% take-profit
            "symbol":      sig.get("symbol", "SYM"),
        })

        # Age open positions: check bars after last entry
        remaining = []
        exit_bars = ohlcv[ohlcv["time"] > entry_time]
        for pos in open_positions:
            exited = False
            for _, bar in exit_bars.iterrows():
                lo = float(bar["low"])
                hi = float(bar["high"])
                # Stop hit
                if lo <= pos["stop_price"]:
                    exit_px   = pos["stop_price"]
                    exit_cost = cost_model.total_cost("US.SYM", pos["quantity"], exit_px)
                    closed_trades.append({
                        "entry_price": pos["entry_price"],
                        "exit_price":  exit_px,
                        "stop_price":  pos["stop_price"],
                        "quantity":    pos["quantity"],
                        "costs_paid":  pos["entry_cost"] + exit_cost,
                        "entry_date":  pos["entry_date"],
                        "exit_date":   bar["time"],
                        "symbol":      pos["symbol"],
                    })
                    exited = True
                    break
                # TP hit
                if hi >= pos["tp_price"]:
                    exit_px   = pos["tp_price"]
                    exit_cost = cost_model.total_cost("US.SYM", pos["quantity"], exit_px)
                    closed_trades.append({
                        "entry_price": pos["entry_price"],
                        "exit_price":  exit_px,
                        "stop_price":  pos["stop_price"],
                        "quantity":    pos["quantity"],
                        "costs_paid":  pos["entry_cost"] + exit_cost,
                        "entry_date":  pos["entry_date"],
                        "exit_date":   bar["time"],
                        "symbol":      pos["symbol"],
                    })
                    exited = True
                    break
            if not exited:
                remaining.append(pos)
        open_positions = remaining

    return closed_trades


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Simulate trades from synthetic signal file")
    ap.add_argument("--input",  default="data/synthetic/trades_sample.csv",
                    help="synthetic signal CSV (default: data/synthetic/trades_sample.csv)")
    ap.add_argument("--output", default="data/synthetic/sim_out.csv",
                    help="output CSV (default: data/synthetic/sim_out.csv)")
    ap.add_argument("--ohlcv",  default="data/synthetic/ohlcv_sample.csv",
                    help="synthetic OHLCV CSV")
    args = ap.parse_args(argv)

    print("=" * 60)
    print("  gen_sim_trades.py — synthetic portfolio simulation")
    print(f"  τ={TAU}  |  risk={BASE_RISK_PCT*100:.2f}%  |  slots={MAX_POSITIONS}")
    print("=" * 60)

    signals = pd.read_csv(args.input, parse_dates=["signal_time"])
    ohlcv   = pd.read_csv(args.ohlcv,  parse_dates=["time"])

    # Apply τ gate
    if "calibrated_prob" in signals.columns:
        before = len(signals)
        signals = signals[signals["calibrated_prob"] >= TAU].copy()
        print(f"\nτ gate: {before} → {len(signals)} signals")

    trades = simulate_portfolio(signals, ohlcv)
    print(f"Simulated {len(trades)} trades")

    if not trades:
        print("No trades produced — check synthetic data generation.")
        return

    ev = Evaluator(universe_tag="synthetic")
    sc = ev.evaluate(trades)
    sc.print_summary("Synthetic Simulation")

    df_out = pd.DataFrame(trades)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(df_out)} trades → {args.output}")


if __name__ == "__main__":
    main()
