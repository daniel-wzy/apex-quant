#!/usr/bin/env python3
"""
data/synthetic/generate.py — Generate synthetic OHLCV and trade data.

Produces two CSV files used by gen_sim_trades.py and the test suite:

  data/synthetic/ohlcv_sample.csv    (~500 bars, random walk with drift)
  data/synthetic/trades_sample.csv   (~50 synthetic signals with R-multiples)

The synthetic data has no relationship to real markets. It is intended only
to demonstrate the pipeline end-to-end and exercise the test suite.

Usage
-----
    python data/synthetic/generate.py
    python data/synthetic/generate.py --bars 1000 --trades 100 --seed 99
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_ohlcv(
    n_bars: int = 500,
    start_price: float = 100.0,
    drift: float = 0.0002,   # slight upward drift per bar
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV bars using a geometric random walk.

    Parameters
    ----------
    n_bars : int
        Number of bars to generate.
    start_price : float
        Starting close price.
    drift : float
        Per-bar log-return drift (e.g. 0.0002 ≈ 5% annualised on 252 bars).
    volatility : float
        Per-bar log-return standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: time, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)

    # Build intraday timestamps: 9:30 AM every weekday
    times = []
    current = datetime(2023, 1, 3, 9, 30)
    while len(times) < n_bars:
        if current.weekday() < 5:   # Mon–Fri only
            times.append(current)
        current += timedelta(hours=1)

    times = times[:n_bars]

    # Geometric random walk for closes
    log_returns = rng.normal(drift, volatility, n_bars)
    closes = np.full(n_bars, start_price)
    for i in range(1, n_bars):
        closes[i] = closes[i - 1] * np.exp(log_returns[i])

    # Synthetic OHLC from close
    bar_range = closes * rng.uniform(0.005, 0.025, n_bars)  # 0.5%–2.5% bar range
    opens  = closes * (1 + rng.uniform(-0.005, 0.005, n_bars))
    highs  = np.maximum(opens, closes) + bar_range * rng.uniform(0.3, 1.0, n_bars)
    lows   = np.minimum(opens, closes) - bar_range * rng.uniform(0.3, 1.0, n_bars)
    lows   = np.maximum(lows, closes * 0.5)   # floor at 50% of close

    # Synthetic volume (mean-reverting around 500k shares)
    volumes = np.abs(rng.normal(500_000, 150_000, n_bars)).astype(int)

    df = pd.DataFrame({
        "time":   times,
        "open":   np.round(opens,  2),
        "high":   np.round(highs,  2),
        "low":    np.round(lows,   2),
        "close":  np.round(closes, 2),
        "volume": volumes,
    })
    return df


def generate_trades(
    ohlcv: pd.DataFrame,
    n_signals: int = 50,
    win_rate: float = 0.55,
    avg_win_R: float = 2.0,
    avg_loss_R: float = -1.0,
    stop_pct: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic signal records (pre-simulation) with known win/loss profile.

    Each row represents a signal bar. The gen_sim_trades.py script will then
    simulate T+1 entry, apply stop/TP exits, and produce actual trades.

    Returns
    -------
    pd.DataFrame with columns:
        signal_time, symbol, entry_price, stop_price,
        calibrated_prob, expected_R
    """
    rng = np.random.default_rng(seed)

    # Sample signal bars (spaced to avoid clustering)
    min_spacing = max(1, len(ohlcv) // (n_signals * 2))
    available = ohlcv.index[ohlcv.index >= 30]   # need 30 bars of warm-up
    step = max(1, len(available) // n_signals)
    signal_rows = [available[min(i * step, len(available) - 1)] for i in range(n_signals)]

    rows = []
    for idx in signal_rows:
        bar = ohlcv.iloc[idx]
        entry_price = float(bar["close"])
        stop_price  = round(entry_price * (1 - stop_pct), 2)

        # Assign calibrated probability (winners get higher probs)
        is_win = rng.random() < win_rate
        if is_win:
            prob = rng.uniform(0.60, 0.90)
            exp_R = avg_win_R * rng.uniform(0.7, 1.5)
        else:
            prob = rng.uniform(0.50, 0.75)
            exp_R = avg_loss_R * rng.uniform(0.7, 1.3)

        rows.append({
            "signal_time":     bar["time"],
            "symbol":          "US.SYM",
            "entry_price":     round(entry_price, 2),
            "stop_price":      stop_price,
            "calibrated_prob": round(prob, 4),
            "expected_R":      round(exp_R, 4),
        })

    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic OHLCV and trade data")
    ap.add_argument("--bars",   type=int, default=500,  help="Number of OHLCV bars")
    ap.add_argument("--trades", type=int, default=50,   help="Number of synthetic signals")
    ap.add_argument("--seed",   type=int, default=42,   help="Random seed")
    args = ap.parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {args.bars} OHLCV bars (seed={args.seed})...")
    ohlcv = generate_ohlcv(n_bars=args.bars, seed=args.seed)
    ohlcv_path = os.path.join(out_dir, "ohlcv_sample.csv")
    ohlcv.to_csv(ohlcv_path, index=False)
    print(f"  ✅ {ohlcv_path}  ({len(ohlcv)} rows)")
    print(f"     price range: {ohlcv['close'].min():.2f} – {ohlcv['close'].max():.2f}")

    print(f"\nGenerating {args.trades} synthetic signals...")
    trades = generate_trades(ohlcv, n_signals=args.trades, seed=args.seed)
    trades_path = os.path.join(out_dir, "trades_sample.csv")
    trades.to_csv(trades_path, index=False)
    print(f"  ✅ {trades_path}  ({len(trades)} rows)")
    above_tau = (trades["calibrated_prob"] >= 0.50).sum()
    print(f"     signals with prob ≥ 0.50: {above_tau} / {len(trades)}")

    print("\nDone. Run the pipeline:")
    print("  python quant/gen_sim_trades.py")


if __name__ == "__main__":
    main()
