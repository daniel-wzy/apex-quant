"""
quant/evaluator.py — Trade-level PnL Evaluator (Phase 0 shared infrastructure).

Everything in Phase 0 depends on this module. Unit test must pass before any
other step runs.

R definition: R = entry_price - stop_price  (per share, always positive for longs)
Trade result in R = net_realized_pnl_per_share / R

Trade dict fields required by evaluate():
    entry_price, exit_price, stop_price, quantity, costs_paid,
    entry_date, exit_date, symbol
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scorecard:
    run_id: str
    git_commit: str
    config_hash: str
    data_snapshot_id: str
    random_seed: int
    n_configs_tried: int

    n_trades: int
    trades_per_month: float
    win_rate: float           # diagnostic only, never a gate
    expectancy_R: float       # PRIMARY
    expectancy_R_ci95_low: float  # bootstrap lower bound, PRIMARY
    expectancy_pct: float
    profit_factor: float      # PRIMARY
    payoff_ratio: float
    avg_win_R: float
    avg_loss_R: float
    gross_profit_R: float
    gross_loss_R: float

    equity_max_drawdown_pct: float
    sharpe: float
    deflated_sharpe: float
    p_sharpe_gt_0: float

    trade_cvar_5_R: float     # mean of worst 5% trades
    worst_trade_R: float

    total_costs_paid_pct: float   # proves costs aren't hidden
    by_tier: dict             # empty until vol-tiering phase

    universe_tag: str         # e.g. "v1_partial" or "v1_full"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Scorecard":
        return cls(**d)

    def print_summary(self, title: str = "Scorecard"):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  Run ID:              {self.run_id}")
        print(f"  Universe:            {self.universe_tag}")
        print(f"  N Trades:            {self.n_trades}")
        print(f"  Trades/Month:        {self.trades_per_month:.2f}")
        print(f"  Win Rate:            {self.win_rate*100:.1f}%  (diagnostic)")
        print(f"  Expectancy R:        {self.expectancy_R:.4f}R")
        print(f"  Expectancy CI95 low: {self.expectancy_R_ci95_low:.4f}R  [PRIMARY GATE]")
        print(f"  Expectancy Pct:      {self.expectancy_pct:.2f}%")
        print(f"  Profit Factor:       {self.profit_factor:.3f}  [PRIMARY GATE]")
        print(f"  Payoff Ratio:        {self.payoff_ratio:.3f}")
        print(f"  Avg Win R:           {self.avg_win_R:.4f}R")
        print(f"  Avg Loss R:          {self.avg_loss_R:.4f}R")
        print(f"  Gross Profit R:      {self.gross_profit_R:.4f}R")
        print(f"  Gross Loss R:        {self.gross_loss_R:.4f}R")
        print(f"  Max Drawdown:        {self.equity_max_drawdown_pct:.2f}%")
        print(f"  Sharpe:              {self.sharpe:.3f}")
        print(f"  Deflated Sharpe:     {self.deflated_sharpe:.3f}")
        print(f"  P(Sharpe>0):         {self.p_sharpe_gt_0:.3f}")
        print(f"  CVaR 5%:             {self.trade_cvar_5_R:.4f}R")
        print(f"  Worst Trade:         {self.worst_trade_R:.4f}R")
        print(f"  Total Costs Paid:    {self.total_costs_paid_pct:.4f}%")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _get_config_hash() -> str:
    """Hash the current qconfig to detect config changes."""
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "qconfig.py")
        with open(cfg_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _load_configs_tried() -> int:
    path = os.path.join(os.path.dirname(__file__), "data", "configs_tried.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("n_configs_tried", 1)
    return 1


def _load_snapshot_id() -> str:
    path = os.path.join(os.path.dirname(__file__), "data", "snapshot_id.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return "unknown"


def _bootstrap_expectancy_ci(r_values: np.ndarray, n_boot: int = 10_000, seed: int = 42,
                               ci_low: float = 0.05) -> float:
    """Bootstrap lower bound of expectancy_R at the ci_low quantile."""
    rng = np.random.default_rng(seed)
    n = len(r_values)
    if n == 0:
        return 0.0
    boot_means = np.array([
        rng.choice(r_values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.quantile(boot_means, ci_low))


def _build_equity_curve(trades: list[dict]) -> np.ndarray:
    """
    Build a simple equity curve from sorted trades.
    Returns array of cumulative net pnl values (starting from 0).
    """
    if not trades:
        return np.array([0.0])
    # Sort by exit_date
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"])
    cumulative = 0.0
    equity = [0.0]
    for t in sorted_trades:
        net_pnl = (t["exit_price"] - t["entry_price"]) * t["quantity"] - t["costs_paid"]
        cumulative += net_pnl
        equity.append(cumulative)
    return np.array(equity)


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Max drawdown as a percentage of peak equity (0 if no trades)."""
    if len(equity) <= 1:
        return 0.0
    # Shift to positive by adding initial capital offset to avoid sign issues
    base = 100_000.0  # notional initial capital
    eq = base + equity
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100.0
    return float(-dd.min())


def _sharpe_from_trades(trades: list[dict], ann_factor: float = 252.0) -> tuple[float, float, float]:
    """
    Compute Sharpe, Deflated Sharpe, and P(Sharpe>0) from trade-level returns.
    Returns (sharpe, deflated_sharpe, p_sharpe_gt_0).
    """
    if not trades or len(trades) < 2:
        return 0.0, 0.0, 0.5

    returns = []
    for t in trades:
        gross = (t["exit_price"] - t["entry_price"]) * t["quantity"]
        net = gross - t["costs_paid"]
        invested = t["entry_price"] * t["quantity"]
        if invested > 0:
            returns.append(net / invested)

    if len(returns) < 2:
        return 0.0, 0.0, 0.5

    r = np.array(returns)
    mean_r = r.mean()
    std_r = r.std(ddof=1)

    if std_r <= 0:
        return 0.0, 0.0, 0.5

    # Annualized Sharpe (scale by trades per year, assuming daily data)
    # Find date span to estimate trades/year
    sorted_t = sorted(trades, key=lambda t: t["entry_date"])
    span_days = (sorted_t[-1]["exit_date"] - sorted_t[0]["entry_date"]).days
    if span_days > 0:
        trades_per_year = len(trades) / span_days * 252
    else:
        trades_per_year = ann_factor

    sharpe = float(mean_r / std_r * np.sqrt(trades_per_year))

    # Deflated Sharpe Ratio (Lopez de Prado 2014) — adjusts for skewness/kurtosis
    T = len(r)
    skew = float(pd.Series(r).skew()) if T >= 3 else 0.0
    kurt = float(pd.Series(r).kurtosis()) if T >= 4 else 0.0  # excess kurtosis
    sr_hat = mean_r / std_r

    # PSR: P(SR > SR*=0) using normal approximation
    # PSR = Φ[√(T-1) * (SR_hat - SR*) / √(1 - γ*SR_hat + (κ/4)*SR_hat²)]
    # where γ = skewness, κ = excess kurtosis
    denom_inner = 1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2
    if denom_inner <= 0:
        denom_inner = 1.0
    dsr_stat = np.sqrt(T - 1) * sr_hat / np.sqrt(denom_inner)
    p_sr_gt_0 = float(stats.norm.cdf(dsr_stat))
    deflated_sharpe = float(stats.norm.ppf(p_sr_gt_0) * std_r / np.sqrt(trades_per_year / T))

    return sharpe, deflated_sharpe, p_sr_gt_0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Trade-level PnL evaluator. The foundation of Phase 0.
    All metrics are computed from trade dicts; no simulation state is required.
    """

    def __init__(self, universe_tag: str = "v0", random_seed: int = 42):
        self.universe_tag = universe_tag
        self.random_seed = random_seed

    def _make_run_id(self) -> str:
        return f"phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def evaluate(self, trades: List[dict]) -> Scorecard:
        """
        Compute Scorecard from a list of trade dicts.

        Required trade fields:
            entry_price  (float): fill price
            exit_price   (float): exit fill price
            stop_price   (float): initial stop, used to compute R
            quantity     (float): shares traded
            costs_paid   (float): total commissions + slippage (both legs)
            entry_date   (datetime): entry timestamp
            exit_date    (datetime): exit timestamp
            symbol       (str): ticker
        """
        run_id = self._make_run_id()
        git_commit = _get_git_commit()
        config_hash = _get_config_hash()
        snapshot_id = _load_snapshot_id()
        n_configs_tried = _load_configs_tried()

        if not trades:
            return self._empty_scorecard(run_id, git_commit, config_hash, snapshot_id, n_configs_tried)

        # Compute R for each trade
        r_values = []
        pct_values = []
        total_costs = 0.0
        total_invested = 0.0

        for t in trades:
            entry = float(t["entry_price"])
            exit_ = float(t["exit_price"])
            stop = float(t["stop_price"])
            qty = float(t["quantity"])
            costs = float(t["costs_paid"])

            R = entry - stop  # per share risk
            if R <= 0:
                # Stop above entry: skip or assign 0 (shouldn't happen for longs)
                R = max(entry * 0.05, 0.01)  # fallback: 5% of entry

            gross_pnl_per_share = exit_ - entry
            costs_per_share = costs / qty if qty > 0 else 0
            net_pnl_per_share = gross_pnl_per_share - costs_per_share
            trade_R = net_pnl_per_share / R

            r_values.append(trade_R)
            pct_values.append(net_pnl_per_share / entry * 100 if entry > 0 else 0)

            total_costs += costs
            total_invested += entry * qty

        r_arr = np.array(r_values)
        pct_arr = np.array(pct_values)

        # Core metrics
        n_trades = len(r_arr)
        wins = r_arr[r_arr > 0]
        losses = r_arr[r_arr <= 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        expectancy_R = float(r_arr.mean())
        expectancy_pct = float(pct_arr.mean())

        # Bootstrap CI
        expectancy_R_ci95_low = _bootstrap_expectancy_ci(r_arr, seed=self.random_seed)

        # Profit factor (in R)
        gross_profit_R = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss_R = float(-losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit_R / gross_loss_R if gross_loss_R > 0 else float("inf")

        # Payoff ratio
        avg_win_R = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss_R = float(losses.mean()) if len(losses) > 0 else 0.0
        payoff_ratio = avg_win_R / abs(avg_loss_R) if avg_loss_R != 0 else float("inf")

        # Trades per month
        sorted_dates = sorted([t["exit_date"] for t in trades])
        span_days = max(1, (sorted_dates[-1] - sorted_dates[0]).days)
        trades_per_month = n_trades / (span_days / 30.0)

        # Equity curve & drawdown
        equity = _build_equity_curve(trades)
        equity_max_drawdown_pct = _max_drawdown_pct(equity)

        # Sharpe metrics
        sharpe, deflated_sharpe, p_sharpe_gt_0 = _sharpe_from_trades(trades)

        # CVaR (worst 5%)
        n_worst = max(1, int(np.ceil(n_trades * 0.05)))
        worst_5pct = np.sort(r_arr)[:n_worst]
        trade_cvar_5_R = float(worst_5pct.mean())
        worst_trade_R = float(r_arr.min())

        # Costs
        total_costs_paid_pct = total_costs / total_invested * 100 if total_invested > 0 else 0.0

        return Scorecard(
            run_id=run_id,
            git_commit=git_commit,
            config_hash=config_hash,
            data_snapshot_id=snapshot_id,
            random_seed=self.random_seed,
            n_configs_tried=n_configs_tried,
            n_trades=n_trades,
            trades_per_month=trades_per_month,
            win_rate=win_rate,
            expectancy_R=expectancy_R,
            expectancy_R_ci95_low=expectancy_R_ci95_low,
            expectancy_pct=expectancy_pct,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            avg_win_R=avg_win_R,
            avg_loss_R=avg_loss_R,
            gross_profit_R=gross_profit_R,
            gross_loss_R=gross_loss_R,
            equity_max_drawdown_pct=equity_max_drawdown_pct,
            sharpe=sharpe,
            deflated_sharpe=deflated_sharpe,
            p_sharpe_gt_0=p_sharpe_gt_0,
            trade_cvar_5_R=trade_cvar_5_R,
            worst_trade_R=worst_trade_R,
            total_costs_paid_pct=total_costs_paid_pct,
            by_tier={},
            universe_tag=self.universe_tag,
        )

    def evaluate_ruleset(self, entry_signals: list[dict], exit_ruleset,
                          bars: dict, cost_model) -> Scorecard:
        """
        Simulate a strategy over a set of bars.

        entry_signals: list of signal dicts with keys:
            symbol, signal_time, signal_bar_idx, stop_price,
            entry_price (next bar open), quantity
        exit_ruleset: callable(open_trade_dict, current_bar_row) -> (exit: bool, reason: str)
        bars: dict of {symbol -> DataFrame with columns [time, open, close, high, low]}
        cost_model: CostModel instance with .total_cost(symbol, qty, price) method
        """
        trades = []

        for sig in entry_signals:
            sym = sig["symbol"]
            if sym not in bars:
                continue
            sym_bars = bars[sym].reset_index(drop=True)
            entry_idx = sig["signal_bar_idx"] + 1  # next bar
            if entry_idx >= len(sym_bars):
                continue

            # Fill at next bar's open
            fill_bar = sym_bars.iloc[entry_idx]
            fill_time = fill_bar["time"]
            fill_price = float(fill_bar["open"])
            stop_price = float(sig["stop_price"])
            qty = float(sig.get("quantity", 100))

            # Assert: fill time must be strictly after signal time
            if fill_time <= sig["signal_time"]:
                raise AssertionError(
                    f"Same-bar fill detected: fill_time={fill_time} <= signal_time={sig['signal_time']}"
                )

            entry_cost = cost_model.total_cost(sym, qty, fill_price)

            # Track open trade
            open_trade = {
                "symbol": sym,
                "entry_price": fill_price,
                "stop_price": stop_price,
                "quantity": qty,
                "entry_date": fill_time,
                "entry_cost": entry_cost,
            }

            # Walk bars from entry+1 to find exit
            exited = False
            for j in range(entry_idx + 1, len(sym_bars)):
                bar_row = sym_bars.iloc[j]
                do_exit, _ = exit_ruleset(open_trade, bar_row)
                if do_exit:
                    exit_price = float(bar_row["open"])  # next-open fill for exits too
                    exit_cost = cost_model.total_cost(sym, qty, exit_price)
                    trades.append({
                        "entry_price": fill_price,
                        "exit_price": exit_price,
                        "stop_price": stop_price,
                        "quantity": qty,
                        "costs_paid": entry_cost + exit_cost,
                        "entry_date": fill_time,
                        "exit_date": bar_row["time"],
                        "symbol": sym,
                    })
                    exited = True
                    break

        return self.evaluate(trades)

    def _empty_scorecard(self, run_id, git_commit, config_hash, snapshot_id, n_configs_tried) -> Scorecard:
        return Scorecard(
            run_id=run_id, git_commit=git_commit, config_hash=config_hash,
            data_snapshot_id=snapshot_id, random_seed=self.random_seed,
            n_configs_tried=n_configs_tried, n_trades=0, trades_per_month=0.0,
            win_rate=0.0, expectancy_R=0.0, expectancy_R_ci95_low=0.0,
            expectancy_pct=0.0, profit_factor=0.0, payoff_ratio=0.0,
            avg_win_R=0.0, avg_loss_R=0.0, gross_profit_R=0.0, gross_loss_R=0.0,
            equity_max_drawdown_pct=0.0, sharpe=0.0, deflated_sharpe=0.0,
            p_sharpe_gt_0=0.5, trade_cvar_5_R=0.0, worst_trade_R=0.0,
            total_costs_paid_pct=0.0, by_tier={}, universe_tag=self.universe_tag,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit test — MUST PASS before any other step runs
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_test():
    """
    Synthetic trades with known expectancy and PF. Asserts recovered within tolerance.
    MUST PASS before Phase 0 can proceed.
    """
    print("\n[UNIT TEST] Evaluator — synthetic trades with known E[R] and PF")

    from datetime import datetime, timedelta

    base_date = datetime(2024, 1, 1)
    qty = 100
    entry = 100.0
    stop = 95.0  # R = 5.0 per share
    R = entry - stop  # = 5.0

    trades = []
    # 100 winners at +2R each, 100 losers at -1R each (n=200 for tight CI)
    for i in range(100):
        exit_p = entry + 2 * R  # +2R
        trades.append({
            "entry_price": entry,
            "exit_price": exit_p,
            "stop_price": stop,
            "quantity": qty,
            "costs_paid": 0.0,  # zero costs for known-value test
            "entry_date": base_date + timedelta(days=i * 2),
            "exit_date": base_date + timedelta(days=i * 2 + 1),
            "symbol": "TEST",
        })
    for i in range(100):
        exit_p = entry - 1 * R  # -1R
        trades.append({
            "entry_price": entry,
            "exit_price": exit_p,
            "stop_price": stop,
            "quantity": qty,
            "costs_paid": 0.0,
            "entry_date": base_date + timedelta(days=i * 2 + 400),
            "exit_date": base_date + timedelta(days=i * 2 + 401),
            "symbol": "TEST",
        })

    # Expected: expectancy = 0.5 * 2R + 0.5 * (-1R) = 0.5R
    # Expected: PF = 100*2R / 100*1R = 2.0
    # With n=200, bootstrap CI95 low ≈ 0.5 - 1.645 * (1.5/sqrt(200)) ≈ 0.33 > 0
    expected_expectancy_R = 0.5
    expected_pf = 2.0

    ev = Evaluator(universe_tag="test", random_seed=42)
    sc = ev.evaluate(trades)

    tol = 0.01
    assert abs(sc.expectancy_R - expected_expectancy_R) < tol, \
        f"Expectancy R mismatch: got {sc.expectancy_R:.4f}, expected {expected_expectancy_R}"
    assert abs(sc.profit_factor - expected_pf) < tol, \
        f"PF mismatch: got {sc.profit_factor:.4f}, expected {expected_pf}"
    assert sc.win_rate == 0.5, f"Win rate mismatch: {sc.win_rate}"
    assert sc.n_trades == 200, f"N trades mismatch: {sc.n_trades}"
    assert sc.expectancy_R_ci95_low > 0, \
        f"Bootstrap CI95 low should be > 0 for positive-expectancy trades: {sc.expectancy_R_ci95_low:.4f}"
    assert sc.total_costs_paid_pct == 0.0, "Costs should be zero in this test"

    print(f"  ✅ expectancy_R = {sc.expectancy_R:.4f} (expected {expected_expectancy_R})")
    print(f"  ✅ profit_factor = {sc.profit_factor:.4f} (expected {expected_pf})")
    print(f"  ✅ win_rate = {sc.win_rate:.2f} (expected 0.50)")
    print(f"  ✅ expectancy_R_ci95_low = {sc.expectancy_R_ci95_low:.4f} > 0")
    print(f"  ✅ total_costs_paid_pct = {sc.total_costs_paid_pct:.4f} == 0")
    print("[UNIT TEST] ✅ PASSED — evaluator.py is correct\n")
    return True


if __name__ == "__main__":
    run_unit_test()
