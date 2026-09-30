"""
tests/test_evaluator.py — Unit tests for quant/evaluator.py.

Uses synthetic trades with known expectancy so the correct value is verifiable
by hand. This is the Phase 0 foundation: if these tests fail, nothing else
in the pipeline should be trusted.

Known-value tests
-----------------
- test_known_expectancy:   5 wins @+2R, 5 losses @-1R → E[R] = 0.5
- test_zero_expectancy:    50 trades, 50/50 +1R/-1R → E[R] ≈ 0
- test_bootstrap_ci_contains_truth: 100 trades E[R]=1.0R, CI must cover truth
- test_profit_factor:      gross_profit_R / gross_loss_R correctness
- test_worst_trade_cvar:   CVaR on known distribution
- test_empty_trades:       empty input returns valid zero scorecard
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta
from quant.evaluator import Evaluator, Scorecard


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_trades(
    n_wins: int,
    n_losses: int,
    win_R: float,
    loss_R: float,
    entry: float = 100.0,
    stop: float  = 95.0,
    qty:   float = 100.0,
    costs: float = 0.0,
    base_date: datetime = datetime(2024, 1, 1),
) -> list[dict]:
    """Build synthetic trade dicts with known R outcomes."""
    R = entry - stop   # per share, e.g. 5.0
    trades = []
    for i in range(n_wins):
        exit_p = entry + win_R * R
        trades.append({
            "entry_price": entry,
            "exit_price":  exit_p,
            "stop_price":  stop,
            "quantity":    qty,
            "costs_paid":  costs,
            "entry_date":  base_date + timedelta(days=i * 2),
            "exit_date":   base_date + timedelta(days=i * 2 + 1),
            "symbol":      "WIN",
        })
    for i in range(n_losses):
        exit_p = entry + loss_R * R   # loss_R is negative
        trades.append({
            "entry_price": entry,
            "exit_price":  exit_p,
            "stop_price":  stop,
            "quantity":    qty,
            "costs_paid":  costs,
            "entry_date":  base_date + timedelta(days=500 + i * 2),
            "exit_date":   base_date + timedelta(days=500 + i * 2 + 1),
            "symbol":      "LOSS",
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluator:

    def test_known_expectancy(self):
        """5 wins @+2R, 5 losses @-1R → E[R] = 0.5R, PF = 2.0."""
        trades = make_trades(n_wins=5, n_losses=5, win_R=2.0, loss_R=-1.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)

        assert sc.n_trades == 10
        assert abs(sc.expectancy_R - 0.5) < 0.01, f"E[R]={sc.expectancy_R:.4f}"
        assert abs(sc.profit_factor - 2.0) < 0.01, f"PF={sc.profit_factor:.4f}"
        assert sc.win_rate == 0.5, f"WR={sc.win_rate}"

    def test_zero_expectancy(self):
        """50 wins @+1R, 50 losses @-1R → E[R] ≈ 0, PF ≈ 1.0."""
        trades = make_trades(n_wins=50, n_losses=50, win_R=1.0, loss_R=-1.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)

        assert abs(sc.expectancy_R) < 0.01, f"E[R]={sc.expectancy_R:.4f}"
        assert abs(sc.profit_factor - 1.0) < 0.01, f"PF={sc.profit_factor:.4f}"
        assert sc.win_rate == 0.5

    def test_bootstrap_ci_contains_truth(self):
        """
        100 trades with E[R]=1.0R (wins @+2R, losses @0R, 50/50).
        Bootstrap 95% CI lower bound should be > 0.
        With n=100 and E[R]=1.0, the CI is reliably positive.
        """
        # 50 wins at +2R, 50 losses at 0R → E[R] = 1.0R
        trades = make_trades(n_wins=50, n_losses=50, win_R=2.0, loss_R=0.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)

        assert sc.expectancy_R_ci95_low > 0, (
            f"Bootstrap CI95 low = {sc.expectancy_R_ci95_low:.4f}, expected > 0"
        )
        # Mean should be close to 1.0R
        assert abs(sc.expectancy_R - 1.0) < 0.01, f"E[R]={sc.expectancy_R:.4f}"

    def test_profit_factor(self):
        """Verify profit_factor = gross_profit_R / gross_loss_R."""
        # 3 wins @+3R, 4 losses @-1R
        # gross_profit = 9R, gross_loss = 4R → PF = 2.25
        trades = make_trades(n_wins=3, n_losses=4, win_R=3.0, loss_R=-1.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)

        expected_pf = (3 * 3.0) / (4 * 1.0)   # 9 / 4 = 2.25
        assert abs(sc.profit_factor - expected_pf) < 0.02, (
            f"PF={sc.profit_factor:.4f}, expected {expected_pf}"
        )

    def test_worst_trade_cvar(self):
        """
        Verify CVaR(5%) uses the worst trades.
        10 trades: 9 wins @+1R, 1 loss @-5R.
        worst 5% of 10 = ceil(10*0.05) = 1 trade → CVaR = -5R.
        """
        trades = make_trades(n_wins=9, n_losses=1, win_R=1.0, loss_R=-5.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)

        assert abs(sc.worst_trade_R - (-5.0)) < 0.01, f"worst={sc.worst_trade_R}"
        # CVaR (worst 5% of 10 = 1 trade) should equal worst trade
        assert abs(sc.trade_cvar_5_R - (-5.0)) < 0.01, f"CVaR={sc.trade_cvar_5_R}"

    def test_zero_costs_passes_through(self):
        """total_costs_paid_pct == 0 when costs_paid=0 for all trades."""
        trades = make_trades(n_wins=5, n_losses=5, win_R=2.0, loss_R=-1.0, costs=0.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate(trades)
        assert sc.total_costs_paid_pct == 0.0

    def test_costs_reduce_expectancy(self):
        """Positive costs_paid must reduce net expectancy vs zero-cost baseline."""
        no_cost = make_trades(n_wins=10, n_losses=10, win_R=2.0, loss_R=-1.0, costs=0.0)
        with_cost = make_trades(n_wins=10, n_losses=10, win_R=2.0, loss_R=-1.0, costs=5.0)
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc_nc = ev.evaluate(no_cost)
        sc_wc = ev.evaluate(with_cost)
        assert sc_wc.expectancy_R < sc_nc.expectancy_R, "Costs should reduce expectancy"

    def test_empty_trades_returns_zero_scorecard(self):
        """evaluate([]) must return a Scorecard with all-zero metrics."""
        ev = Evaluator(universe_tag="test", random_seed=42)
        sc = ev.evaluate([])
        assert isinstance(sc, Scorecard)
        assert sc.n_trades == 0
        assert sc.expectancy_R == 0.0
        assert sc.profit_factor == 0.0
        assert sc.win_rate == 0.0
