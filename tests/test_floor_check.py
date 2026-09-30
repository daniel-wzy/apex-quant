"""
tests/test_floor_check.py — Unit tests for the MTM floor check and
data-outage fail-safe.

Scenarios tested
----------------
- above_floor:    portfolio MTM > floor → no halt
- at_floor_edge:  portfolio MTM exactly at floor → no halt
- below_floor:    portfolio MTM < floor → halt
- fail_safe_none_price: one position price unavailable → halt (fail-safe)
- fail_safe_all_none:   all prices unavailable → halt (fail-safe)
- fail_safe_uses_stop:  worst-case MTM uses stop_price, not entry_price
- cash_counts_toward_mtm: cash balance included in MTM total
- no_positions:   no open positions, only cash → check against cash
"""
from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from quant.floor_check import FloorCheck, FloorStatus, Position


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_checker(floor: float = 15_000, notional: float = 20_000) -> FloorCheck:
    return FloorCheck(floor_usd=floor, notional=notional)


def price_fn_live(prices: dict) -> callable:
    """Return a price_fn that looks up from a dict (None = data unavailable)."""
    def fn(symbol: str) -> float | None:
        return prices.get(symbol)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFloorCheck:

    def test_above_floor_no_halt(self):
        """MTM well above floor → no halt."""
        checker = make_checker(floor=15_000, notional=20_000)
        pos = [Position("AAPL", entry_price=150.0, stop_price=142.5, quantity=100)]
        # Live price 160 → position MTM = 16,000; cash = 2,000 → total = 18,000
        status = checker.check(pos, price_fn=price_fn_live({"AAPL": 160.0}), cash=2_000.0)
        assert not status.halt
        assert not status.fail_safe
        assert status.mtm_value == pytest.approx(18_000.0)
        assert status.margin_usd == pytest.approx(3_000.0)

    def test_at_floor_edge_no_halt(self):
        """MTM exactly equal to floor → no halt (floor is a strict < check)."""
        checker = make_checker(floor=15_000, notional=20_000)
        # 100 shares at 150.0 → 15,000; cash = 0
        pos = [Position("MSFT", entry_price=148.0, stop_price=140.0, quantity=100)]
        status = checker.check(pos, price_fn=price_fn_live({"MSFT": 150.0}), cash=0.0)
        assert not status.halt
        assert status.mtm_value == pytest.approx(15_000.0)

    def test_below_floor_halts(self):
        """MTM below floor → halt."""
        checker = make_checker(floor=15_000, notional=20_000)
        # 100 shares at 140.0 → 14,000; cash = 0 → total = 14,000 < 15,000
        pos = [Position("NVDA", entry_price=155.0, stop_price=145.0, quantity=100)]
        status = checker.check(pos, price_fn=price_fn_live({"NVDA": 140.0}), cash=0.0)
        assert status.halt
        assert not status.fail_safe
        assert status.mtm_value == pytest.approx(14_000.0)
        assert status.margin_usd == pytest.approx(-1_000.0)

    def test_fail_safe_on_none_price(self):
        """If any price is None, fail-safe activates and halt is True."""
        checker = make_checker(floor=15_000, notional=20_000)
        pos = [
            Position("AAPL", entry_price=150.0, stop_price=142.5, quantity=100),
            Position("MSFT", entry_price=300.0, stop_price=285.0, quantity=10),
        ]
        # MSFT price unavailable
        status = checker.check(
            pos,
            price_fn=price_fn_live({"AAPL": 160.0, "MSFT": None}),
            cash=1_000.0,
        )
        assert status.halt
        assert status.fail_safe

    def test_fail_safe_uses_stop_price_not_entry(self):
        """
        Fail-safe worst-case must use stop_price (not entry_price) for the
        MTM of positions with unavailable prices.
        """
        checker = make_checker(floor=1_000, notional=20_000)
        # 10 shares: entry=200, stop=190 → worst-case MTM = 190*10 = 1,900
        pos = [Position("SYM", entry_price=200.0, stop_price=190.0, quantity=10)]
        status = checker.check(pos, price_fn=price_fn_live({"SYM": None}), cash=0.0)
        assert status.fail_safe
        assert status.mtm_value == pytest.approx(190.0 * 10)

    def test_cash_counts_toward_mtm(self):
        """Cash balance is included in the MTM total."""
        checker = make_checker(floor=15_000, notional=20_000)
        pos = []  # no open positions
        # Only cash: 16,000 → above floor
        status = checker.check(pos, price_fn=price_fn_live({}), cash=16_000.0)
        assert not status.halt
        assert status.mtm_value == pytest.approx(16_000.0)

    def test_no_positions_below_floor_halts(self):
        """No open positions, cash below floor → halt."""
        checker = make_checker(floor=15_000, notional=20_000)
        status = checker.check([], price_fn=price_fn_live({}), cash=10_000.0)
        assert status.halt
        assert status.mtm_value == pytest.approx(10_000.0)

    def test_all_prices_none_halts(self):
        """All position prices unavailable → fail-safe halt regardless of floor."""
        checker = make_checker(floor=1, notional=20_000)  # floor near zero
        pos = [
            Position("A", entry_price=100.0, stop_price=90.0, quantity=10),
            Position("B", entry_price=200.0, stop_price=180.0, quantity=5),
        ]
        status = checker.check(pos, price_fn=price_fn_live({}), cash=0.0)
        assert status.halt      # always halt on data outage
        assert status.fail_safe

    def test_invalid_floor_raises(self):
        """floor_usd <= 0 or >= notional should raise ValueError."""
        with pytest.raises(ValueError):
            FloorCheck(floor_usd=0, notional=20_000)
        with pytest.raises(ValueError):
            FloorCheck(floor_usd=-100, notional=20_000)
        with pytest.raises(ValueError):
            FloorCheck(floor_usd=20_000, notional=20_000)
        with pytest.raises(ValueError):
            FloorCheck(floor_usd=25_000, notional=20_000)
