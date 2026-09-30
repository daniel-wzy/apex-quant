"""
quant/floor_check.py — Mark-to-market portfolio floor check with fail-safe.

Purpose
-------
Provide a hard stop-loss at the portfolio level. If the mark-to-market value
of all open positions plus available cash falls below a configurable floor,
all trading activity is halted until manual review.

Key design decisions
--------------------
1. **Mark-to-market, not cost basis**: The floor is evaluated against current
   market prices, not what was paid. A position that has lost 40% is already
   below floor even if no trade has been closed.

2. **Data-outage fail-safe**: If live price fetching fails for any open
   position, the floor check assumes worst-case (stop price) rather than
   silently passing. "Fail safe, never fail silent."

3. **Separation from trading logic**: This module has no imports from the
   trading bot or indicator layer. It takes a plain dict of positions and a
   price-fetch callable. Easy to test, easy to audit.

Usage
-----
    from quant.floor_check import FloorCheck, Position

    checker = FloorCheck(floor_usd=15_000, notional=20_000)

    positions = [
        Position("AAPL", entry_price=150.0, stop_price=142.5, quantity=10),
        Position("MSFT", entry_price=300.0, stop_price=285.0, quantity=5),
    ]

    status = checker.check(positions, price_fn=my_price_fetcher, cash=5_000.0)
    if status.halt:
        # Stop all trading immediately
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single open position."""
    symbol:      str
    entry_price: float   # fill price (informational only)
    stop_price:  float   # initial hard stop (used as worst-case in fail-safe)
    quantity:    float   # shares held


@dataclass
class FloorStatus:
    """Result of a floor check."""
    halt:          bool         # True → stop all trading immediately
    reason:        str          # human-readable explanation
    mtm_value:     float        # total portfolio value (cash + positions at live prices)
    floor_usd:     float        # configured floor
    margin_usd:    float        # mtm_value - floor_usd  (negative → below floor)
    fail_safe:     bool         # True → triggered because of a data fetch failure
    position_mtm:  dict[str, float] = field(default_factory=dict)  # symbol → mtm value


# ─────────────────────────────────────────────────────────────────────────────
# Floor checker
# ─────────────────────────────────────────────────────────────────────────────

class FloorCheck:
    """
    Mark-to-market portfolio floor enforcer.

    Parameters
    ----------
    floor_usd : float
        Minimum allowable portfolio value. Trading halts if MTM falls below.
    notional : float
        Starting/reference portfolio size (used only for logging/context).
    """

    def __init__(self, floor_usd: float, notional: float) -> None:
        if floor_usd <= 0:
            raise ValueError(f"floor_usd must be positive, got {floor_usd}")
        if floor_usd >= notional:
            raise ValueError(
                f"floor_usd ({floor_usd}) must be less than notional ({notional})"
            )
        self.floor_usd = floor_usd
        self.notional  = notional

    def check(
        self,
        positions: list[Position],
        price_fn:  Callable[[str], float | None],
        cash:      float = 0.0,
    ) -> FloorStatus:
        """
        Evaluate the current portfolio MTM against the floor.

        Parameters
        ----------
        positions : list[Position]
            All currently open positions.
        price_fn : callable(symbol: str) -> float | None
            Function that returns the current market price for a symbol, or
            None if the price is unavailable (data outage, API error, etc.).
            The fail-safe activates if ANY position price is None.
        cash : float
            Cash / uninvested balance (included in MTM total).

        Returns
        -------
        FloorStatus
            See FloorStatus docstring. Always check .halt before placing orders.
        """
        position_mtm: dict[str, float] = {}
        fail_safe    = False
        total_pos_value = 0.0

        for pos in positions:
            price = price_fn(pos.symbol)

            if price is None:
                # Data outage fail-safe: assume the position is at its stop
                # price (worst case we'd realise if stopped out right now).
                worst_case = pos.stop_price * pos.quantity
                position_mtm[pos.symbol] = worst_case
                total_pos_value += worst_case
                fail_safe = True
            else:
                mtm = price * pos.quantity
                position_mtm[pos.symbol] = mtm
                total_pos_value += mtm

        mtm_value  = cash + total_pos_value
        margin_usd = mtm_value - self.floor_usd

        if fail_safe:
            reason = (
                f"FAIL-SAFE: price data unavailable for ≥1 position; "
                f"worst-case MTM {mtm_value:,.2f} vs floor {self.floor_usd:,.2f}"
            )
            halt = True  # always halt on data outage
        elif mtm_value < self.floor_usd:
            reason = (
                f"MTM {mtm_value:,.2f} < floor {self.floor_usd:,.2f} "
                f"(margin: {margin_usd:,.2f})"
            )
            halt = True
        else:
            reason = (
                f"OK — MTM {mtm_value:,.2f} ≥ floor {self.floor_usd:,.2f} "
                f"(margin: +{margin_usd:,.2f})"
            )
            halt = False

        return FloorStatus(
            halt=halt,
            reason=reason,
            mtm_value=mtm_value,
            floor_usd=self.floor_usd,
            margin_usd=margin_usd,
            fail_safe=fail_safe,
            position_mtm=position_mtm,
        )
