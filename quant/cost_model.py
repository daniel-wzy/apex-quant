"""
quant/cost_model.py — Realistic cost model for backtesting.

Models
------
- Moomoo US commission: $0.0049/share, min $0.99/trade
- Slippage: spread + simple market impact, scaled by vol tier
- Vol tiers: low/mid/high based on average daily volume and price range

Vol tier assignment
-------------------
- low  : mega-cap, highly-liquid symbols with tight spreads
- mid  : standard liquid names (default)
- high : small caps, low-liquidity, wide spreads
"""
from __future__ import annotations

# Mega-cap / highly-liquid symbols → low slippage.
# These are illustrative examples only; the live universe is larger and
# reviewed on a per-symbol basis using trailing average daily volume.
LOW_TIER = {
    "US.AAPL", "US.MSFT", "US.GOOGL", "US.AMZN",  # illustrative examples only
}

# High-beta small caps → wide slippage.
# These are illustrative examples only.
HIGH_TIER = {
    "US.RKLB", "US.OKLO", "US.ASTS", "US.NOK",  # illustrative examples only
}


class CostModel:
    MOOMOO_COMMISSION = 0.0049   # per share
    MOOMOO_MIN        = 0.99     # per trade side

    def commission(self, quantity: float, price: float) -> float:
        """Per-side commission (applies once per fill)."""
        return max(self.MOOMOO_MIN, quantity * self.MOOMOO_COMMISSION)

    def slippage(self, symbol: str, quantity: float, price: float,
                 vol_tier: str = "mid") -> float:
        """
        Estimated spread + market impact per side.

        spread: % of price, scaled by vol_tier
        impact: proportional to sqrt(qty) for simple linear market impact
        """
        spreads = {"low": 0.01, "mid": 0.03, "high": 0.08}   # % of price
        spread_pct  = spreads.get(vol_tier, 0.03) / 100.0
        spread_cost = price * spread_pct * quantity

        impact_per_share = {"low": 0.001, "mid": 0.005, "high": 0.015}.get(vol_tier, 0.005)
        impact_cost = impact_per_share * quantity

        return spread_cost + impact_cost

    def get_vol_tier(self, symbol: str) -> str:
        """Classify a symbol into a vol tier for slippage estimation."""
        sym = symbol.upper()
        if sym in LOW_TIER:
            return "low"
        if sym in HIGH_TIER:
            return "high"
        return "mid"

    def total_cost(self, symbol: str, quantity: float, price: float,
                   vol_tier: str | None = None) -> float:
        """Total one-side cost: commission + slippage."""
        if vol_tier is None:
            vol_tier = self.get_vol_tier(symbol)
        return self.commission(quantity, price) + self.slippage(symbol, quantity, price, vol_tier)

    def round_trip_cost(self, symbol: str, quantity: float, entry_price: float,
                        exit_price: float, vol_tier: str | None = None) -> float:
        """Full round-trip cost (entry + exit legs)."""
        if vol_tier is None:
            vol_tier = self.get_vol_tier(symbol)
        return (
            self.total_cost(symbol, quantity, entry_price, vol_tier)
            + self.total_cost(symbol, quantity, exit_price, vol_tier)
        )
