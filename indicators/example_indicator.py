"""
indicators/example_indicator.py — Moving-average crossover indicator.

Demonstrates the Indicator interface with a simple, fully transparent
implementation. This is intentionally basic — it is here to show the
pipeline structure, not to represent the live strategy's indicators.

Signals produced
----------------
MACROSS_BUY  : fast MA crosses above slow MA (golden cross)
MACROSS_SELL : fast MA crosses below slow MA (death cross)

These are registered in feature_store.SIGNAL_COLUMNS as:
    "MACROSS_BUY":  ("MACROSS", "BUY")
    "MACROSS_SELL": ("MACROSS", "SELL")
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from indicators.interface import Indicator


class MACrossoverIndicator(Indicator):
    """Simple dual-SMA crossover signal generator."""

    name = "MACROSS"

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        """
        Parameters
        ----------
        fast : int
            Period of the fast simple moving average.
        slow : int
            Period of the slow simple moving average. Must be > fast.
        """
        if slow <= fast:
            raise ValueError(f"slow ({slow}) must be greater than fast ({fast})")
        self.fast = fast
        self.slow = slow

    @property
    def signal_columns(self) -> list[str]:
        return ["MACROSS_BUY", "MACROSS_SELL"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append MACROSS_BUY and MACROSS_SELL columns to df.

        A crossover is detected by comparing the sign of (fast_ma - slow_ma)
        between the current bar and the previous bar. Both columns are 0/1
        integers; exactly one can be 1 on any given bar (never both).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a ``close`` column.

        Returns
        -------
        pd.DataFrame
            Original df extended with MACROSS_BUY and MACROSS_SELL columns.
        """
        out = df.copy()
        close = out["close"].astype(float)

        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        diff = fast_ma - slow_ma
        prev_diff = diff.shift(1)

        # Golden cross: fast crosses above slow (diff flips from ≤0 to >0)
        out["MACROSS_BUY"] = (
            (diff > 0) & (prev_diff <= 0)
        ).fillna(False).astype(int)

        # Death cross: fast crosses below slow (diff flips from ≥0 to <0)
        out["MACROSS_SELL"] = (
            (diff < 0) & (prev_diff >= 0)
        ).fillna(False).astype(int)

        return out


# Module-level helper so feature_store can call ``module.compute(df)``
# directly (matching the live indicator module pattern).
_indicator = MACrossoverIndicator()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level entry point used by feature_store.INDICATOR_MODULES."""
    return _indicator.compute(df)
