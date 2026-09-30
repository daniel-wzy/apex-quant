"""
indicators/interface.py — Abstract base class for all technical indicators.

Every indicator in the pipeline must implement this contract. Doing so
ensures feature_store.compute_indicator_columns() can call compute() on
any indicator module without knowing its internal logic.

Signal columns
--------------
compute() must return the same DataFrame it received, extended with one or
more *boolean* signal columns (int 0/1, never float). Column names must
match the keys registered in feature_store.SIGNAL_COLUMNS so the feature
store can route them to the correct (indicator, direction) metadata.

Registration
------------
Add new indicators to:
  1. feature_store.INDICATOR_MODULES  — the name→module map
  2. feature_store.SIGNAL_COLUMNS     — per-column (indicator, direction)
"""
import pandas as pd
from abc import ABC, abstractmethod


class Indicator(ABC):
    """Base class for all technical indicators in the pipeline."""

    #: Short human-readable name, e.g. "MACROSS". Used in logging.
    name: str = ""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame of OHLCV bars, return the same DataFrame with
        signal columns appended.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum: time, open, high, low, close, volume.
            Index should be positional (reset_index before calling).

        Returns
        -------
        pd.DataFrame
            Same rows as input, same OHLCV columns, plus one or more boolean
            (int 0/1) signal columns defined by this indicator.

        Lookahead discipline
        --------------------
        All signal columns must be computable from data available AT (or
        before) each bar's close. No forward-looking calculations allowed.
        """
        ...

    @property
    def signal_columns(self) -> list[str]:
        """
        Ordered list of boolean signal column names this indicator produces.

        Must match the keys registered in feature_store.SIGNAL_COLUMNS.
        """
        raise NotImplementedError
