"""
indicators/ — Technical indicator layer.

Each indicator implements the Indicator abstract base class defined in
indicators/interface.py. The live implementations are proprietary; this
package ships a demonstrable moving-average crossover example so the
pipeline structure can be exercised end-to-end against synthetic data.
"""
from indicators.example_indicator import MACrossoverIndicator

__all__ = ["MACrossoverIndicator"]
