"""
quant_utils — leakage-safe XAUUSD direction & signal engine utilities.

Heavy, reusable, unit-testable logic lives here. The notebooks
(01_preprocess / 02_train / 03_backtest) only orchestrate, validate and display.

Design rules baked into this package (see README §Anti-leakage):
  * Never shuffle / random-split time series.
  * Higher-timeframe features only ever see the last *closed* HTF bar.
  * The future label builds the target but never enters any feature.
  * Every dependency is optional with a graceful fallback — nothing hard-crashes.
"""

__version__ = "0.1.0"

__all__ = [
    "env",
    "io_data",
    "features",
    "mtf",
    "fracdiff",
    "labeling",
    "cv",
    "models",
    "regime",
    "backtest",
    "metrics",
    "export",
    "validation",
]
