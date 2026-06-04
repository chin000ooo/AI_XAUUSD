"""
fracdiff.py — fractional differentiation (López de Prado, AFML Ch. 5).

XAUUSD ran from ~$1960 (2023) to ~$4480 (2026): a strongly non-stationary
series. Integer differencing (returns) makes it stationary but erases all
memory. Fractional differencing finds the *smallest* d that passes the ADF
stationarity test while preserving as much memory (correlation) as possible.

Fixed-width window FFD is used (constant weights, no expanding window).
statsmodels is optional — if absent we fall back to a sensible default d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAS_STATSMODELS = False

DEFAULT_D = 0.35


def get_ffd_weights(d: float, thresh: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    """Fixed-width FFD weights for order ``d`` until |w_k| < thresh."""
    w = [1.0]
    k = 1
    while k < max_size:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1])  # oldest .. newest


def fractional_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """Fractionally difference ``series`` with fixed-width window FFD.

    out[i] = sum_k w[k] * x[i-width+1+k]  (w ordered oldest..newest).
    Vectorised with np.correlate when the series has no gaps; otherwise a
    NaN-safe loop. width-1 leading values are NaN (insufficient history).
    """
    w = get_ffd_weights(d, thresh)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < width:
        return pd.Series(out, index=series.index, name=f"ffd_{series.name or 'x'}")
    if np.all(np.isfinite(x)):
        # np.correlate(x, w, 'valid')[j] = sum_k x[j+k] w[k] == out[j+width-1]
        out[width - 1:] = np.correlate(x, w, mode="valid")
    else:
        for i in range(width - 1, len(x)):
            window = x[i - width + 1: i + 1]
            if np.any(~np.isfinite(window)):
                continue
            out[i] = float(np.dot(w, window))
    return pd.Series(out, index=series.index, name=f"ffd_{series.name or 'x'}")


def _adf_pvalue(series: pd.Series) -> float | None:
    if not _HAS_STATSMODELS:
        return None
    s = series.dropna()
    if len(s) < 50:
        return None
    try:
        return float(adfuller(s, maxlag=1, regression="c", autolag=None)[1])
    except Exception:
        return None


def find_min_d(series: pd.Series, d_grid=None, thresh: float = 1e-5,
               p_target: float = 0.05) -> dict:
    """Search the smallest d whose FFD series is stationary at ``p_target``.

    Returns a dict with chosen ``d``, the ADF p-value (if statsmodels present),
    correlation between original and differenced series, and method used.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)

    if not _HAS_STATSMODELS:
        d = DEFAULT_D
        ffd = fractional_diff_ffd(series, d, thresh)
        corr = float(series.corr(ffd))
        return {"d": d, "adf_pvalue": None, "corr": corr,
                "method": "default (statsmodels unavailable)", "passed": None}

    chosen = None
    last_p = None
    for d in d_grid:
        ffd = fractional_diff_ffd(series, float(d), thresh)
        p = _adf_pvalue(ffd)
        last_p = p
        if p is not None and p <= p_target:
            chosen = float(d)
            break
    if chosen is None:
        chosen = float(d_grid[-1])
    ffd = fractional_diff_ffd(series, chosen, thresh)
    corr = float(series.corr(ffd))
    return {"d": chosen, "adf_pvalue": _adf_pvalue(ffd), "corr": corr,
            "method": "ADF search (statsmodels)", "passed": (last_p is not None and last_p <= p_target)}
