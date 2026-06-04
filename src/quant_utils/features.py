"""
features.py — pandas-native indicators, candle/session/stat features, Ehlers DSP.

Everything here is *causal*: a feature at bar t uses only OHLC up to and
including bar t. (The forward-looking part of the pipeline lives entirely in
labeling.py.) TA-Lib is NOT required; these are dependency-light implementations.

Volume features are gated behind ``has_volume`` and skipped (with a logged note)
on this dataset, which has no volume column.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------
# returns & rolling statistics
# --------------------------------------------------------------------------
def log_return(close: pd.Series, n: int = 1) -> pd.Series:
    return np.log(close).diff(n)


def pct_return(close: pd.Series, n: int = 1) -> pd.Series:
    return close.pct_change(n)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(high, low, close, k: int = 14, d: int = 3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    kf = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return kf, kf.rolling(d).mean()


def true_range(high, low, close) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False).mean()


def adx(high, low, close, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / atr_
    minus_di = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def bollinger(close, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width


def parkinson_vol(high, low, n: int = 14) -> pd.Series:
    return np.sqrt((1.0 / (4 * np.log(2))) * (np.log(high / low) ** 2).rolling(n).mean())


def garman_klass_vol(open_, high, low, close, n: int = 14) -> pd.Series:
    rs = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt(rs.rolling(n).mean().clip(lower=0))


def realized_vol(close, n: int = 20) -> pd.Series:
    return log_return(close).rolling(n).std() * np.sqrt(n)


def rolling_autocorr(s: pd.Series, n: int = 20, lag: int = 1) -> pd.Series:
    return s.rolling(n).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)


def _hurst_window(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 20 or not np.all(np.isfinite(ts)):
        return np.nan
    lags = np.arange(2, min(20, len(ts) // 2))
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    mask = tau > 0
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0])


def rolling_hurst(close: pd.Series, n: int = 100) -> pd.Series:
    return np.log(close).rolling(n).apply(_hurst_window, raw=True)


def rolling_entropy(returns: pd.Series, n: int = 50, bins: int = 8) -> pd.Series:
    def _ent(x):
        x = x[np.isfinite(x)]
        if len(x) < 5:
            return np.nan
        h, _ = np.histogram(x, bins=bins)
        p = h / h.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return returns.rolling(n).apply(_ent, raw=True)


# --------------------------------------------------------------------------
# Ehlers DSP filters (low-lag, match the user's Pine work)
# --------------------------------------------------------------------------
def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers 2-pole SuperSmoother (Butterworth)."""
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3
    x = series.to_numpy(dtype=float)
    y = x.copy()
    for i in range(2, len(x)):
        if not np.isfinite(x[i]):
            y[i] = y[i - 1]
            continue
        y[i] = c1 * (x[i] + x[i - 1]) / 2.0 + c2 * y[i - 1] + c3 * y[i - 2]
    return pd.Series(y, index=series.index)


def highpass_filter(series: pd.Series, period: int = 48) -> pd.Series:
    """Ehlers 2-pole high-pass (removes low-frequency trend)."""
    a = 0.707 * 2 * math.pi / period
    alpha = (math.cos(a) + math.sin(a) - 1) / math.cos(a)
    x = series.to_numpy(dtype=float)
    y = np.zeros_like(x)
    k = (1 - alpha / 2) ** 2
    for i in range(2, len(x)):
        y[i] = k * (x[i] - 2 * x[i - 1] + x[i - 2]) + 2 * (1 - alpha) * y[i - 1] - (1 - alpha) ** 2 * y[i - 2]
    return pd.Series(y, index=series.index)


def roofing_filter(series: pd.Series, hp_period: int = 48, ss_period: int = 10) -> pd.Series:
    """High-pass then SuperSmoother — Ehlers' band-pass 'roofing' filter."""
    return super_smoother(highpass_filter(series, hp_period), ss_period)


def decycler(series: pd.Series, period: int = 60) -> pd.Series:
    """Ehlers Decycler — trend with cyclic component removed."""
    a = 2 * math.pi / period
    alpha = (math.cos(a) + math.sin(a) - 1) / math.cos(a)
    x = series.to_numpy(dtype=float)
    y = x.copy()
    for i in range(1, len(x)):
        y[i] = (alpha / 2) * (x[i] + x[i - 1]) + (1 - alpha) * y[i - 1]
    return pd.Series(y, index=series.index)


def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers Fisher Transform of the price position in its recent range."""
    hh = series.rolling(period).max()
    ll = series.rolling(period).min()
    raw = 2 * ((series - ll) / (hh - ll).replace(0, np.nan) - 0.5)
    r = raw.clip(-0.999, 0.999).fillna(0).to_numpy(dtype=float)
    val = np.zeros(len(r))
    fish = np.zeros(len(r))
    for i in range(1, len(r)):
        val[i] = 0.66 * r[i] + 0.67 * val[i - 1]
        val[i] = min(max(val[i], -0.999), 0.999)
        fish[i] = 0.5 * math.log((1 + val[i]) / (1 - val[i])) + 0.5 * fish[i - 1]
    return pd.Series(fish, index=series.index)


def inverse_fisher(series: pd.Series, scale: float = 1.0) -> pd.Series:
    """Inverse Fisher Transform — squashes an unbounded oscillator into (-1,1)."""
    x = (scale * series).to_numpy(dtype=float)
    e2 = np.exp(2 * x)
    return pd.Series((e2 - 1) / (e2 + 1), index=series.index)


# --------------------------------------------------------------------------
# candle & session features
# --------------------------------------------------------------------------
def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = c - o
    out = pd.DataFrame(index=df.index)
    out["cdl_body"] = body
    out["cdl_body_abs"] = body.abs()
    out["cdl_range"] = h - l
    out["cdl_upper_wick"] = h - np.maximum(o, c)
    out["cdl_lower_wick"] = np.minimum(o, c) - l
    out["cdl_body_to_range"] = body.abs() / rng
    out["cdl_upper_wick_ratio"] = (h - np.maximum(o, c)) / rng
    out["cdl_lower_wick_ratio"] = (np.minimum(o, c) - l) / rng
    out["cdl_close_pos"] = (c - l) / rng           # 0 = closed on low, 1 = on high
    out["cdl_dir"] = np.sign(body)
    return out


def session_features(index: pd.DatetimeIndex, data_tz: str = "UTC", sessions=None) -> pd.DataFrame:
    """Clock-based session flags. ``data_tz`` is the assumed wall-clock zone of
    the bar timestamps — WRONG offset silently corrupts every session feature,
    so it must be verified against the broker."""
    idx = index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(data_tz) if (data_tz and data_tz.upper() != "UTC") else idx
    h = local.hour
    out = pd.DataFrame(index=index)
    out["hour"] = h
    out["day_of_week"] = local.dayofweek
    out["month"] = local.month
    sessions = sessions or {"asian": (0, 8), "london": (8, 16), "ny": (13, 21)}

    def _flag(rng):
        a, b = rng
        return ((h >= a) & (h < b)).astype(int)

    out["sess_asian"] = _flag(sessions["asian"])
    out["sess_london"] = _flag(sessions["london"])
    out["sess_ny"] = _flag(sessions["ny"])
    out["sess_london_ny_overlap"] = (out["sess_london"].astype(bool) & out["sess_ny"].astype(bool)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    return out


# --------------------------------------------------------------------------
# master assembler
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Build the full causal feature matrix from a single-timeframe OHLC frame.

    Returns ``(features_df, meta)``. ``features_df`` always contains an ``atr``
    column (consumed by the triple-barrier labeller). ``meta`` records skipped
    volume features and warnings.
    """
    cfg = cfg or {}
    data_tz = cfg.get("data_tz", "UTC")
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    f = pd.DataFrame(index=df.index)
    meta = {"skipped": [], "warnings": []}

    # --- returns & stats ---
    f["ret_log_1"] = log_return(c, 1)
    f["ret_pct_1"] = pct_return(c, 1)
    for n in (5, 10, 20):
        f[f"ret_log_{n}"] = log_return(c, n)
    r = f["ret_log_1"]
    for n in (10, 20, 50):
        f[f"ret_mean_{n}"] = r.rolling(n).mean()
        f[f"ret_std_{n}"] = r.rolling(n).std()
    f["ret_skew_20"] = r.rolling(20).skew()
    f["ret_kurt_20"] = r.rolling(20).kurt()
    f["ret_autocorr_20"] = rolling_autocorr(r, 20, 1)
    f["entropy_50"] = rolling_entropy(r, 50, 8)
    if cfg.get("use_hurst", True):
        f["hurst_100"] = rolling_hurst(c, cfg.get("hurst_window", 100))

    # --- trend ---
    for span in (10, 20, 50, 100, 200):
        e = ema(c, span)
        f[f"ema_{span}"] = e
        f[f"px_vs_ema_{span}"] = (c - e) / e
        f[f"ema_slope_{span}"] = e.diff() / e.shift(1)
        f[f"px_above_ema_{span}"] = (c > e).astype(int)
    adx_, pdi, mdi = adx(h, l, c, cfg.get("adx_period", 14))
    f["adx_14"], f["plus_di_14"], f["minus_di_14"] = adx_, pdi, mdi

    # --- momentum ---
    f["rsi_14"] = rsi(c, cfg.get("rsi_period", 14))
    ml, ms, mh = macd(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = ml, ms, mh
    kf, kd = stochastic(h, l, c)
    f["stoch_k"], f["stoch_d"] = kf, kd
    for n in (5, 10, 20):
        f[f"roc_{n}"] = c.pct_change(n) * 100

    # --- volatility ---
    f["atr"] = atr(h, l, c, cfg.get("atr_period", 14))
    f["atr_pct"] = f["atr"] / c                      # % normalisation for non-stationary price
    _, bu, bl, bw = bollinger(c, 20, 2.0)
    f["bb_width"] = bw
    f["realized_vol_20"] = realized_vol(c, 20)
    f["parkinson_14"] = parkinson_vol(h, l, 14)
    f["garman_klass_14"] = garman_klass_vol(o, h, l, c, 14)

    # --- candle ---
    f = pd.concat([f, candle_features(df)], axis=1)

    # --- Ehlers DSP ---
    if cfg.get("use_ehlers", True):
        f["ehlers_ss_10"] = super_smoother(c, 10)
        f["ehlers_roofing"] = roofing_filter(c, 48, 10)
        f["ehlers_decycler_60"] = decycler(c, 60)
        f["px_vs_decycler"] = (c - f["ehlers_decycler_60"]) / f["ehlers_decycler_60"]
        f["ehlers_fisher_10"] = fisher_transform(c, 10)
        f["ehlers_inv_fisher_rsi"] = inverse_fisher((f["rsi_14"] - 50) / 25.0)

    # --- session / time (uses configurable DATA_TZ) ---
    f = pd.concat([f, session_features(df.index, data_tz=data_tz, sessions=cfg.get("sessions"))], axis=1)

    # --- volume (auto-skip — this dataset has none) ---
    if any(col in df.columns for col in ("volume", "tick_volume", "vol")):
        vcol = next(col for col in ("volume", "tick_volume", "vol") if col in df.columns)
        f["vol_z_20"] = (df[vcol] - df[vcol].rolling(20).mean()) / df[vcol].rolling(20).std()
    else:
        meta["skipped"].append("volume_features")
        meta["warnings"].append("no volume column present — OBV / volume-z / VWAP features skipped")

    meta["n_features"] = int(f.shape[1])
    return f, meta
