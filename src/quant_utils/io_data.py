"""
io_data.py — robust CSV loader, timeframe mapping, validation helpers.

The real data are native per-timeframe TradingView/GBE exports:
    time,open,high,low,close
    1680472800,1968.22,1968.55,1960.14,1962.52

  * ``time`` is Unix epoch in SECONDS (auto-detected vs ms/us/ns).
  * There is NO volume column — volume features are auto-skipped downstream.
  * Tick files (``1T``) are ``time,close`` only; treated as optional/advanced.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# timeframe token (as it appears in the filename) -> canonical name
_TF_CANON = {
    "1": "M1", "2": "M2", "3": "M3", "5": "M5", "10": "M10", "15": "M15",
    "30": "M30", "45": "M45", "60": "H1", "120": "H2", "180": "H3", "240": "H4",
    "1D": "D1", "1W": "W1", "1M": "MN1", "3M": "Q1", "6M": "S1", "12M": "Y1",
}
_TICK_TOKENS = {"1T", "10T", "100T", "1000T"}

# canonical name -> pandas resample rule (for the resample fallback only)
TF_RESAMPLE = {
    "M1": "1min", "M2": "2min", "M3": "3min", "M5": "5min", "M10": "10min",
    "M15": "15min", "M30": "30min", "M45": "45min", "H1": "1h", "H2": "2h",
    "H3": "3h", "H4": "4h", "D1": "1D", "W1": "1W", "MN1": "1MS",
}

# canonical name -> bar duration (used by the leakage-safe MTF merge)
TF_TIMEDELTA = {
    "M1": pd.Timedelta(minutes=1), "M2": pd.Timedelta(minutes=2), "M3": pd.Timedelta(minutes=3),
    "M5": pd.Timedelta(minutes=5), "M10": pd.Timedelta(minutes=10), "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30), "M45": pd.Timedelta(minutes=45), "H1": pd.Timedelta(hours=1),
    "H2": pd.Timedelta(hours=2), "H3": pd.Timedelta(hours=3), "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1), "W1": pd.Timedelta(weeks=1), "MN1": pd.Timedelta(days=30),
}


def tf_timedelta(tf: str) -> pd.Timedelta:
    if tf not in TF_TIMEDELTA:
        raise ValueError(f"unknown timeframe '{tf}'")
    return TF_TIMEDELTA[tf]


def detect_timeframe_from_name(filename: str) -> str | None:
    """Map a raw filename to a canonical timeframe, or None if unrecognised.

    Handles both our canonical names (``XAUUSD_H1.csv``) and the raw GBE export
    names (``GBEBROKERS_XAUUSD, 60_Start_To_02-06-2026.csv``,
    ``GBEBROKERS_XAUUSD, 1D_a4759.csv``)."""
    name = Path(filename).name
    # canonical: ..._H1.csv / ..._M15.csv
    m = re.search(r"_(M1|M2|M3|M5|M10|M15|M30|M45|H1|H2|H3|H4|D1|W1|MN1)\.csv$", name, re.I)
    if m:
        return m.group(1).upper()
    # GBE: "..., 60_..." / "..., 1D_..." / "..., 1000T_..."
    m = re.search(r",\s*([0-9]+T|[0-9]+[DWM]|[0-9]+)_", name)
    if m:
        tok = m.group(1).upper()
        if tok in _TICK_TOKENS:
            return "TICK_" + tok
        return _TF_CANON.get(tok)
    return None


def _parse_epoch(series: pd.Series) -> tuple[pd.DatetimeIndex, str]:
    """Parse an epoch column to a UTC DatetimeIndex, auto-detecting the unit.

    Magnitudes: seconds ~1.7e9, ms ~1.7e12, us ~1.7e15, ns ~1.7e18.
    Float sub-second epochs (tick files) are handled too."""
    v = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(v.to_numpy(dtype=float)))
    if med > 1e17:
        unit = "ns"
    elif med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    else:
        unit = "s"
    dt = pd.to_datetime(v, unit=unit, origin="unix", utc=True)
    return dt, unit


def parse_time_column(series: pd.Series) -> pd.DatetimeIndex:
    """Public helper: parse a time column (numeric epoch OR ISO string) to UTC.
    Used by notebooks when merging optional external series (DXY / real yield)."""
    tnum = pd.to_numeric(series, errors="coerce")
    if tnum.notna().mean() > 0.9:
        return _parse_epoch(tnum)[0]
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_ohlcv(path, data_tz: str = "UTC", relabel_naive_as_tz: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load one OHLC(V) CSV into a UTC-indexed, de-duplicated, sorted frame.

    Parameters
    ----------
    data_tz : str
        Assumed wall-clock timezone of the *session flags* (NOT applied here).
        Epoch is absolute; this is recorded in meta for downstream session logic.
    relabel_naive_as_tz : bool
        If True, treat the epoch as if it were ``data_tz`` local time that was
        mislabelled as UTC, and shift it to true UTC. Default False (epoch == UTC).
        Only enable this if you have *verified* the broker stored local time.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    tcol = "time" if "time" in df.columns else df.columns[0]
    dt, unit = _parse_epoch(df[tcol])
    df = df.drop(columns=[tcol])
    df.insert(0, "datetime", dt)
    df = df.dropna(subset=["datetime"])

    if relabel_naive_as_tz and data_tz and data_tz.upper() != "UTC":
        # interpret stamps as data_tz local, convert to true UTC
        local = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["datetime"] = local.dt.tz_localize(data_tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        df = df.dropna(subset=["datetime"])

    df = df.set_index("datetime").sort_index()
    # pandas >=2 preserves datetime resolution; normalise to nanoseconds so that
    # later `index + Timedelta` and merge_asof keys always share one unit.
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    n_dup = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")]

    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    vol_cols = [c for c in ("volume", "tick_volume", "vol", "real_volume") if c in df.columns]
    has_volume = len(vol_cols) > 0
    is_tick = not {"open", "high", "low"}.issubset(df.columns)

    meta = {
        "path": str(path),
        "rows": int(len(df)),
        "epoch_unit": unit,
        "has_volume": has_volume,
        "is_tick": is_tick,
        "duplicates_dropped": n_dup,
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "data_tz": data_tz,
    }
    if not has_volume:
        print(f"[WARNING] {path.name}: no volume column — volume-based features will be auto-skipped.")
    return df, meta


def discover_raw_files(raw_dir) -> dict:
    """Map every recognisable bar CSV under ``raw_dir`` to its timeframe.
    Tick files are reported under ``TICK_*`` keys and skipped by the core."""
    raw = Path(raw_dir)
    found: dict[str, Path] = {}
    if not raw.exists():
        return found
    for p in sorted(raw.glob("*.csv")):
        tf = detect_timeframe_from_name(p.name)
        if tf and not tf.startswith("TICK_"):
            found.setdefault(tf, p)
    return found


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Fallback: build a higher timeframe by resampling when a native file
    is missing. Prefer the native per-TF file whenever available."""
    rule = TF_RESAMPLE.get(tf)
    if rule is None:
        raise ValueError(f"unknown timeframe '{tf}' for resample")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    cols = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(cols).dropna(how="any")
    return out
