"""
mtf.py — leakage-safe multi-timeframe alignment.

THE RULE (López de Prado / common-sense causality): a primary bar that opens at
time ``t`` may only use a higher-timeframe bar that has *fully closed* by ``t``.

We implement this with ``merge_asof`` on the higher-TF bar's **close time**
(= open time + bar duration), matching backward onto the primary bar's open
time. The higher bar's *open* timestamp is carried through so the notebook can
assert ``source_open <= primary_open`` (no look-ahead). This is the concrete
form of the classic ``.shift(1)`` rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io_data


def align_higher_tf_without_leakage(
    primary_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    higher_tf: str,
    feature_cols: list[str] | None = None,
    prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Merge higher-TF features onto the primary index without look-ahead.

    Parameters
    ----------
    primary_df : DataFrame indexed by primary bar OPEN time (UTC).
    higher_df  : DataFrame indexed by higher-TF bar OPEN time (UTC).
    higher_tf  : canonical timeframe of ``higher_df`` (e.g. "H4", "D1").
    feature_cols : columns of ``higher_df`` to merge (default: all).
    prefix : column prefix for merged features (default: ``f"{higher_tf}_"``).

    Returns
    -------
    merged : DataFrame on the primary index with prefixed HTF feature columns.
    source_open : Series (primary index) of the HTF source bar's OPEN time,
                  for the leakage assertion.
    """
    prefix = prefix if prefix is not None else f"{higher_tf}_"
    cols = feature_cols if feature_cols is not None else list(higher_df.columns)

    dur = io_data.tf_timedelta(higher_tf)
    # close time = open time + one bar duration => the bar is *known* only at/after this instant.
    # Force a single datetime unit (ns) so merge_asof keys never mismatch under pandas 3.x.
    hopen = pd.DatetimeIndex(higher_df.index).as_unit("ns")
    hclose = (hopen + dur).as_unit("ns")
    popen = pd.DatetimeIndex(primary_df.index).as_unit("ns")
    work = pd.DataFrame({"htf_open": hopen, "htf_close_time": hclose})
    for c in cols:
        work[c] = higher_df[c].to_numpy()
    work = work.sort_values("htf_close_time").reset_index(drop=True)

    left = pd.DataFrame({"prim_open": popen})  # primary index is already sorted ascending

    merged = pd.merge_asof(
        left, work,
        left_on="prim_open", right_on="htf_close_time",
        direction="backward",
        allow_exact_matches=True,   # a HTF bar closing exactly at the primary open IS available
    )

    source_open = pd.Series(merged["htf_open"].to_numpy(), index=primary_df.index,
                            name=f"{prefix}source_open")
    feat = pd.DataFrame({f"{prefix}{c}": merged[c].to_numpy() for c in cols},
                        index=primary_df.index)
    return feat, source_open


def attach_multi_timeframe(
    primary_df: pd.DataFrame,
    higher_frames: dict[str, pd.DataFrame],
    feature_builder,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build features on each higher timeframe, then leakage-safe-merge them
    onto the primary index.

    Parameters
    ----------
    primary_df : primary OHLC frame.
    higher_frames : {tf_name: ohlc_df} for each higher / lower context timeframe.
    feature_builder : callable(df, cfg) -> (features_df, meta) (features.build_features).
    """
    cfg = cfg or {}
    merged_blocks = []
    source_map: dict[str, pd.Series] = {}
    meta = {"timeframes": {}}
    for tf, hdf in higher_frames.items():
        hfeat, hmeta = feature_builder(hdf, cfg)
        # keep a compact, robust subset of HTF features to avoid blowing up width
        keep = [c for c in hfeat.columns if any(
            c.startswith(p) for p in (
                "ret_log_1", "ret_std_", "px_vs_ema_", "ema_slope_", "adx_",
                "rsi_", "macd_hist", "atr_pct", "bb_width", "realized_vol_",
                "px_vs_decycler", "ehlers_fisher", "hurst_",
            ))]
        keep = keep or list(hfeat.columns)
        feat, src = align_higher_tf_without_leakage(primary_df, hfeat[keep], tf)
        merged_blocks.append(feat)
        source_map[tf] = src
        meta["timeframes"][tf] = {"n_features": len(keep), "rows": int(len(hdf))}
    # return ONLY the leakage-safe MTF feature blocks (indexed on the primary bars);
    # the notebook concatenates these with the primary-TF features.
    mtf_features = pd.concat(merged_blocks, axis=1) if merged_blocks else pd.DataFrame(index=primary_df.index)
    return mtf_features, {"meta": meta, "source_map": source_map}
