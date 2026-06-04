"""
labeling.py — triple-barrier labels + AFML sample weights (López de Prado Ch. 3-4).

Triple barrier (ATR-scaled):
    upper  = close_t + ATR_t * tp_mult     (take-profit, label +1)
    lower  = close_t - ATR_t * sl_mult     (stop-loss,   label -1)
    vertical = t + max_holding bars        (timeout,     label  0)

The first barrier touched wins. Highs/lows of FUTURE bars are read ONLY to
build the label — they never become features.

Sample weights follow AFML Ch.4: average label uniqueness (1 / concurrency)
× |event_return| × time-decay (recency) × class-balance. Sequential
bootstrapping is left as a clearly-marked optional TODO.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


def apply_triple_barrier(
    df: pd.DataFrame,
    atr: pd.Series,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 24,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels for every bar.

    Returns a frame indexed like ``df`` with columns:
        label (-1/0/+1), event_end_time, event_end_idx, event_return,
        touched_barrier ('tp'/'sl'/'vertical'), side_label (sign of return),
        outcome_dir (directional outcome used by meta-labelling).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    a = atr.reindex(df.index).to_numpy(dtype=float)
    idx = df.index
    n = len(df)

    label = np.zeros(n, dtype=float)
    end_idx = np.full(n, -1, dtype=int)
    ev_ret = np.full(n, np.nan)
    touched = np.array(["vertical"] * n, dtype=object)

    for i in tqdm(range(n), desc="triple-barrier", leave=False):
        if not np.isfinite(a[i]) or not np.isfinite(close[i]) or a[i] <= 0:
            end_idx[i] = i
            ev_ret[i] = 0.0
            touched[i] = "invalid"
            label[i] = 0
            continue
        upper = close[i] + a[i] * tp_mult
        lower = close[i] - a[i] * sl_mult
        last = min(i + max_holding, n - 1)
        hit = None
        for j in range(i + 1, last + 1):
            hi_hit = high[j] >= upper
            lo_hit = low[j] <= lower
            if hi_hit and lo_hit:
                # both barriers inside one bar: assume the adverse (SL) touched
                # first — conservative / pessimistic.
                hit = ("sl", j)
                break
            if hi_hit:
                hit = ("tp", j)
                break
            if lo_hit:
                hit = ("sl", j)
                break
        if hit is None:
            j = last
            touched[i] = "vertical"
            label[i] = 0
        else:
            kind, j = hit
            touched[i] = kind
            label[i] = 1.0 if kind == "tp" else -1.0
        end_idx[i] = j
        ev_ret[i] = (close[j] - close[i]) / close[i]

    out = pd.DataFrame(index=idx)
    out["label"] = label.astype(int)
    out["event_end_idx"] = end_idx
    out["event_end_time"] = idx[np.clip(end_idx, 0, n - 1)]
    out["event_return"] = ev_ret
    out["touched_barrier"] = touched
    out["side_label"] = np.sign(out["event_return"]).fillna(0).astype(int)
    # directional outcome for meta-labelling: tp=+1, sl=-1, vertical=sign(return)
    outcome = np.where(out["touched_barrier"].values == "tp", 1,
                       np.where(out["touched_barrier"].values == "sl", -1,
                                out["side_label"].values))
    out["outcome_dir"] = outcome.astype(int)
    return out


def average_uniqueness(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """AFML Ch.4: average label uniqueness = mean over the event's life of
    1 / concurrency, where concurrency = number of labels simultaneously live."""
    n = len(index)
    pos = pd.Series(np.arange(n), index=index)
    start_pos = pos.reindex(events.index).to_numpy()
    end_pos = events["event_end_idx"].to_numpy()

    concurrency = np.zeros(n, dtype=float)
    for s, e in zip(start_pos, end_pos):
        if s < 0 or e < 0:
            continue
        concurrency[int(s): int(e) + 1] += 1.0
    concurrency[concurrency == 0] = 1.0

    avg_u = np.zeros(len(events), dtype=float)
    for k, (s, e) in enumerate(zip(start_pos, end_pos)):
        if s < 0 or e < 0 or e < s:
            avg_u[k] = 0.0
            continue
        avg_u[k] = float(np.mean(1.0 / concurrency[int(s): int(e) + 1]))
    return pd.Series(avg_u, index=events.index, name="avg_uniqueness")


def compute_sample_weights(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    time_decay: float = 0.5,
    use_class_balance: bool = True,
) -> pd.DataFrame:
    """weight = avg_uniqueness × |event_return| × time_decay(recency) × class_balance.

    ``time_decay`` in [0,1] is the weight of the OLDEST sample (1 = no decay).
    All weights are strictly positive (a small floor avoids zeros).
    """
    avg_u = average_uniqueness(events, index)

    ret_w = events["event_return"].abs().fillna(0.0)
    ret_w = ret_w / (ret_w.mean() + 1e-12)            # normalise around 1
    base = avg_u * (ret_w + 1e-3)

    # linear time decay over avg-uniqueness-cumsum (recent => closer to 1)
    cum = avg_u.cumsum()
    if cum.iloc[-1] > 0:
        decay = time_decay + (1.0 - time_decay) * (cum / cum.iloc[-1])
    else:
        decay = pd.Series(1.0, index=events.index)
    w = base * decay

    if use_class_balance:
        lab = events["label"]
        counts = lab.value_counts()
        inv = {c: (len(lab) / (len(counts) * n)) for c, n in counts.items()}
        cb = lab.map(inv).fillna(1.0)
        w = w * cb

    w = w.clip(lower=1e-6)
    w = w / w.mean()                                   # mean-1 normalisation
    out = pd.DataFrame(index=events.index)
    out["avg_uniqueness"] = avg_u
    out["sample_weight"] = w
    return out

# TODO (optional, AFML Ch.4): sequential bootstrapping to draw maximally-uncorrelated
# samples using the indicator matrix. Average uniqueness above already de-weights
# overlapping labels; sequential bootstrapping would further improve bagging.
