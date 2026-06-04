"""
backtest.py — confidence-based sizing + event-driven, cost-aware backtest.

Hard rules (anti-leakage):
  * A signal computed at the CLOSE of bar t is executed at the OPEN of bar t+1.
  * Exits: take-profit, stop-loss, max-holding timeout, optional opposite signal.
  * Costs: half-spread + slippage on BOTH fills, commission per lot per side,
    with optional session-dependent spread (wider Asian session for XAUUSD).

Contract maths for XAUUSD: contract_size = 100 oz/lot, so a $1 move on 1.0 lot
= $100. Position sizing is risk-based off the ATR stop distance, capped by
max exposure, with an optional conservative fractional-Kelly overlay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x


@dataclass
class CostConfig:
    spread_points: float = 20.0          # in points (point = `point` below)
    slippage_points: float = 5.0
    commission_per_lot: float = 3.5      # USD per lot per side (round turn = 2x)
    point: float = 0.01                  # price value of 1 "point" for XAUUSD
    contract_size: float = 100.0         # oz per lot
    asian_spread_mult: float = 1.5       # session-dependent widening
    asian_hours: tuple = (0, 8)          # in data_tz clock


@dataclass
class SizingConfig:
    initial_capital: float = 10_000.0
    max_risk_per_trade: float = 0.01     # 1% risk per trade
    min_primary_confidence: float = 0.50
    meta_threshold: float = 0.55
    max_exposure_lots: float = 5.0
    lot_step: float = 0.01
    min_lot: float = 0.01
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    max_holding: int = 24
    use_kelly: bool = False
    kelly_fraction: float = 0.25         # quarter-Kelly (conservative)
    turbulent_regime: int = 2            # regime index to de-risk
    turbulent_size_mult: float = 0.0     # 0 = flat in turbulent regime


def position_size(confidence: float, meta_prob: float, atr_value: float, price: float,
                  equity: float, scfg: SizingConfig, regime: int | None = None,
                  contract_size: float = 100.0) -> float:
    """Return lots for a candidate trade (0 if filtered out).

    Risk-based: lots = (equity * risk_frac) / (sl_distance * contract_size),
    where sl_distance = sl_mult * ATR in price units. ``contract_size`` is the
    oz/lot (100 for XAUUSD) so $1 move on 1.0 lot = $contract_size.
    """
    if not np.isfinite(atr_value) or atr_value <= 0 or price <= 0:
        return 0.0
    if confidence < scfg.min_primary_confidence or meta_prob < scfg.meta_threshold:
        return 0.0

    if scfg.use_kelly:
        payoff = scfg.tp_mult / max(scfg.sl_mult, 1e-9)
        p = float(np.clip(meta_prob, 0, 1))
        kelly = max(0.0, (payoff * p - (1 - p)) / payoff)
        eff_risk = min(scfg.max_risk_per_trade, scfg.kelly_fraction * kelly)
    else:
        # scale risk between 50% and 100% of the cap by meta confidence above threshold
        conf_scale = np.clip((meta_prob - scfg.meta_threshold) / max(1 - scfg.meta_threshold, 1e-9), 0, 1)
        eff_risk = scfg.max_risk_per_trade * (0.5 + 0.5 * conf_scale)

    risk_amount = equity * eff_risk
    stop_distance = scfg.sl_mult * atr_value                      # price units
    lots = risk_amount / (stop_distance * contract_size)
    # regime de-risking (risk control only — never flips side)
    if regime is not None and regime == scfg.turbulent_regime:
        lots *= scfg.turbulent_size_mult
    lots = min(lots, scfg.max_exposure_lots)
    lots = np.floor(lots / scfg.lot_step) * scfg.lot_step
    if lots < scfg.min_lot:
        return 0.0
    return float(lots)


def _session_spread(hour: int, ccfg: CostConfig) -> float:
    a, b = ccfg.asian_hours
    mult = ccfg.asian_spread_mult if (a <= hour < b) else 1.0
    return ccfg.spread_points * mult


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame, scfg: SizingConfig,
                 ccfg: CostConfig, data_tz: str = "UTC") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-driven backtest.

    ``df``      : primary OHLC frame with an ``atr`` column (UTC index).
    ``signals`` : same index, columns = signal(+1/-1/0), confidence, meta_prob,
                  size_lots (pre-computed), regime (optional).

    Returns ``(trades_df, equity_df)``.
    """
    idx = df.index
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)

    sig = signals["signal"].reindex(idx).fillna(0).to_numpy(int)
    size = signals["size_lots"].reindex(idx).fillna(0.0).to_numpy(float)

    local = idx.tz_convert(data_tz) if (idx.tz is not None and data_tz.upper() != "UTC") else idx
    hours = (local.hour if local.tz is not None else local.tz_localize("UTC").hour)
    hours = np.asarray(hours)

    point = ccfg.point
    csize = ccfg.contract_size

    realized = scfg.initial_capital
    equity_curve = np.full(n, np.nan)
    pos_side_arr = np.zeros(n, dtype=int)
    pos = None  # dict: side, lots, entry_price, entry_i, entry_time, sl, tp
    trades = []

    def _cost_price(hour):
        half_spread = 0.5 * _session_spread(hour, ccfg) * point
        slip = ccfg.slippage_points * point
        return half_spread + slip

    for i in tqdm(range(n), desc="backtest", leave=False):
        # ---------- manage an open position on bar i ----------
        if pos is not None:
            side = pos["side"]
            exit_price = None
            reason = None
            sig_prev = sig[i - 1] if i > 0 else 0

            # (a) opposite signal -> exit at this bar's open
            if sig_prev == -side and sig_prev != 0:
                exit_price, reason = o[i], "opposite"
            else:
                # (b) intrabar SL/TP using this bar's range (SL checked first = pessimistic)
                if side == 1:
                    if l[i] <= pos["sl"]:
                        exit_price, reason = pos["sl"], "sl"
                    elif h[i] >= pos["tp"]:
                        exit_price, reason = pos["tp"], "tp"
                else:
                    if h[i] >= pos["sl"]:
                        exit_price, reason = pos["sl"], "sl"
                    elif l[i] <= pos["tp"]:
                        exit_price, reason = pos["tp"], "tp"
                # (c) max-holding timeout -> exit at close
                if exit_price is None and (i - pos["entry_i"]) >= scfg.max_holding:
                    exit_price, reason = c[i], "vertical"

            if exit_price is not None:
                cp = _cost_price(hours[i])
                entry_fill = pos["entry_fill"]
                exit_fill = exit_price - side * cp
                lots = pos["lots"]
                gross = side * (exit_price - pos["entry_price"]) * lots * csize
                commission = ccfg.commission_per_lot * lots * 2.0
                net = side * (exit_fill - entry_fill) * lots * csize - commission
                realized += net
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": idx[i],
                    "side": side, "lots": lots,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "entry_fill": entry_fill, "exit_fill": exit_fill,
                    "bars_held": i - pos["entry_i"], "reason": reason,
                    "gross_pnl": gross, "commission": commission, "net_pnl": net,
                    "equity_after": realized,
                })
                pos = None

        # ---------- consider a new entry on bar i (signal from bar i-1) ----------
        # never open on the final bar — there is no future bar to manage/exit into,
        # which would otherwise create a zero-duration (exit_time == entry_time) trade.
        if pos is None and 0 < i < n - 1:
            s_prev = sig[i - 1]
            lots = size[i - 1]
            if s_prev != 0 and lots > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                side = int(np.sign(s_prev))
                entry_price = o[i]
                cp = _cost_price(hours[i])
                entry_fill = entry_price + side * cp
                sl = entry_price - side * scfg.sl_mult * atr[i - 1]
                tp = entry_price + side * scfg.tp_mult * atr[i - 1]
                pos = {"side": side, "lots": lots, "entry_price": entry_price,
                       "entry_fill": entry_fill, "entry_i": i, "entry_time": idx[i],
                       "sl": sl, "tp": tp}

        # ---------- mark-to-market equity ----------
        if pos is not None:
            unreal = pos["side"] * (c[i] - pos["entry_price"]) * pos["lots"] * csize
            equity_curve[i] = realized + unreal
            pos_side_arr[i] = pos["side"]
        else:
            equity_curve[i] = realized
            pos_side_arr[i] = 0

    # force-close any open position at the last close
    if pos is not None:
        side = pos["side"]
        exit_price = c[-1]
        cp = _cost_price(hours[-1])
        exit_fill = exit_price - side * cp
        lots = pos["lots"]
        gross = side * (exit_price - pos["entry_price"]) * lots * csize
        commission = ccfg.commission_per_lot * lots * 2.0
        net = side * (exit_fill - pos["entry_fill"]) * lots * csize - commission
        realized += net
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": idx[-1], "side": side, "lots": lots,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "entry_fill": pos["entry_fill"], "exit_fill": exit_fill,
            "bars_held": (n - 1) - pos["entry_i"], "reason": "eod_close",
            "gross_pnl": gross, "commission": commission, "net_pnl": net, "equity_after": realized,
        })
        equity_curve[-1] = realized

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity_curve, index=idx).ffill().fillna(scfg.initial_capital)
    running_max = eq.cummax()
    equity_df = pd.DataFrame({
        "equity": eq,
        "drawdown": eq / running_max - 1.0,
        "position_side": pd.Series(pos_side_arr, index=idx),
    })
    return trades_df, equity_df
