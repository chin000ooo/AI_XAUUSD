"""
metrics.py — trading performance metrics + Deflated Sharpe Ratio + PBO.

All ratios guard against division-by-zero / zero-variance. The Deflated Sharpe
Ratio (Bailey & López de Prado 2014) corrects an observed Sharpe for the number
of trials, non-normal returns and sample length. PBO (Probability of Backtest
Overfitting) is estimated via CSCV (Bailey et al. 2017).
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _norm_cdf(x: float) -> float:
    if _HAS_SCIPY:
        return float(norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    if _HAS_SCIPY:
        return float(norm.ppf(p))
    # Acklam rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 6240, rf: float = 0.0) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std()
    if len(r) < 2 or sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float((r.mean() - rf) / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float = 6240) -> float:
    r = pd.Series(returns).dropna()
    downside = r[r < 0].std()
    if len(r) < 2 or downside == 0 or not np.isfinite(downside):
        return float("nan")
    return float(r.mean() / downside * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity).dropna()
    if len(e) == 0:
        return float("nan")
    dd = e / e.cummax() - 1.0
    return float(dd.min())


def performance_metrics(equity: pd.Series, trades: pd.DataFrame,
                        periods_per_year: float = 6240,
                        initial_capital: float = 10_000.0) -> dict:
    eq = pd.Series(equity).dropna()
    ret = eq.pct_change().dropna()
    out = {}
    final_eq = float(eq.iloc[-1]) if len(eq) else initial_capital
    out["initial_capital"] = float(initial_capital)
    out["final_equity"] = final_eq
    out["total_return"] = final_eq / initial_capital - 1.0
    # CAGR if the span is meaningful
    if len(eq) > 1 and isinstance(eq.index, pd.DatetimeIndex):
        years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
        out["years"] = float(years)
        out["cagr"] = float((final_eq / initial_capital) ** (1 / years) - 1) if years > 0 and final_eq > 0 else float("nan")
    else:
        out["years"] = float("nan"); out["cagr"] = float("nan")
    out["sharpe"] = sharpe_ratio(ret, periods_per_year)
    out["sortino"] = sortino_ratio(ret, periods_per_year)
    out["max_drawdown"] = max_drawdown(eq)
    out["calmar"] = float(out["cagr"] / abs(out["max_drawdown"])) if (out["max_drawdown"] and out["max_drawdown"] < 0 and np.isfinite(out["cagr"])) else float("nan")

    # trade-based stats
    if trades is not None and len(trades) > 0:
        pnl = trades["net_pnl"]
        wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
        gross_win = float(wins.sum()); gross_loss = float(-losses.sum())
        out["n_trades"] = int(len(trades))
        out["win_rate"] = float((pnl > 0).mean())
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        out["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan")
        out["expectancy"] = float(pnl.mean())
        out["avg_bars_held"] = float(trades["bars_held"].mean())
        out["total_commission"] = float(trades["commission"].sum())
        out["gross_pnl"] = float(trades["gross_pnl"].sum())
        out["net_pnl"] = float(trades["net_pnl"].sum())
    else:
        out.update({"n_trades": 0, "win_rate": float("nan"), "avg_win": 0.0, "avg_loss": 0.0,
                    "profit_factor": float("nan"), "expectancy": float("nan"),
                    "avg_bars_held": float("nan"), "total_commission": 0.0,
                    "gross_pnl": 0.0, "net_pnl": 0.0})
    # exposure
    out["return_skew"] = float(ret.skew()) if len(ret) > 2 else float("nan")
    out["return_kurtosis"] = float(ret.kurt()) if len(ret) > 3 else float("nan")
    return out


def deflated_sharpe_ratio(observed_sr: float, n_obs: int, skew: float, kurt: float,
                          n_trials: int, var_trials_sr: float) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Parameters
    ----------
    observed_sr : the strategy's *non-annualised* Sharpe (per observation).
    n_obs : number of return observations T.
    skew, kurt : skew and (non-excess) kurtosis of the returns.
    n_trials : number of independent strategy configurations tried (N).
    var_trials_sr : variance of the trial Sharpe ratios.

    Returns dict with the DSR (probability the true SR > benchmark) and the
    expected-maximum benchmark SR*.
    """
    if not np.isfinite(observed_sr) or n_obs < 3 or n_trials < 1:
        return {"dsr": float("nan"), "sr_benchmark": float("nan"),
                "note": "insufficient data for DSR"}
    emc = 0.5772156649015329  # Euler-Mascheroni
    sd_trials = math.sqrt(max(var_trials_sr, 1e-12))
    if n_trials >= 2:
        z1 = _norm_ppf(1 - 1.0 / n_trials)
        z2 = _norm_ppf(1 - 1.0 / (n_trials * math.e))
        sr_star = sd_trials * ((1 - emc) * z1 + emc * z2)
    else:
        sr_star = 0.0
    denom = math.sqrt(max(1 - skew * observed_sr + (kurt - 1) / 4.0 * observed_sr ** 2, 1e-9))
    dsr = _norm_cdf((observed_sr - sr_star) * math.sqrt(n_obs - 1) / denom)
    return {"dsr": float(dsr), "sr_benchmark": float(sr_star),
            "observed_sr": float(observed_sr), "n_trials": int(n_trials),
            "note": "DSR = P(true SR > benchmark) after deflating for N trials & non-normality"}


def probability_of_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 8) -> dict:
    """PBO via CSCV (Combinatorially-Symmetric Cross-Validation).

    ``returns_matrix`` : T x N array of per-period returns for N candidate
    configurations. We split the T rows into ``n_splits`` contiguous blocks,
    take every half-size combination as the in-sample (IS) set, pick the best
    config IS, and measure its OUT-of-sample rank. PBO = fraction of splits
    where the IS-best config ranks below the OOS median (logit <= 0).
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": float("nan"), "n_configs": int(M.shape[1] if M.ndim == 2 else 0),
                "note": "need >=2 configs for PBO"}
    T, N = M.shape
    S = n_splits if n_splits % 2 == 0 else n_splits - 1
    S = max(2, min(S, T))
    bounds = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(S)]

    def _sr(x):
        sd = x.std()
        return x.mean() / sd if sd > 0 else 0.0

    logits = []
    for combo in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in combo])
        if len(is_rows) == 0 or len(oos_rows) == 0:
            continue
        is_perf = np.array([_sr(M[is_rows, j]) for j in range(N)])
        oos_perf = np.array([_sr(M[oos_rows, j]) for j in range(N)])
        n_star = int(np.argmax(is_perf))
        # rank of the IS-best config among OOS performances
        rank = (oos_perf <= oos_perf[n_star]).sum()
        w = rank / (N + 1.0)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    if not logits:
        return {"pbo": float("nan"), "n_configs": int(N), "note": "no valid CSCV splits"}
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())
    return {"pbo": pbo, "n_configs": int(N), "n_splits": int(S),
            "median_logit": float(np.median(logits)),
            "note": "PBO = P(IS-best config underperforms OOS median); lower is better"}


def write_metrics_summary(path, metrics: dict, dsr: dict | None = None, pbo: dict | None = None,
                          extra_lines: list[str] | None = None) -> None:
    lines = ["=" * 60, "BACKTEST METRICS SUMMARY", "=" * 60]
    for k in ["initial_capital", "final_equity", "total_return", "cagr", "sharpe",
              "sortino", "calmar", "max_drawdown", "profit_factor", "win_rate",
              "expectancy", "n_trades", "avg_bars_held", "gross_pnl", "net_pnl",
              "total_commission"]:
        if k in metrics:
            v = metrics[k]
            sv = f"{v:,.4f}" if isinstance(v, float) else str(v)
            lines.append(f"  {k:<18}: {sv}")
    if dsr:
        lines += ["-" * 60, "DEFLATED SHARPE RATIO",
                  f"  dsr            : {dsr.get('dsr'):.4f}" if np.isfinite(dsr.get('dsr', float('nan'))) else "  dsr            : n/a",
                  f"  sr_benchmark   : {dsr.get('sr_benchmark')}", f"  {dsr.get('note','')}"]
    if pbo:
        lines += ["-" * 60, "PROBABILITY OF BACKTEST OVERFITTING",
                  f"  pbo            : {pbo.get('pbo')}", f"  n_configs      : {pbo.get('n_configs')}",
                  f"  {pbo.get('note','')}"]
    if extra_lines:
        lines += ["-" * 60] + extra_lines
    lines.append("=" * 60)
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[OUTPUT SAVED] metrics summary -> {path}")
