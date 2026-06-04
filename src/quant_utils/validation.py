"""
validation.py — PASS / FAIL / WARN checks collected into a JSON report.

Every check returns a ``(status, message)`` tuple where status is one of
"PASS" / "FAIL" / "WARN". A ``ValidationReport`` collects them, prints a
checklist, and serialises to ``reports/validation_report.json``.

The realistic-ceiling smoke test (``validate_smoke_metric``) is itself a
leakage detector: implausibly high OOS accuracy/AUC is treated as a FAIL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class ValidationReport:
    """Collects check results and renders a checklist + JSON."""

    def __init__(self, name: str = ""):
        self.name = name
        self.checks: list[dict] = []

    def add(self, check_name: str, status: str, message) -> str:
        self.checks.append({"check": check_name, "status": status, "message": str(message)})
        icon = {PASS: "[VALIDATION PASS]", FAIL: "[VALIDATION FAIL]", WARN: "[WARNING]"}.get(status, "[?]")
        print(f"{icon} {check_name}: {message}")
        return status

    def record(self, check_name: str, result) -> str:
        """``result`` is a (status, message) tuple from a check_* function."""
        status, message = result
        return self.add(check_name, status, message)

    def summary(self) -> dict:
        n = len(self.checks)
        p = sum(c["status"] == PASS for c in self.checks)
        f = sum(c["status"] == FAIL for c in self.checks)
        w = sum(c["status"] == WARN for c in self.checks)
        return {"total": n, "pass": p, "fail": f, "warn": w}

    def has_failures(self) -> bool:
        return any(c["status"] == FAIL for c in self.checks)

    def print_checklist(self) -> dict:
        print("\n" + "=" * 64)
        print(f"VALIDATION CHECKLIST{(' — ' + self.name) if self.name else ''}")
        print("=" * 64)
        for c in self.checks:
            print(f"  [{c['status']:<4}] {c['check']}: {c['message']}")
        s = self.summary()
        print("-" * 64)
        verdict = "ALL CLEAR" if s["fail"] == 0 else f"{s['fail']} FAILURE(S) — INVESTIGATE"
        print(f"  TOTAL={s['total']}  PASS={s['pass']}  FAIL={s['fail']}  WARN={s['warn']}  =>  {verdict}")
        print("=" * 64)
        return s

    def to_dict(self) -> dict:
        return {"name": self.name, "checks": self.checks, "summary": self.summary()}


def save_validation_report(report, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, ValidationReport) else report
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[OUTPUT SAVED] validation report -> {path}")
    return path


# --------------------------------------------------------------------------
# individual checks — each returns (status, message)
# --------------------------------------------------------------------------
def check_file_exists(path) -> tuple[str, str]:
    return (PASS, f"exists: {path}") if Path(path).exists() else (FAIL, f"missing: {path}")


def validate_ohlcv(df: pd.DataFrame) -> tuple[str, str]:
    cols = {"open", "high", "low", "close"}
    if not cols.issubset(df.columns):
        return (FAIL, f"missing OHLC columns; have {list(df.columns)}")
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    bad_hl = int((h < l).sum())
    bad_ho = int((h < df[["open", "close"]].max(axis=1) - 1e-9).sum())
    bad_lo = int((l > df[["open", "close"]].min(axis=1) + 1e-9).sum())
    non_numeric = [c_ for c_ in ["open", "high", "low", "close"] if not np.issubdtype(df[c_].dtype, np.number)]
    if non_numeric:
        return (FAIL, f"non-numeric OHLC columns: {non_numeric}")
    if bad_hl or bad_ho or bad_lo:
        return (FAIL, f"invalid OHLC rows: high<low={bad_hl}, high<max(o,c)={bad_ho}, low>min(o,c)={bad_lo}")
    return (PASS, f"OHLC consistent across {len(df):,} rows")


def validate_no_duplicate_timestamp(df: pd.DataFrame) -> tuple[str, str]:
    d = int(df.index.duplicated().sum())
    return (PASS, "no duplicate timestamps") if d == 0 else (FAIL, f"{d} duplicate timestamps")


def validate_no_nan_inf(df: pd.DataFrame, cols=None) -> tuple[str, str]:
    sub = df[cols] if cols else df.select_dtypes("number")
    if sub.shape[1] == 0:
        return (WARN, "no numeric columns to check")
    vals = sub.to_numpy(dtype=float)
    n_nan = int(np.isnan(vals).sum())
    n_inf = int(np.isinf(vals).sum())
    if n_nan == 0 and n_inf == 0:
        return (PASS, "no NaN / inf in numeric columns")
    return (WARN, f"NaN={n_nan}, inf={n_inf} (handle before modelling)")


def validate_time_order(df: pd.DataFrame) -> tuple[str, str]:
    if df.index.is_monotonic_increasing:
        return (PASS, "index sorted ascending (no shuffle)")
    return (FAIL, "index NOT monotonic — time order violated")


def validate_feature_leakage_mtf(source_ts, primary_ts) -> tuple[str, str]:
    """source_ts = higher-TF *open* timestamp merged onto each primary bar.
    primary_ts = primary bar open timestamp. Leakage iff source > primary."""
    src = pd.to_datetime(pd.Series(list(source_ts)), utc=True, errors="coerce")
    pri = pd.to_datetime(pd.Series(list(primary_ts)), utc=True, errors="coerce")
    mask = src.notna() & pri.notna()
    if mask.sum() == 0:
        return (WARN, "no overlapping timestamps to compare")
    bad = int((src[mask].values > pri[mask].values).sum())
    if bad == 0:
        return (PASS, f"all {int(mask.sum()):,} merged HTF source bars <= primary bar time (no look-ahead)")
    return (FAIL, f"{bad} HTF features reference a FUTURE source bar — LEAKAGE")


def validate_labels(labels, allowed=(-1, 0, 1)) -> tuple[str, str]:
    u = set(pd.unique(pd.Series(labels).dropna()))
    extra = u - set(allowed)
    if extra:
        return (FAIL, f"labels contain disallowed values: {extra}")
    return (PASS, f"labels subset of {allowed}; present={sorted(u)}")


def validate_event_end_after_start(start_ts, end_ts) -> tuple[str, str]:
    s = pd.to_datetime(pd.Series(list(start_ts)), utc=True, errors="coerce")
    e = pd.to_datetime(pd.Series(list(end_ts)), utc=True, errors="coerce")
    mask = s.notna() & e.notna()
    bad = int((e[mask].values < s[mask].values).sum())
    if bad == 0:
        return (PASS, "every event_end_time >= event start")
    return (FAIL, f"{bad} events end before they start")


def validate_model_outputs(proba) -> tuple[str, str]:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        return (FAIL, f"proba not 2-D (shape {proba.shape})")
    if np.any(proba < -1e-6) or np.any(proba > 1 + 1e-6):
        return (FAIL, "probabilities outside [0,1]")
    s = proba.sum(axis=1)
    if not np.allclose(s, 1.0, atol=1e-3):
        return (WARN, f"proba rows not ~sum-1 (max dev {np.max(np.abs(s - 1)):.3g})")
    return (PASS, f"proba shape {proba.shape}, rows sum~1, all in [0,1]")


def validate_no_label_in_features(feature_cols, banned=("label", "event_return", "event_end_time",
                                                        "touched_barrier", "side_label", "outcome_dir",
                                                        "meta_y", "sample_weight")) -> tuple[str, str]:
    leaked = sorted(set(feature_cols) & set(banned))
    if leaked:
        return (FAIL, f"future/label columns present in feature set: {leaked}")
    return (PASS, "no label / future columns in feature set")


def validate_backtest_trades(trades: pd.DataFrame) -> tuple[str, str]:
    if trades is None or len(trades) == 0:
        return (WARN, "no trades generated (edge too weak under current thresholds)")
    bad_time = int((pd.to_datetime(trades["exit_time"]) <= pd.to_datetime(trades["entry_time"])).sum())
    if bad_time > 0:
        return (FAIL, f"{bad_time} trades with exit_time <= entry_time")
    if trades[["gross_pnl", "net_pnl"]].isna().any().any():
        return (FAIL, "NaN in trade pnl")
    cost_ok = bool((trades["net_pnl"] <= trades["gross_pnl"] + 1e-6).all())
    if not cost_ok:
        return (FAIL, "net_pnl exceeds gross_pnl — costs not deducted")
    return (PASS, f"{len(trades)} trades; exit>entry; costs deducted; pnl finite")


def validate_smoke_metric(metric_value, metric_name="accuracy", warn_threshold=0.65) -> tuple[str, str]:
    """The §0 realistic-ceiling smoke test. OOS metric above the threshold is
    treated as a likely LEAKAGE BUG (FAIL), not as success."""
    if metric_value is None or (isinstance(metric_value, float) and math.isnan(metric_value)):
        return (WARN, f"{metric_name} not computable")
    if metric_value > warn_threshold:
        return (FAIL, f"{metric_name}={metric_value:.3f} > {warn_threshold} — TOO GOOD; "
                      f"treat as leakage and investigate BEFORE trusting")
    return (PASS, f"{metric_name}={metric_value:.3f} within realistic ceiling (<= {warn_threshold})")
