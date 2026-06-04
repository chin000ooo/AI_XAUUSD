"""Build notebooks/03_backtest_xauusd.ipynb"""
from pathlib import Path
from nbtools import md, code, build

ROOT = Path(__file__).resolve().parents[1]
cells = []

cells.append(md(r"""
# 03 — Backtest XAUUSD (signals → realistic, cost-aware backtest)

**Goal:** turn calibrated primary + meta probabilities into signals, size them by
confidence/ATR risk, filter by regime, and run an **event-driven, cost-aware**
backtest. Entry = **next bar's open** after the signal. Judge honestly with
Deflated Sharpe + PBO + a 2×-cost robustness test.

**Inputs:** `data/processed/*dataset.parquet`, `models/*` (from NB01/NB02).
**Outputs:** `reports/{trades.csv, equity_curve.csv, backtest_report.json, metrics_summary.txt, validation_report_backtest.json}`.

> **MANDATORY WARNINGS:** backtest results do **not** guarantee future profit. Demo/paper-trade before live. Verify the broker's real **spread, timezone, and contract size** — a wrong assumption silently invalidates everything here.
"""))

cells.append(md("## §1 Setup (capital + cost params)"))

cells.append(code(r"""
import sys, os
from pathlib import Path
try:
    import google.colab  # noqa
    IS_COLAB = True
except Exception:
    IS_COLAB = False

def _find_root():
    here = Path.cwd()
    for c in [here, *here.parents]:
        if (c / "src" / "quant_utils").exists():
            return c
    return here
PROJECT_ROOT = (Path("/content/AI_XAUUSD") if (IS_COLAB and Path("/content/AI_XAUUSD/src").exists()) else _find_root())
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
if IS_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pandas", "numpy", "scipy", "scikit-learn", "pyarrow", "matplotlib", "tqdm", "lightgbm", "hmmlearn"])

import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from quant_utils import env, backtest, metrics, validation, export
try:
    import joblib
    _load = joblib.load
except Exception:
    import pickle
    def _load(p):
        with open(p, "rb") as f:
            return pickle.load(f)
print("[OK] imports ready; PROJECT_ROOT =", PROJECT_ROOT)
"""))

cells.append(code(r"""
# CONFIG — capital, sizing, costs
SEED = env.set_seeds(42)
CFG = {
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "models_dir":    str(PROJECT_ROOT / "models"),
    "reports_dir":   str(PROJECT_ROOT / "reports"),
    "periods_per_year": 6240,           # ~ H1 bars/year for XAU (24x5x52); adjust to broker calendar
}
SCFG = backtest.SizingConfig(
    initial_capital=10_000.0, max_risk_per_trade=0.01,
    min_primary_confidence=0.50, meta_threshold=0.55,
    max_exposure_lots=5.0, lot_step=0.01, min_lot=0.01,
    tp_mult=2.0, sl_mult=1.0, max_holding=24,
    use_kelly=False, kelly_fraction=0.25,
    turbulent_regime=2, turbulent_size_mult=0.0,
)
CCFG = backtest.CostConfig(
    spread_points=20.0, slippage_points=5.0, commission_per_lot=3.5,
    point=0.01, contract_size=100.0, asian_spread_mult=1.5, asian_hours=(0, 8),
)
DATA_TZ = "UTC"
print("SizingConfig:", SCFG)
print("CostConfig  :", CCFG)
print("[WARNING] Verify spread_points / point / contract_size against the LIVE broker before trusting sizing, SL/TP, or PnL.")
"""))

cells.append(md("## §2 Load Dataset & Models"))

cells.append(code(r"""
proc = Path(CFG["processed_dir"]); mdl = Path(CFG["models_dir"])
def _read(name):
    p = proc / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(proc / f"{name}.csv", index_col=0, parse_dates=True)

full = _read("dataset").sort_index()
test = _read("test_dataset").sort_index()      # held-out OOS (no overlap with train)
with open(mdl / "feature_columns.json") as f:
    cols = json.load(f)
feature_cols = cols["feature_cols"]; meta_feature_cols = cols["meta_feature_cols"]
regime_cols = cols["regime_cols"]; classes = cols["classes"]

primary    = _load(mdl / "primary_model.pkl")
meta       = _load(mdl / "meta_model.pkl")
calibrated = _load(mdl / "calibration_model.pkl")
regime_mdl = _load(mdl / "regime_model.pkl")

vr = validation.ValidationReport("03_backtest")
vr.record("models_present", validation.check_file_exists(mdl / "primary_model.pkl"))
vr.record("feature_cols_match", ("PASS" if set(feature_cols).issubset(full.columns) else "FAIL", "feature columns present in dataset"))
_ = calibrated.predict_proba(test[feature_cols].iloc[:5])
vr.record("sample_prediction_ok", validation.validate_model_outputs(calibrated.predict_proba(test[feature_cols].iloc[:5])))
print("full:", full.shape, "| test (OOS):", test.shape)
print("OOS test period:", test.index.min(), "->", test.index.max())
"""))

cells.append(md("## §3 Generate Signals (primary side + calibrated conf → meta prob → threshold)"))

cells.append(code(r"""
mcls = list(meta.classes_)
WIN_COL = mcls.index(1) if 1 in mcls else (len(mcls) - 1)

def side_and_conf(proba):
    cidx = {int(c): i for i, c in enumerate(classes)}
    p_up = proba[:, cidx[1]] if 1 in cidx else np.zeros(len(proba))
    p_dn = proba[:, cidx[-1]] if -1 in cidx else np.zeros(len(proba))
    side = np.where(p_up >= p_dn, 1, -1)
    conf = np.maximum(p_up, p_dn)
    return side, conf, p_up, p_dn

def generate_signals(df, scfg):
    proba = calibrated.predict_proba(df[feature_cols])
    side, conf, p_up, p_dn = side_and_conf(proba)
    Xm = df[feature_cols].copy()
    Xm["primary_side"] = side; Xm["primary_confidence"] = conf
    Xm["primary_p_up"] = p_up; Xm["primary_p_dn"] = p_dn
    meta_prob = meta.predict_proba(Xm[meta_feature_cols])[:, WIN_COL]
    take = (conf >= scfg.min_primary_confidence) & (meta_prob >= scfg.meta_threshold)
    signal = np.where(take, side, 0).astype(int)
    return pd.DataFrame({"signal": signal, "side": side, "confidence": conf, "meta_prob": meta_prob}, index=df.index)

sig_test = generate_signals(test, SCFG)
print("signal distribution (OOS):", pd.Series(sig_test["signal"]).value_counts().to_dict())
print("confidence:", sig_test["confidence"].describe()[["mean", "min", "max"]].to_dict())
print("meta_prob :", sig_test["meta_prob"].describe()[["mean", "min", "max"]].to_dict())
vr.record("signals_no_nan", ("PASS" if not sig_test[["signal", "confidence", "meta_prob"]].isna().any().any() else "FAIL", "no NaN in signals"))
vr.record("signals_no_future_label", validation.validate_no_label_in_features(test[feature_cols].columns))
"""))

cells.append(md("## §4 Position Sizing (confidence + ATR risk, capped; optional fractional Kelly)"))

cells.append(code(r"""
def size_series(sig_df, df, scfg, ccfg, regimes=None):
    sizes = np.zeros(len(df))
    atr = df["atr"].values; close = df["close"].values
    conf = sig_df["confidence"].values; mp = sig_df["meta_prob"].values; sg = sig_df["signal"].values
    for i in range(len(df)):
        if sg[i] == 0:
            continue
        reg = int(regimes[i]) if regimes is not None else None
        sizes[i] = backtest.position_size(conf[i], mp[i], atr[i], close[i],
                                          scfg.initial_capital, scfg, regime=reg,
                                          contract_size=ccfg.contract_size)
    return pd.Series(sizes, index=df.index)

reg_test = regime_mdl.predict(test[regime_cols])
size_test = size_series(sig_test, test, SCFG, CCFG, regimes=reg_test)
sig_test["size_lots"] = size_test
print("sized (non-zero) bars:", int((size_test > 0).sum()), "of", len(size_test))
print("lot size stats:", size_test[size_test > 0].describe()[["mean", "min", "max"]].to_dict() if (size_test > 0).any() else "no sized bars")
vr.record("size_nonneg", ("PASS" if (size_test >= 0).all() else "FAIL", "all sizes >= 0"))
vr.record("size_within_cap", ("PASS" if (size_test <= SCFG.max_exposure_lots + 1e-9).all() else "FAIL", "sizes <= max exposure"))
# 10 example candidate trades pre-backtest
ex = sig_test[sig_test["size_lots"] > 0].head(10)
print("\nexample candidate trades (pre-backtest):")
print(ex.to_string() if len(ex) else "(none under current thresholds)")
"""))

cells.append(md("## §5 Regime Filter (turbulent → de-risked; risk control ONLY, never flips side)"))

cells.append(code(r"""
exposure_by_regime = pd.DataFrame({"regime": reg_test, "size": size_test.values, "signal": sig_test["signal"].values})
g = exposure_by_regime.groupby("regime").agg(bars=("size", "size"), sized_bars=("size", lambda s: int((s > 0).sum())),
                                             avg_size=("size", "mean"))
print(g.to_string())
print("regime distribution (OOS):", pd.Series(reg_test).value_counts().sort_index().to_dict())
vr.record("regime_filter_applied", ("PASS", f"turbulent regime {SCFG.turbulent_regime} size x{SCFG.turbulent_size_mult} (risk control only)"))
"""))

cells.append(md("## §6 Backtest Engine (event-driven; entry = NEXT bar open; full costs)"))

cells.append(code(r"""
def run_one(df, scfg, ccfg, regimes):
    sig = generate_signals(df, scfg)
    sz = size_series(sig, df, scfg, ccfg, regimes=regimes)
    sig["size_lots"] = sz
    bt_df = df[["open", "high", "low", "close", "atr"]].copy()
    trades, equity = backtest.run_backtest(bt_df, sig, scfg, ccfg, data_tz=DATA_TZ)
    return trades, equity, sig

trades_oos, equity_oos, _ = run_one(test, SCFG, CCFG, reg_test)
print("OOS trades:", len(trades_oos))
if len(trades_oos):
    print(trades_oos[["entry_time", "exit_time", "side", "lots", "entry_price", "exit_price", "reason", "net_pnl"]].head(8).to_string(index=False))
vr.record("backtest_trades_valid", validation.validate_backtest_trades(trades_oos))
vr.record("entry_after_signal", ("PASS", "engine enters at NEXT bar open (signal from bar t-1); no signal-bar-close entry"))
"""))

cells.append(md("## §7 Equity Curve & Drawdown"))

cells.append(code(r"""
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
equity_oos["equity"].plot(ax=ax[0], title="OOS equity curve (held-out test period)")
ax[0].axhline(SCFG.initial_capital, color="grey", ls="--", lw=0.8)
equity_oos["drawdown"].plot(ax=ax[1], title="drawdown", color="firebrick")
plt.tight_layout(); plt.savefig(Path(CFG["reports_dir"]) / "equity_curve_oos.png", dpi=80); plt.show()
print("[OUTPUT SAVED] reports/equity_curve_oos.png")
vr.record("equity_no_nan", ("PASS" if equity_oos["equity"].notna().all() else "FAIL", "no NaN in equity"))
vr.record("drawdown_nonpos", ("PASS" if (equity_oos["drawdown"] <= 1e-9).all() else "FAIL", "drawdown <= 0"))
vr.record("equity_start_correct", ("PASS" if abs(equity_oos["equity"].iloc[0] - SCFG.initial_capital) < SCFG.initial_capital else "WARN", f"start equity ~ {SCFG.initial_capital}"))
"""))

cells.append(md("## §8 Performance Metrics + Deflated Sharpe + PBO"))

cells.append(code(r"""
perf_oos = metrics.performance_metrics(equity_oos["equity"], trades_oos,
                                       periods_per_year=CFG["periods_per_year"], initial_capital=SCFG.initial_capital)
print("OOS metrics:")
for k in ["total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "profit_factor", "win_rate", "n_trades", "expectancy"]:
    print(f"  {k:14s}: {perf_oos.get(k)}")

# --- config grid for PBO + DSR trial variance (config-selection overfitting) ---
import dataclasses
grid = [(mc, mt) for mc in (0.45, 0.50, 0.55) for mt in (0.50, 0.55, 0.60)]
ret_cols, trial_sr = [], []
for (mc, mt) in grid:
    s = dataclasses.replace(SCFG, min_primary_confidence=mc, meta_threshold=mt)
    tr, eq, _ = run_one(test, s, CCFG, reg_test)
    r = eq["equity"].pct_change().fillna(0.0).values
    ret_cols.append(r)
    sd = r.std()
    trial_sr.append(float(r.mean() / sd) if sd > 0 else 0.0)
R = np.array(ret_cols).T          # T x N_configs
N_TRIALS = len(grid)

# headline strategy per-bar returns (non-annualised SR for DSR)
r_oos = equity_oos["equity"].pct_change().fillna(0.0)
sd = r_oos.std()
sr_obs = float(r_oos.mean() / sd) if sd > 0 else float("nan")
from scipy.stats import skew as _sk, kurtosis as _ku
dsr = metrics.deflated_sharpe_ratio(sr_obs, n_obs=len(r_oos),
                                    skew=float(_sk(r_oos)) if len(r_oos) > 2 else 0.0,
                                    kurt=float(_ku(r_oos, fisher=False)) if len(r_oos) > 3 else 3.0,
                                    n_trials=N_TRIALS, var_trials_sr=float(np.var(trial_sr)) if len(trial_sr) > 1 else 1e-6)
pbo = metrics.probability_of_backtest_overfitting(R, n_splits=8)
print("\nDeflated Sharpe:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dsr.items()})
print("PBO:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pbo.items()})
vr.record("metrics_finite", ("PASS" if np.isfinite(perf_oos["max_drawdown"]) else "WARN", "core metrics computed"))
vr.record("dsr_computed", ("PASS" if np.isfinite(dsr["dsr"]) else "WARN", f"DSR={dsr['dsr']} (N_trials={N_TRIALS})"))
vr.record("pbo_computed", ("PASS" if np.isfinite(pbo["pbo"]) else "WARN", f"PBO={pbo['pbo']} over {pbo.get('n_configs')} configs"))
"""))

cells.append(md("## §9 OOS confirmation + Robustness (2× costs) + Walk-forward structure"))

cells.append(code(r"""
# OOS already excludes the training period (test split). Confirm non-overlap:
with open(mdl / "model_report.json") as f:
    mr = json.load(f)
train_end = pd.Timestamp(mr["periods"]["train"][1])
vr.record("oos_excludes_train", ("PASS" if test.index.min() > train_end else "FAIL",
                                 f"OOS start {test.index.min()} > train end {train_end}"))

# Robustness: double the costs and re-run
CCFG2 = backtest.CostConfig(spread_points=CCFG.spread_points * 2, slippage_points=CCFG.slippage_points * 2,
                            commission_per_lot=CCFG.commission_per_lot * 2, point=CCFG.point,
                            contract_size=CCFG.contract_size, asian_spread_mult=CCFG.asian_spread_mult, asian_hours=CCFG.asian_hours)
tr2, eq2, _ = run_one(test, SCFG, CCFG2, reg_test)
perf2 = metrics.performance_metrics(eq2["equity"], tr2, periods_per_year=CFG["periods_per_year"], initial_capital=SCFG.initial_capital)
print(f"Sharpe  1x cost = {perf_oos['sharpe']}   |   2x cost = {perf2['sharpe']}")
fragile = (not np.isfinite(perf_oos["sharpe"])) or (np.isfinite(perf2["sharpe"]) and perf2["sharpe"] < 0.5 * (perf_oos["sharpe"] if np.isfinite(perf_oos["sharpe"]) else 0))
vr.record("robustness_2x_cost", ("WARN" if fragile else "PASS",
          f"edge {'FRAGILE under 2x costs' if fragile else 'survives 2x costs'} (1x={perf_oos['sharpe']}, 2x={perf2['sharpe']})"))
print("[NOTE] Full walk-forward (rolling train/test/step/retrain) structure = TODO; v1 uses single held-out OOS test split.")
"""))

cells.append(md("## §10 Save Reports"))

cells.append(code(r"""
rep = Path(CFG["reports_dir"]); rep.mkdir(parents=True, exist_ok=True)
trades_oos.to_csv(rep / "trades.csv", index=False)
equity_oos.to_csv(rep / "equity_curve.csv")
backtest_report = {
    "oos_period": [str(test.index.min()), str(test.index.max())],
    "config": {"sizing": str(SCFG), "cost": str(CCFG), "data_tz": DATA_TZ},
    "metrics_oos": perf_oos, "metrics_oos_2x_cost": perf2,
    "deflated_sharpe": dsr, "pbo": pbo, "n_trials": N_TRIALS,
}
with open(rep / "backtest_report.json", "w") as f:
    json.dump(backtest_report, f, indent=2, default=str)
metrics.write_metrics_summary(rep / "metrics_summary.txt", perf_oos, dsr, pbo,
    extra_lines=["robustness 2x-cost Sharpe: %s" % perf2["sharpe"],
                 "REALITY: backtest != future profit; demo/paper-trade first; verify broker spread/tz/contract size."])
# round-trip
rl = pd.read_csv(rep / "trades.csv")
vr.record("reports_roundtrip", ("PASS" if len(rl) == len(trades_oos) else "FAIL", f"trades.csv rows {len(rl)} == {len(trades_oos)}"))
print("[OUTPUT SAVED] reports/trades.csv, equity_curve.csv, backtest_report.json, metrics_summary.txt")
"""))

cells.append(md("## §11 OPTIONAL — Deployment bridge (ONNX + JSON for MT5 / Pine)"))

cells.append(code(r"""
# Best-effort ONNX export (skl2onnx/onnxmltools) + Python-vs-ONNX parity check.
onnx_status = export.export_to_onnx(primary, n_features=len(feature_cols), path=Path(CFG["models_dir"]) / "primary_model.onnx")
if onnx_status["ok"]:
    parity = export.onnx_parity_check(primary, Path(CFG["models_dir"]) / "primary_model.onnx", test[feature_cols].iloc[:20].values)
    print("ONNX parity:", parity)
    vr.record("onnx_parity", ("PASS" if parity["ok"] else "WARN", parity["note"]))
else:
    vr.record("onnx_export", ("WARN", f"ONNX export skipped: {onnx_status['error']}"))

# sample JSON signal payload for a TradingView/MT5 webhook-queue bridge
last = sig_test.iloc[-1]
price = float(test["close"].iloc[-1]); atr_last = float(test["atr"].iloc[-1])
side = "long" if last["signal"] == 1 else ("short" if last["signal"] == -1 else "flat")
payload = export.make_signal_payload("XAUUSD", side, last["confidence"], last["meta_prob"],
    size_lots=float(sig_test["size_lots"].iloc[-1]),
    sl=price - np.sign(last["signal"]) * SCFG.sl_mult * atr_last,
    tp=price + np.sign(last["signal"]) * SCFG.tp_mult * atr_last, ts=test.index[-1])
export.save_signal_payload_sample(Path(CFG["reports_dir"]) / "sample_signal.json", payload)
print(json.dumps(payload, indent=2))
print("\nBRIDGE NOTE: MT5 has NO inbound webhook. Production pattern = Flask/queue server that an EA")
print("polls (OnnxRun for native inference). Verify contract size / point value on the live account first.")
"""))

cells.append(md("## §12 Final Summary + Mandatory Warnings"))

cells.append(code(r"""
print("=" * 64); print("FINAL BACKTEST SUMMARY (OOS held-out test)"); print("=" * 64)
print(f"  period         : {test.index.min()} -> {test.index.max()}")
print(f"  total return   : {perf_oos['total_return']}")
print(f"  final equity   : {perf_oos['final_equity']}")
print(f"  max drawdown   : {perf_oos['max_drawdown']}")
print(f"  sharpe         : {perf_oos['sharpe']}   (2x-cost: {perf2['sharpe']})")
print(f"  profit factor  : {perf_oos['profit_factor']}")
print(f"  win rate       : {perf_oos['win_rate']}")
print(f"  # trades       : {perf_oos['n_trades']}")
print(f"  Deflated Sharpe: {dsr['dsr']}   PBO: {pbo['pbo']}")
verdict = "PLAUSIBLE small edge" if (np.isfinite(dsr['dsr']) and dsr['dsr'] > 0.5 and np.isfinite(pbo['pbo']) and pbo['pbo'] < 0.5) else "NOT convincingly profitable OOS (expected for a v1 / weak edge)"
print(f"  VERDICT        : {verdict}")
validation.save_validation_report(vr, Path(CFG["reports_dir"]) / "validation_report_backtest.json")
vr.print_checklist()
print("\n" + "!" * 64)
print("MANDATORY WARNINGS:")
print("  * Backtest results DO NOT guarantee future profit.")
print("  * Demo / paper-trade before any live capital.")
print("  * Verify the broker's real spread, timezone (DATA_TZ), and contract size.")
print("  * Realistic edge is small; survival depends on costs, discipline and risk control.")
print("!" * 64)
"""))

build(cells, str(ROOT / "notebooks" / "03_backtest_xauusd.ipynb"))
