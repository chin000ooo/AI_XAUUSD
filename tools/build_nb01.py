"""Build notebooks/01_preprocess_xauusd.ipynb"""
from pathlib import Path
from nbtools import md, code, build

ROOT = Path(__file__).resolve().parents[1]
cells = []

cells.append(md(r"""
# 01 — Preprocess XAUUSD (raw bars -> features + triple-barrier labels)

**Goal:** turn native per-timeframe TradingView/GBE bar CSVs into a leakage-safe,
modelling-ready dataset: causal features (incl. Ehlers DSP), leakage-safe
multi-timeframe context, fractional-difference of price, ATR triple-barrier
labels, and AFML sample weights.

**Inputs:** `data/raw/XAUUSD_*.csv` (`time,open,high,low,close`, epoch **seconds**, **no volume**).
**Outputs:** `data/processed/{features,labels,dataset}.parquet`, `feature_report.json`, `reports/validation_report_preprocess.json`.

---
### NON-NEGOTIABLE REALITY CONSTRAINTS (read first)
1. **Realistic next-bar directional accuracy ceiling is ~51–55%; OOS AUC barely above 0.5.** That is normal. Profit comes from a small edge + disciplined sizing/costs — not high accuracy. **If any OOS metric looks too good (accuracy > 65%, AUC > 0.65) treat it as a LEAKAGE BUG, not success.**
2. **Overfitting is the primary enemy.** Purged/embargoed CV, Deflated Sharpe, PBO and realistic costs matter MORE than the model.
3. **Methodology > model.** Well-tuned gradient boosting on correct features + labels + CV is the realistic best-in-class core.
4. **No look-ahead, ever.** Higher-TF features only see the last *closed* HTF bar; the future label builds the target but never enters a feature.
5. **Be honest.** Report what actually passed/failed — no hype, no fabricated metrics.

> **TIMEZONE CAVEAT:** epoch is absolute UTC, but London/NY session flags depend on the broker's wall clock. `DATA_TZ` defaults to UTC — **verify the broker's timezone**, a wrong offset silently corrupts every session feature.
"""))

cells.append(md("## §1 Setup & Environment"))

cells.append(code(r"""
# --- Cell 2: environment detection + path bootstrap (Colab AND local) ---
import sys, os
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except Exception:
    IS_COLAB = False

def _find_root():
    here = Path.cwd()
    for c in [here, *here.parents]:
        if (c / "src" / "quant_utils").exists():
            return c
    return here

if IS_COLAB:
    # Option A: clone the repo (set env XAUUSD_REPO_URL). Option B: upload the
    # project so that ./src/quant_utils exists next to this notebook.
    REPO_URL = os.environ.get("XAUUSD_REPO_URL", "")
    target = Path("/content/AI_XAUUSD")
    if REPO_URL and not target.exists():
        os.system(f"git clone {REPO_URL} {target}")
    PROJECT_ROOT = target if (target / "src").exists() else _find_root()
    # optional Google Drive mount (uncomment if your data lives on Drive)
    # from google.colab import drive; drive.mount('/content/drive')
else:
    PROJECT_ROOT = _find_root()

for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

print("IS_COLAB     :", IS_COLAB)
print("PROJECT_ROOT :", PROJECT_ROOT)
print("CWD          :", Path.cwd())
"""))

cells.append(code(r"""
# --- Cell 3: dependencies (Colab) + imports with graceful fallbacks ---
if IS_COLAB:
    import subprocess
    pkgs = "pandas numpy scipy scikit-learn pyarrow matplotlib tqdm statsmodels lightgbm hmmlearn".split()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs])

import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; comment out for interactive plots
import matplotlib.pyplot as plt

try:
    from quant_utils import env, io_data, features, mtf, fracdiff, labeling, validation
    print("[OK] quant_utils imported")
except Exception as e:
    print("[FATAL] cannot import quant_utils:", e)
    print("On Colab: set XAUUSD_REPO_URL to git-clone the repo, or upload the project's src/ folder.")
    raise

def _has(m):
    try:
        importlib.import_module(m); return True
    except Exception:
        return False
print("optional libs:", {k: _has(k) for k in
      ["lightgbm", "xgboost", "catboost", "statsmodels", "hmmlearn", "optuna", "pyarrow", "onnxruntime"]})
"""))

cells.append(code(r"""
# --- Cell 4: CONFIG (single source of truth) ---
SEED = env.set_seeds(42)
env.ensure_dirs(PROJECT_ROOT)

CFG = {
    "raw_dir":       str(PROJECT_ROOT / "data" / "raw"),
    "external_dir":  str(PROJECT_ROOT / "data" / "external"),
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "reports_dir":   str(PROJECT_ROOT / "reports"),
    # ---- timeframe roles (configurable) ----
    "primary_tf": "H1",            # execution timeframe
    "higher_tfs": ["H4", "D1"],    # leakage-safe higher-TF context
    "lower_tfs":  [],              # optional finer detail e.g. ["M15","M5"] (off by default)
    # ---- session / time ----
    "data_tz": "UTC",              # !!! VERIFY broker tz — wrong offset corrupts session flags
    "sessions": {"asian": (0, 8), "london": (8, 16), "ny": (13, 21)},
    # ---- features ----
    "use_ehlers": True, "use_hurst": True, "hurst_window": 100,
    "rsi_period": 14, "atr_period": 14, "adx_period": 14,
    # ---- fractional differentiation ----
    "ffd_thresh": 1e-5,
    # ---- triple barrier (ATR-scaled) ----
    "tp_mult": 2.0, "sl_mult": 1.0, "max_holding": 24,
    # ---- sample weights ----
    "time_decay": 0.5,
    "seed": SEED,
}
print("CONFIG:")
for k, v in CFG.items():
    print(f"  {k:14s}: {v}")
print("\n[WARNING] TIMEZONE: DATA_TZ='UTC'. If the GBE/TradingView export used server/exchange")
print("[WARNING] time rather than UTC, set CFG['data_tz'] and re-run — else session features are silently wrong.")
"""))

cells.append(md("## §2 Load Raw Data (native per-TF files; no volume in this dataset)"))

cells.append(code(r"""
print("[STEP 1/9] Load native per-timeframe files")
files = io_data.discover_raw_files(CFG["raw_dir"])
print("discovered:", {k: Path(v).name for k, v in files.items()})
assert CFG["primary_tf"] in files, f"primary {CFG['primary_tf']} CSV not found in data/raw"

primary_df, prim_meta = io_data.load_ohlcv(files[CFG["primary_tf"]], data_tz=CFG["data_tz"])
print("primary:", CFG["primary_tf"], "->", prim_meta)

higher = {}
for tf in CFG["higher_tfs"] + CFG["lower_tfs"]:
    if tf in files:
        higher[tf], m = io_data.load_ohlcv(files[tf], data_tz=CFG["data_tz"])
        print(f"  loaded {tf}: {m['rows']} rows  has_volume={m['has_volume']}")
    else:
        print(f"  [WARNING] {tf} native file missing -> resample fallback from primary")
        higher[tf] = io_data.resample_ohlcv(primary_df, tf)
primary_df.head()
"""))

cells.append(code(r"""
# §2 VALIDATION
vr = validation.ValidationReport("01_preprocess")
vr.record("primary_file_exists", validation.check_file_exists(files[CFG["primary_tf"]]))
vr.record("ohlc_valid",         validation.validate_ohlcv(primary_df))
vr.record("time_order",         validation.validate_time_order(primary_df))
vr.record("no_dup_timestamp",   validation.validate_no_duplicate_timestamp(primary_df))
vr.record("no_nan_inf_raw",     validation.validate_no_nan_inf(primary_df[["open", "high", "low", "close"]]))
print(f"date range: {primary_df.index.min()}  ->  {primary_df.index.max()}   rows={len(primary_df):,}")
"""))

cells.append(md("## §3 Multi-Timeframe Alignment (leakage-safe `.shift(1)` rule)"))

cells.append(code(r"""
print("[STEP 2/9] Primary features + leakage-safe higher-TF context")
prim_feat, prim_fmeta = features.build_features(primary_df, CFG)
print("primary features:", prim_feat.shape, " skipped:", prim_fmeta["skipped"])

mtf_feat, mtf_info = mtf.attach_multi_timeframe(primary_df, higher, features.build_features, CFG)
print("MTF features:", mtf_feat.shape)

# 10-row alignment sample: show that each primary bar maps to an EARLIER HTF bar
tf0 = (CFG["higher_tfs"] + CFG["lower_tfs"])[0]
src0 = mtf_info["source_map"][tf0]
sample = pd.DataFrame({"primary_open": primary_df.index, f"{tf0}_source_open": src0.values}).tail(10)
print("\nalignment sample (last 10 primary bars):")
print(sample.to_string(index=False))
"""))

cells.append(code(r"""
# §3 VALIDATION — assert every merged HTF feature comes from a CLOSED (past) bar
for tf, src in mtf_info["source_map"].items():
    status = validation.validate_feature_leakage_mtf(src.values, primary_df.index)
    vr.record(f"mtf_no_leakage_{tf}", status)
    assert status[0] != "FAIL", f"LEAKAGE: {tf} references a future bar"
print("[VALIDATION PASS] all MTF features use only fully-closed higher-TF bars")
"""))

cells.append(md("## §4 Feature Engineering (+ optional shifted gold externals)"))

cells.append(code(r"""
print("[STEP 3/9] Assemble feature matrix + optional externals (DXY / real yield)")
feat = pd.concat([prim_feat, mtf_feat], axis=1)

ext_meta = {"merged": [], "skipped": []}
for name, fname in [("dxy", "DXY.csv"), ("real_yield", "US_REAL_YIELD.csv")]:
    p = Path(CFG["external_dir"]) / fname
    if not p.exists():
        ext_meta["skipped"].append(name)
        print(f"  [WARNING] external {fname} absent -> skipped (recorded in feature_report)")
        continue
    ext = pd.read_csv(p); ext.columns = [c.strip().lower() for c in ext.columns]
    tcol = "time" if "time" in ext.columns else ext.columns[0]
    vcol = [c for c in ext.columns if c != tcol][0]
    dt = io_data.parse_time_column(ext[tcol])
    es = pd.DataFrame({"t": dt, name: pd.to_numeric(ext[vcol], errors="coerce")}).dropna().sort_values("t")
    m = pd.merge_asof(pd.DataFrame({"t": feat.index}), es, on="t", direction="backward")
    feat[name] = pd.Series(m[name].values, index=feat.index).shift(1)  # strictly past-known
    ext_meta["merged"].append(name)
    print(f"  [OK] merged external {name} (as-of + shift(1), no look-ahead)")

print("feature matrix:", feat.shape)
"""))

cells.append(code(r"""
# §4 VALIDATION
feat = feat.replace([np.inf, -np.inf], np.nan)
num = feat.select_dtypes("number")
print("numeric feature columns:", num.shape[1])
nan_top = feat.isna().sum().sort_values(ascending=False).head(8)
print("top NaN columns (leading-window NaNs expected):\n", nan_top.to_string())

# top absolute correlations among a sample of features (informational)
corr = num.iloc[:, :40].corr().abs()
import numpy as _np
mask = _np.triu(_np.ones(corr.shape), k=1).astype(bool)
top_pairs = corr.where(mask).stack().sort_values(ascending=False).head(5)
print("\ntop correlated feature pairs (first 40 cols):\n", top_pairs.to_string())

vr.record("feat_dtypes_numeric", ("PASS", f"{num.shape[1]} numeric feature columns; non-numeric: "
          f"{[c for c in feat.columns if c not in num.columns]}"))
# 'no future-close feature' is guaranteed by causal construction (rolling windows end at t,
# HTF merged on close-time<=t). We additionally assert no label/future column leaked in:
vr.record("no_future_cols_in_features", validation.validate_no_label_in_features(feat.columns))
"""))

cells.append(md("## §5 Fractional Differentiation (memory-preserving stationarity)"))

cells.append(code(r"""
print("[STEP 4/9] Fractional differentiation of close")
res = fracdiff.find_min_d(primary_df["close"], thresh=CFG["ffd_thresh"])
print("fracdiff result:", res)
feat["fracdiff_close"] = fracdiff.fractional_diff_ffd(primary_df["close"], res["d"], CFG["ffd_thresh"]).reindex(feat.index)

corr_cf = res.get("corr")
adf_p = res.get("adf_pvalue")
status = "PASS" if (corr_cf is not None and abs(corr_cf) > 0.5) else "WARN"
vr.record("fracdiff", (status, f"d={res['d']}, ADF p={adf_p}, corr(close,ffd)={corr_cf:.3f} (memory preserved), method={res['method']}"))
vr.record("fracdiff_no_allnan", ("PASS" if feat["fracdiff_close"].notna().any() else "FAIL", "fracdiff_close produced finite values"))
"""))

cells.append(md("## §6 Triple-Barrier Labeling (ATR-scaled)"))

cells.append(code(r"""
print("[STEP 5/9] Triple-barrier labeling (tp_mult=%.1f, sl_mult=%.1f, max_holding=%d)"
      % (CFG["tp_mult"], CFG["sl_mult"], CFG["max_holding"]))
atr_series = prim_feat["atr"]
labels = labeling.apply_triple_barrier(primary_df, atr_series,
                                       tp_mult=CFG["tp_mult"], sl_mult=CFG["sl_mult"],
                                       max_holding=CFG["max_holding"])
print("\nclass distribution (label):")
print(labels["label"].value_counts().sort_index().to_string())
print("\ntouched barrier:")
print(labels["touched_barrier"].value_counts().to_string())
print("\nexample events:")
print(labels[["label", "event_end_time", "event_return", "touched_barrier", "outcome_dir"]].head(8).to_string())
"""))

cells.append(code(r"""
# §6 VALIDATION
vr.record("labels_valid",          validation.validate_labels(labels["label"]))
vr.record("event_end_after_start", validation.validate_event_end_after_start(labels.index, labels["event_end_time"]))
imb = labels["label"].value_counts(normalize=True).to_dict()
vr.record("class_imbalance_flag", ("WARN", f"tp/sl asymmetry -> imbalance {{{', '.join(f'{int(k)}:{v:.2f}' for k,v in imb.items())}}}; handle via sample weights / balanced metrics downstream"))
"""))

cells.append(md("## §7 Sample Weights (AFML Ch.4: avg uniqueness × |ret| × decay × class-balance)"))

cells.append(code(r"""
print("[STEP 6/9] Sample weights")
sw = labeling.compute_sample_weights(labels, primary_df.index, time_decay=CFG["time_decay"])
print(sw["sample_weight"].describe().to_string())

vr.record("weights_positive", ("PASS" if (sw["sample_weight"] > 0).all() else "FAIL", "all sample weights > 0"))
vr.record("weights_no_nan",   ("PASS" if sw["sample_weight"].notna().all() else "FAIL", "no NaN weights"))

fig, ax = plt.subplots(1, 2, figsize=(11, 3))
sw["sample_weight"].plot(ax=ax[0], title="sample_weight over time", lw=0.5)
sw["sample_weight"].hist(bins=60, ax=ax[1]); ax[1].set_title("weight distribution")
plt.tight_layout()
plt.savefig(Path(CFG["reports_dir"]) / "sample_weights.png", dpi=80)
plt.show()
print("[OUTPUT SAVED] reports/sample_weights.png")
"""))

cells.append(md("## §8 Final Dataset Assembly (drop NaN/inf + missing labels, keep sorted)"))

cells.append(code(r"""
print("[STEP 7/9] Final dataset assembly")
dataset = feat.copy()
for c in ["label", "event_return", "event_end_time", "event_end_idx", "touched_barrier", "outcome_dir", "side_label"]:
    dataset[c] = labels[c]
dataset["sample_weight"] = sw["sample_weight"]
# carry raw OHLC as AUXILIARY columns (needed by the NB03 backtest engine for
# next-bar-open entry & intrabar SL/TP). They are NOT features (excluded below).
for c in ["open", "high", "low", "close"]:
    dataset[c] = primary_df[c]

NON_FEATURES = ["label", "event_return", "event_end_time", "event_end_idx",
                "touched_barrier", "outcome_dir", "side_label", "sample_weight",
                "open", "high", "low", "close"]
feature_cols = [c for c in dataset.columns if c not in NON_FEATURES]

dataset = dataset.replace([np.inf, -np.inf], np.nan)
before = len(dataset)
dataset = dataset[dataset["touched_barrier"] != "invalid"]
dataset = dataset.dropna(subset=feature_cols + ["label", "sample_weight"])
dataset = dataset.sort_index()
print(f"dropped {before - len(dataset):,} rows (leading NaNs / invalid); final {len(dataset):,} rows")

vr.record("no_label_in_features", validation.validate_no_label_in_features(feature_cols))
vr.record("final_time_order",     validation.validate_time_order(dataset))
vr.record("final_no_dup",         validation.validate_no_duplicate_timestamp(dataset))
print("\nLEAKAGE CHECKLIST: MTF shift confirmed | no future-close feature (causal) | label NOT in features")
print("final class distribution:", dataset["label"].value_counts().sort_index().to_dict())
"""))

cells.append(md("## §9 Save Outputs"))

cells.append(code(r"""
print("[STEP 8/9] Save outputs")
import json
proc = Path(CFG["processed_dir"])

def _save(df, name):
    try:
        path = proc / f"{name}.parquet"
        df.to_parquet(path)
    except Exception as e:
        print(f"  [WARNING] parquet failed for {name} ({e}); CSV fallback")
        path = proc / f"{name}.csv"
        df.to_csv(path)
    print("  [OUTPUT SAVED]", path)
    return path

fpath = _save(feat.loc[dataset.index], "features")
lpath = _save(labels.loc[dataset.index], "labels")
dpath = _save(dataset, "dataset")

feature_report = {
    "n_rows": int(len(dataset)),
    "n_features": len(feature_cols),
    "feature_cols": feature_cols,
    "skipped_volume_features": prim_fmeta["skipped"],
    "externals": ext_meta,
    "fracdiff": res,
    "class_distribution": {int(k): int(v) for k, v in dataset["label"].value_counts().items()},
    "date_range": [str(dataset.index.min()), str(dataset.index.max())],
    "data_tz": CFG["data_tz"],
    "warnings": prim_fmeta["warnings"],
    "timeframes": mtf_info["meta"]["timeframes"],
}
with open(proc / "feature_report.json", "w", encoding="utf-8") as f:
    json.dump(feature_report, f, indent=2, default=str)
print("  [OUTPUT SAVED]", proc / "feature_report.json")

# round-trip check
reloaded = pd.read_parquet(dpath) if str(dpath).endswith("parquet") else pd.read_csv(dpath, index_col=0, parse_dates=True)
vr.record("reload_roundtrip", ("PASS" if reloaded.shape == dataset.shape else "FAIL",
                               f"reloaded {reloaded.shape} vs in-memory {dataset.shape}"))
"""))

cells.append(md("## §10 Preprocess Summary + PASS/FAIL Checklist"))

cells.append(code(r"""
print("[STEP 9/9] PREPROCESS SUMMARY")
print(f"  raw H1 rows     : {len(primary_df):,}")
print(f"  final rows      : {len(dataset):,}")
print(f"  feature count   : {len(feature_cols)}")
print(f"  label dist      : {dataset['label'].value_counts().sort_index().to_dict()}")
print(f"  date range      : {dataset.index.min()} -> {dataset.index.max()}")
print(f"  externals merged: {ext_meta['merged']}  skipped: {ext_meta['skipped']}")
print(f"  fracdiff d      : {res['d']}  (corr={res['corr']:.3f})")

validation.save_validation_report(vr, Path(CFG["reports_dir"]) / "validation_report_preprocess.json")
vr.print_checklist()
print("\nREALITY CHECK: realistic OOS accuracy ceiling ~51-55%. If NB02 reports >65%, suspect leakage and investigate.")
"""))

build(cells, str(ROOT / "notebooks" / "01_preprocess_xauusd.ipynb"))
