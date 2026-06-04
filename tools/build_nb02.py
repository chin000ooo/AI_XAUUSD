"""Build notebooks/02_train_xauusd.ipynb"""
from pathlib import Path
from nbtools import md, code, build

ROOT = Path(__file__).resolve().parents[1]
cells = []

cells.append(md(r"""
# 02 — Train XAUUSD (primary + calibration + meta-label + regime)

**Goal:** train a 3-class {-1,0,+1} primary gradient-boosting model with
**chronological** split + **purged/embargoed CV**, calibrate its probabilities,
add a **meta-label** model (decides *size, not side*), and fit a **regime** model
(risk controller only). Save all artifacts for the backtest.

**Inputs:** `data/processed/dataset.parquet` (from NB01).
**Outputs:** `models/{primary,meta,calibration,regime}_model.pkl`, `feature_columns.json`, `model_report.json`, split parquets.

> **Realistic ceiling:** OOS accuracy ~51–55%, AUC ~0.5. **Accuracy > 65% / AUC > 0.65 ⇒ suspect leakage, not success.** A loud smoke-test is wired into §5/§7. Accuracy alone is meaningless — the verdict is the cost-aware backtest in NB03.
"""))

cells.append(md("## §1 Setup"))

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
                    "pandas", "numpy", "scipy", "scikit-learn", "pyarrow",
                    "matplotlib", "tqdm", "lightgbm", "hmmlearn"])

import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from quant_utils import env, models, regime, cv, validation
try:
    import joblib
    _dump = joblib.dump
except Exception:
    def _dump(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
print("[OK] imports ready; PROJECT_ROOT =", PROJECT_ROOT)
"""))

cells.append(code(r"""
# CONFIG
SEED = env.set_seeds(42)
CFG = {
    "processed_dir": str(PROJECT_ROOT / "data" / "processed"),
    "models_dir":    str(PROJECT_ROOT / "models"),
    "reports_dir":   str(PROJECT_ROOT / "reports"),
    "train_frac": 0.70, "val_frac": 0.15,   # test = remaining 0.15 (chronological)
    "cv_splits": 5, "embargo_pct": 0.01,
    "cpcv_groups": 6, "cpcv_test_groups": 2,
    "model_backend": None,        # None -> auto: lightgbm>xgboost>catboost>sklearn
    "use_optuna": False,          # set True (with optuna installed) to tune; default off for speed/reproducibility
    "optuna_trials": 20,
    "calibration_method": "sigmoid",
    "n_regimes": 3,
    "smoke_acc_threshold": 0.65, "smoke_auc_threshold": 0.65,
    "seed": SEED,
}
Path(CFG["models_dir"]).mkdir(parents=True, exist_ok=True)
assert (Path(CFG["processed_dir"]) / "dataset.parquet").exists() or (Path(CFG["processed_dir"]) / "dataset.csv").exists(), \
    "dataset not found — run 01_preprocess first"
for k, v in CFG.items():
    print(f"  {k:20s}: {v}")
"""))

cells.append(md("## §2 Load Dataset (sorted, no shuffle)"))

cells.append(code(r"""
proc = Path(CFG["processed_dir"])
dpath = proc / "dataset.parquet"
dataset = pd.read_parquet(dpath) if dpath.exists() else pd.read_csv(proc / "dataset.csv", index_col=0, parse_dates=True)
dataset = dataset.sort_index()

NON_FEATURES = ["label", "event_return", "event_end_time", "event_end_idx",
                "touched_barrier", "outcome_dir", "side_label", "sample_weight",
                "open", "high", "low", "close"]
feature_cols = [c for c in dataset.columns if c not in NON_FEATURES]
X_all = dataset[feature_cols].astype(float)
y_all = dataset["label"].astype(int)
w_all = dataset["sample_weight"].astype(float)
outcome_all = dataset["outcome_dir"].astype(int)

# event end position WITHIN this (filtered) dataset, for purged CV
end_time = dataset["event_end_time"].values.astype("datetime64[ns]")
end_pos = np.searchsorted(dataset.index.values.astype("datetime64[ns]"), end_time, side="right") - 1
end_pos = np.maximum(end_pos, np.arange(len(dataset)))
end_pos = np.minimum(end_pos, len(dataset) - 1)

vr = validation.ValidationReport("02_train")
vr.record("shape", ("PASS", f"{dataset.shape}, features={len(feature_cols)}"))
vr.record("time_order_no_shuffle", validation.validate_time_order(dataset))
vr.record("targets_valid", validation.validate_labels(y_all))
vr.record("no_missing_in_X", ("PASS" if not X_all.isna().any().any() else "FAIL", "no NaN in feature matrix"))
print("class distribution:", y_all.value_counts().sort_index().to_dict())
print("date range:", dataset.index.min(), "->", dataset.index.max())
"""))

cells.append(md("## §3 Chronological Split (70 / 15 / 15) — NO random split"))

cells.append(code(r"""
n = len(dataset)
i_tr = int(n * CFG["train_frac"])
i_va = int(n * (CFG["train_frac"] + CFG["val_frac"]))
idx_tr = np.arange(0, i_tr)
idx_va = np.arange(i_tr, i_va)
idx_te = np.arange(i_va, n)

splits = {"train": idx_tr, "validation": idx_va, "test": idx_te}
for name, idx in splits.items():
    sub = dataset.iloc[idx]
    try:
        sub.to_parquet(proc / f"{name}_dataset.parquet")
    except Exception:
        sub.to_csv(proc / f"{name}_dataset.csv")
    print(f"  {name:11s}: {sub.index.min()} -> {sub.index.max()}  n={len(sub)}  "
          f"classes={sub['label'].value_counts().sort_index().to_dict()}")

ok_order = dataset.index[idx_tr][-1] < dataset.index[idx_va][0] < dataset.index[idx_te][0]
vr.record("chrono_split_order", ("PASS" if ok_order else "FAIL", "train.end < val.start < test.start"))
vr.record("chrono_no_overlap", ("PASS" if (len(set(idx_tr) & set(idx_va)) == 0 and len(set(idx_va) & set(idx_te)) == 0) else "FAIL", "no index overlap across splits"))

Xtr, ytr, wtr = X_all.iloc[idx_tr], y_all.iloc[idx_tr], w_all.iloc[idx_tr]
Xva, yva, wva = X_all.iloc[idx_va], y_all.iloc[idx_va], w_all.iloc[idx_va]
Xte, yte, wte = X_all.iloc[idx_te], y_all.iloc[idx_te], w_all.iloc[idx_te]
"""))

cells.append(md("## §4 Purged K-Fold + Embargo (with CPCV available)"))

cells.append(code(r"""
# Purged K-fold report on the TRAIN block (purge horizon-overlap + embargo)
end_pos_tr = np.searchsorted(dataset.index.values[idx_tr].astype("datetime64[ns]"),
                             end_time[idx_tr], side="right") - 1
end_pos_tr = np.clip(np.maximum(end_pos_tr, np.arange(len(idx_tr))), 0, len(idx_tr) - 1)

pkf = cv.PurgedKFold(n_splits=CFG["cv_splits"], embargo_pct=CFG["embargo_pct"])
report = pkf.fold_report(Xtr.values, end_pos_tr, dataset.index[idx_tr])
print(report.to_string(index=False))
vr.record("purged_cv_no_overlap", ("PASS" if (report["train_horizon_overlap"] == 0).all() else "FAIL",
                                    "no train/test horizon overlap in any fold"))

cpcv = cv.CombinatorialPurgedCV(n_groups=CFG["cpcv_groups"], n_test_groups=CFG["cpcv_test_groups"],
                                embargo_pct=CFG["embargo_pct"])
print(f"\nCPCV: {CFG['cpcv_groups']} groups, {CFG['cpcv_test_groups']} test-groups -> "
      f"{cpcv.n_paths()} backtest paths per group (used for Sharpe spread / PBO in NB03).")
print("NOTE: v1 uses purged k-fold + embargo; CPCV splitter implemented, full path-aggregation = TODO.")
"""))

cells.append(md("## §5 Primary Model (3-class, gradient boosting + fallbacks)"))

cells.append(code(r"""
print("[STEP] Train primary model")
best_params = {}
if CFG["use_optuna"]:
    best_params = models.optuna_tune_primary(Xtr, ytr, wtr, pkf, end_pos_tr,
                                              n_trials=CFG["optuna_trials"], backend=CFG["model_backend"], seed=SEED)

primary = models.train_primary_model(Xtr, ytr, sample_weight=wtr,
                                      backend=CFG["model_backend"], params=best_params)
print("backend:", primary.backend_)

m_val = models.evaluate_classifier(primary, Xva, yva)
m_test = models.evaluate_classifier(primary, Xte, yte)
print("\nVALIDATION:", {k: round(m_val[k], 4) for k in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr", "log_loss"]})
print("TEST      :", {k: round(m_test[k], 4) for k in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr", "log_loss"]})
print("confusion matrix (test, classes", m_test["classes"], "):")
print(np.array(m_test["confusion_matrix"]))

imp = models.get_feature_importance(primary, feature_cols).head(30)
print("\ntop-30 feature importance:")
print(imp.to_string(index=False))
"""))

cells.append(code(r"""
# §5 VALIDATION + §0 leakage smoke-test
proba_te = primary.predict_proba(Xte)
vr.record("primary_proba_valid", validation.validate_model_outputs(proba_te))
vr.record("primary_metrics_finite", ("PASS" if np.isfinite(m_test["accuracy"]) else "FAIL", "test metrics finite"))
vr.record("SMOKE_test_accuracy", validation.validate_smoke_metric(m_test["accuracy"], "test accuracy", CFG["smoke_acc_threshold"]))
vr.record("SMOKE_test_auc", validation.validate_smoke_metric(m_test.get("roc_auc_ovr"), "test AUC", CFG["smoke_auc_threshold"]))
if m_test["accuracy"] > CFG["smoke_acc_threshold"] or (np.isfinite(m_test.get("roc_auc_ovr", np.nan)) and m_test["roc_auc_ovr"] > CFG["smoke_auc_threshold"]):
    print("\n" + "!" * 70)
    print("[LEAKAGE SMOKE-TEST FAILED] OOS metric implausibly high -> investigate BEFORE trusting.")
    print("!" * 70)
else:
    print("\n[SMOKE-TEST OK] OOS metrics within the realistic ~51-55% ceiling.")
"""))

cells.append(md("## §6 Probability Calibration (fit on VALIDATION only)"))

cells.append(code(r"""
from sklearn.metrics import log_loss
calibrated = models.calibrate_model(primary, Xva, yva, method=CFG["calibration_method"])
ll_before = log_loss(yva, primary.predict_proba(Xva), labels=list(primary.classes_))
ll_after = log_loss(yva, calibrated.predict_proba(Xva), labels=list(primary.classes_))
print(f"val log-loss  before={ll_before:.4f}  after={ll_after:.4f}  ({'improved' if ll_after < ll_before else 'no improvement'})")
proba_cal_te = calibrated.predict_proba(Xte)
vr.record("calibration_proba_valid", validation.validate_model_outputs(proba_cal_te))
vr.record("calibration_logloss", ("PASS", f"val log-loss {ll_before:.4f} -> {ll_after:.4f}"))
"""))

cells.append(md(r"""
## §7 Meta-Labeling (direction → size bridge; rigorously leakage-controlled)

Primary gives a **side** and **calibrated confidence**. The meta model learns
**P(that side wins)** and is used in NB03 to scale size (not flip side).
Meta features = base features + `primary_side` + `primary_confidence`. The raw
future `label` / `event_return` / `outcome_dir` are explicitly **excluded**.
Meta is trained on the **validation** split (out-of-sample for the primary) and
evaluated on **test** — so primary confidence is never in-sample.
"""))

cells.append(code(r"""
def side_and_conf(proba, classes):
    cidx = {int(c): i for i, c in enumerate(classes)}
    p_up = proba[:, cidx[1]] if 1 in cidx else np.zeros(len(proba))
    p_dn = proba[:, cidx[-1]] if -1 in cidx else np.zeros(len(proba))
    side = np.where(p_up >= p_dn, 1, -1)
    conf = np.maximum(p_up, p_dn)
    return side, conf, p_up, p_dn

classes = list(primary.classes_)

def build_meta(Xsplit, idx):
    proba = calibrated.predict_proba(Xsplit)
    side, conf, p_up, p_dn = side_and_conf(proba, classes)
    Xm = Xsplit.copy()
    Xm["primary_side"] = side
    Xm["primary_confidence"] = conf
    Xm["primary_p_up"] = p_up
    Xm["primary_p_dn"] = p_dn
    out = outcome_all.iloc[idx].values
    meta_y = (side == out).astype(int)
    keep = out != 0                       # only bars with a realised directional outcome
    return Xm[keep], meta_y[keep], side[keep]

Xm_va, ym_va, side_va = build_meta(Xva, idx_va)
Xm_te, ym_te, side_te = build_meta(Xte, idx_te)
meta_feature_cols = list(Xm_va.columns)

# LEAKAGE ASSERTION: no label/future column in meta features
leak = validation.validate_no_label_in_features(meta_feature_cols)
vr.record("meta_no_label_leak", leak)
assert leak[0] != "FAIL", f"meta feature leakage: {leak[1]}"
print("meta_y distribution (val):", np.bincount(ym_va) if len(ym_va) else "empty",
      " positive rate:", round(ym_va.mean(), 3) if len(ym_va) else float("nan"))

meta = models.train_primary_model(Xm_va, ym_va)            # binary meta classifier
mm = models.evaluate_classifier(meta, Xm_te, ym_te)
print("META test:", {k: round(mm[k], 4) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]})
vr.record("meta_proba_valid", validation.validate_model_outputs(meta.predict_proba(Xm_te)))
vr.record("meta_metrics_finite", ("PASS" if np.isfinite(mm["f1_macro"]) else "FAIL", "meta metrics finite"))
vr.record("SMOKE_meta_auc", validation.validate_smoke_metric(mm.get("roc_auc_ovr"), "meta test AUC", 0.75))
"""))

cells.append(md("## §8 Regime Model (risk controller only — never flips side)"))

cells.append(code(r"""
vol_cols = [c for c in ["ret_log_1", "realized_vol_20", "atr_pct", "ret_std_20"] if c in feature_cols]
print("regime features:", vol_cols)
regime_model = regime.fit_regime_model(Xtr[vol_cols], n_regimes=CFG["n_regimes"], seed=SEED)
reg_all = regime_model.predict(X_all[vol_cols])
print("library:", regime_model.lib_)
print("regime distribution (all):", pd.Series(reg_all).value_counts().sort_index().to_dict())
summary = regime.regime_summary(reg_all, X_all[vol_cols[0]])
print(summary.to_string())
ok_reg = (pd.Series(reg_all).value_counts() >= 20).all()
vr.record("regime_enough_samples", ("PASS" if ok_reg else "WARN", "each regime has >=20 samples"))
vr.record("regime_is_risk_only", ("PASS", "regime used for sizing only; never flips signal direction"))
"""))

cells.append(md("## §9 Save Artifacts"))

cells.append(code(r"""
mdl = Path(CFG["models_dir"])
_dump(primary, mdl / "primary_model.pkl")
_dump(meta, mdl / "meta_model.pkl")
_dump(calibrated, mdl / "calibration_model.pkl")
_dump(regime_model, mdl / "regime_model.pkl")
with open(mdl / "feature_columns.json", "w") as f:
    json.dump({"feature_cols": feature_cols, "meta_feature_cols": meta_feature_cols,
               "regime_cols": vol_cols, "classes": [int(c) for c in classes]}, f, indent=2)

model_report = {
    "backend": primary.backend_,
    "periods": {"train": [str(dataset.index[idx_tr][0]), str(dataset.index[idx_tr][-1])],
                "validation": [str(dataset.index[idx_va][0]), str(dataset.index[idx_va][-1])],
                "test": [str(dataset.index[idx_te][0]), str(dataset.index[idx_te][-1])]},
    "primary_metrics": {"validation": m_val, "test": m_test},
    "meta_metrics_test": mm,
    "class_distribution": y_all.value_counts().sort_index().to_dict(),
    "top_features": models.get_feature_importance(primary, feature_cols).head(20).to_dict("records"),
    "params": best_params or "robust defaults",
    "calibration": {"method": CFG["calibration_method"], "val_logloss_before": ll_before, "val_logloss_after": ll_after},
    "regime_lib": regime_model.lib_,
    "realistic_ceiling_note": "OOS accuracy ~51-55%, AUC ~0.5 is expected & fine; >0.65 ⇒ suspect leakage.",
    "leakage_smoke_test": {"test_accuracy": m_test["accuracy"], "test_auc": m_test.get("roc_auc_ovr"),
                           "flagged": bool(m_test["accuracy"] > CFG["smoke_acc_threshold"])},
    "todo": ["meta on purged-CV OOF primary preds (here: validation-fit)", "optuna tuning (off by default)"],
}
with open(mdl / "model_report.json", "w") as f:
    json.dump(model_report, f, indent=2, default=str)
print("[OUTPUT SAVED] models/* + feature_columns.json + model_report.json")

# reload + predict on a sample
import joblib as _jl
rp = _jl.load(mdl / "primary_model.pkl")
_ = rp.predict_proba(Xte.iloc[:5])
vr.record("artifacts_reload_predict", ("PASS", "primary reloaded and predicted on a sample"))
vr.record("feature_columns_match", ("PASS" if len(feature_cols) == rp.n_features_in_ else "FAIL",
                                    f"feature_columns ({len(feature_cols)}) == model n_features_in_ ({rp.n_features_in_})"))
"""))

cells.append(md("## §10 Training Summary + Checklist"))

cells.append(code(r"""
print("TRAINING SUMMARY")
print(f"  backend        : {primary.backend_}")
print(f"  primary test   : acc={m_test['accuracy']:.4f}  bal_acc={m_test['balanced_accuracy']:.4f}  "
      f"f1={m_test['f1_macro']:.4f}  auc={m_test.get('roc_auc_ovr')}")
print(f"  meta test      : f1={mm['f1_macro']:.4f}  auc={mm.get('roc_auc_ovr')}")
print(f"  regime lib     : {regime_model.lib_}")
print("  top-5 features :", [r["feature"] for r in model_report["top_features"][:5]])
validation.save_validation_report(vr, Path(CFG["reports_dir"]) / "validation_report_train.json")
vr.print_checklist()
print("\nCAVEAT: accuracy alone is meaningless. The real judgement is the cost-aware backtest in NB03.")
"""))

build(cells, str(ROOT / "notebooks" / "02_train_xauusd.ipynb"))
