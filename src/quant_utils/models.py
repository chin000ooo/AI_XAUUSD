"""
models.py — gradient-boosting wrapper with graceful fallbacks, evaluation,
probability calibration, and meta-label helpers.

Backend priority:  LightGBM -> XGBoost -> CatBoost -> sklearn HistGradientBoosting.

A thin ``GBMClassifier`` (sklearn-compatible, pickle-safe) label-encodes the
3-class target {-1,0,+1} -> {0,1,2} internally so every backend behaves the
same and ``classes_`` always reports the original labels. This keeps
calibration and meta-labelling backend-agnostic.

Methodology > model: the realistic OOS ceiling is ~51-55% accuracy / AUC ~0.5.
``evaluate_classifier`` returns honest metrics; the §0 smoke test lives in
validation.validate_smoke_metric.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, log_loss,
                             roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")


def _build_estimator(backend: str | None, params: dict, n_classes: int):
    """Return (estimator, backend_name) using the first available backend."""
    order = [backend] if backend else ["lightgbm", "xgboost", "catboost", "sklearn"]
    errors = {}
    for b in order:
        try:
            if b == "lightgbm":
                import lightgbm as lgb
                p = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=-1,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                         min_child_samples=50, random_state=42, n_jobs=-1, verbose=-1)
                p.update(params)
                return lgb.LGBMClassifier(**p), "lightgbm"
            if b == "xgboost":
                import xgboost as xgb
                p = dict(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1,
                         tree_method="hist", eval_metric="mlogloss")
                p.update(params)
                if n_classes > 2:
                    p["objective"] = "multi:softprob"
                return xgb.XGBClassifier(**p), "xgboost"
            if b == "catboost":
                from catboost import CatBoostClassifier
                p = dict(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
                         random_seed=42, verbose=False, allow_writing_files=False)
                p.update(params)
                return CatBoostClassifier(**p), "catboost"
            if b == "sklearn":
                from sklearn.ensemble import HistGradientBoostingClassifier
                p = dict(max_iter=400, learning_rate=0.05, max_depth=None,
                         l2_regularization=1.0, random_state=42, early_stopping=False)
                p.update(params)
                return HistGradientBoostingClassifier(**p), "sklearn"
        except Exception as e:  # backend not installed / failed to build
            errors[b] = str(e)
            continue
    raise RuntimeError(f"no gradient-boosting backend available; tried {order}; errors={errors}")


class GBMClassifier(BaseEstimator, ClassifierMixin):
    """Backend-agnostic GBM classifier with internal label encoding."""

    def __init__(self, backend: str | None = None, params: dict | None = None):
        self.backend = backend
        self.params = params or {}

    def fit(self, X, y, sample_weight=None):
        self.le_ = LabelEncoder().fit(y)
        yt = self.le_.transform(y)
        self.estimator_, self.backend_ = _build_estimator(self.backend, self.params, len(self.le_.classes_))
        try:
            self.estimator_.fit(X, yt, sample_weight=sample_weight)
        except TypeError:
            self.estimator_.fit(X, yt)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def predict_proba(self, X):
        return np.asarray(self.estimator_.predict_proba(X))

    def predict(self, X):
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    @property
    def feature_importances_(self):
        imp = getattr(self.estimator_, "feature_importances_", None)
        return imp


def train_primary_model(X, y, sample_weight=None, backend=None, params=None) -> GBMClassifier:
    return GBMClassifier(backend=backend, params=params).fit(X, y, sample_weight=sample_weight)


def evaluate_classifier(model, X, y, label_set=None) -> dict:
    """Honest metric bundle; every value guarded against NaN / degenerate cases."""
    y = np.asarray(y)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    classes = list(getattr(model, "classes_", np.unique(y)))
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "n": int(len(y)),
        "classes": [int(c) for c in classes],
    }
    # per-class
    out["f1_per_class"] = {int(c): float(v) for c, v in
                           zip(classes, f1_score(y, pred, average=None, labels=classes, zero_division=0))}
    # AUC + log loss when computable (binary uses P(positive); multiclass uses OvR)
    try:
        present = np.unique(y)
        if len(classes) == 2:
            out["roc_auc_ovr"] = float(roc_auc_score(y, proba[:, 1])) if len(present) == 2 else float("nan")
        elif len(present) == len(classes) and len(classes) > 1:
            out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=classes))
        else:
            out["roc_auc_ovr"] = float("nan")
    except Exception:
        out["roc_auc_ovr"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y, proba, labels=classes))
    except Exception:
        out["log_loss"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=classes).tolist()
    return out


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})
    s = pd.DataFrame({"feature": list(feature_names), "importance": np.asarray(imp, dtype=float)})
    return s.sort_values("importance", ascending=False).reset_index(drop=True)


def calibrate_model(fitted_clf, X_val, y_val, method: str = "sigmoid"):
    """Probability calibration on the VALIDATION set only (prefit estimator).
    Robust to sklearn's prefit deprecation via FrozenEstimator when available."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(fitted_clf), method=method)
        cal.fit(X_val, y_val)
        return cal
    except Exception:
        cal = CalibratedClassifierCV(fitted_clf, method=method, cv="prefit")
        cal.fit(X_val, y_val)
        return cal


# --------------------------------------------------------------------------
# meta-labelling helpers
# --------------------------------------------------------------------------
def make_meta_target(primary_pred: np.ndarray, outcome_dir: np.ndarray) -> np.ndarray:
    """meta_y = 1 if the primary's side matches the realised directional outcome,
    else 0. Defined ONLY on bars where the primary took a side (pred != 0).
    Returns an int array with -1 where there is no meta sample (primary neutral)."""
    primary_pred = np.asarray(primary_pred)
    outcome_dir = np.asarray(outcome_dir)
    meta = np.full(len(primary_pred), -1, dtype=int)
    took = primary_pred != 0
    meta[took] = (primary_pred[took] == outcome_dir[took]).astype(int)
    return meta


def optuna_tune_primary(X, y, sample_weight, splitter, event_end_idx,
                        n_trials: int = 20, backend=None, seed: int = 42) -> dict:
    """Tune GBM params inside the purged-CV loop (scalers/encoders fit on train
    folds only — there are none here beyond the encoder). Returns best params,
    or {} if optuna is unavailable."""
    try:
        import optuna
    except Exception:
        print("[WARNING] optuna unavailable — using robust default params.")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xv = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X)
    yv = np.asarray(y)
    sw = np.asarray(sample_weight) if sample_weight is not None else None

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        }
        scores = []
        for tr, te in splitter.split(Xv, event_end_idx):
            clf = GBMClassifier(backend=backend, params=params)
            clf.fit(Xv[tr], yv[tr], sample_weight=(sw[tr] if sw is not None else None))
            scores.append(balanced_accuracy_score(yv[te], clf.predict(Xv[te])))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[OPTUNA] best balanced_accuracy={study.best_value:.4f}")
    return study.best_params
