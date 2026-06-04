"""
regime.py — market regime model (HMM if available, else GaussianMixture).

Fit on stationary volatility-family features (log return, realized vol, ATR%,
rolling std). Default 3 regimes ordered by volatility: 0=calm, 1=normal,
2=turbulent.

CRITICAL: the regime is a RISK CONTROLLER only. It may shrink or zero position
size in turbulent conditions; it must NEVER flip signal direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    _HAS_HMM = True
except Exception:  # pragma: no cover
    _HAS_HMM = False

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeModel:
    """Wraps an HMM/GMM with a volatility-ordered state mapping (pickle-safe)."""

    def __init__(self, n_regimes: int = 3, backend: str = "auto", seed: int = 42):
        self.n_regimes = n_regimes
        self.backend = backend
        self.seed = seed
        self.scaler_ = None
        self.model_ = None
        self.order_ = None         # raw_state -> ordered regime (by vol)
        self.lib_ = None
        self.feature_cols_ = None

    def _make(self):
        if (self.backend in ("auto", "hmm")) and _HAS_HMM:
            self.lib_ = "hmmlearn.GaussianHMM"
            return GaussianHMM(n_components=self.n_regimes, covariance_type="diag",
                               n_iter=200, random_state=self.seed)
        self.lib_ = "sklearn.GaussianMixture"
        return GaussianMixture(n_components=self.n_regimes, covariance_type="diag",
                               n_init=3, random_state=self.seed)

    def fit(self, X: pd.DataFrame):
        self.feature_cols_ = list(X.columns)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X.to_numpy(dtype=float))
        self.model_ = self._make()
        self.model_.fit(Xs)
        raw = self.model_.predict(Xs)
        # order states by mean of the first (volatility) feature
        vol = X.iloc[:, 0].to_numpy(dtype=float)
        means = [np.nanmean(vol[raw == s]) if np.any(raw == s) else np.inf
                 for s in range(self.n_regimes)]
        order = np.argsort(np.argsort(means))   # raw state -> rank (0=lowest vol)
        self.order_ = {s: int(order[s]) for s in range(self.n_regimes)}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler_.transform(X[self.feature_cols_].to_numpy(dtype=float))
        raw = self.model_.predict(Xs)
        return np.vectorize(lambda s: self.order_.get(int(s), int(s)))(raw)


def fit_regime_model(X: pd.DataFrame, n_regimes: int = 3, seed: int = 42) -> RegimeModel:
    return RegimeModel(n_regimes=n_regimes, seed=seed).fit(X)


def regime_summary(regimes: np.ndarray, vol: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"regime": regimes, "vol": vol.to_numpy(dtype=float)})
    g = df.groupby("regime")["vol"].agg(["count", "mean", "std", "min", "max"])
    g["label"] = g.index.map({0: "calm", 1: "normal", 2: "turbulent"}).fillna("regime")
    return g
