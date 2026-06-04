"""
cv.py — Purged K-Fold + embargo and Combinatorial Purged CV (López de Prado Ch.7).

Standard k-fold leaks in finance because (a) labels span multiple bars (their
horizons overlap test windows) and (b) serial correlation bleeds train into
test. PurgedKFold removes training samples whose label horizon
[start, event_end] overlaps the test window, then embargoes a fraction of bars
immediately AFTER each test fold.

CombinatorialPurgedCV (CPCV) generates many train/test splits → multiple
backtest paths, used for the Sharpe distribution / PBO in Notebook 3.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _embargo_count(n: int, embargo_pct: float) -> int:
    return int(np.ceil(n * embargo_pct)) if embargo_pct > 0 else 0


class PurgedKFold:
    """K contiguous test folds with purging + embargo.

    Parameters
    ----------
    n_splits : number of folds.
    embargo_pct : fraction of total samples embargoed after each test fold.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, event_end_idx):
        """Yield (train_idx, test_idx) integer-position arrays.

        ``event_end_idx`` : array of the integer end position of each sample's
        label horizon (from labeling.apply_triple_barrier).
        """
        n = len(X)
        idx = np.arange(n)
        end = np.asarray(event_end_idx)
        emb = _embargo_count(n, self.embargo_pct)
        fold_bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            t0, t1 = fold_bounds[k], fold_bounds[k + 1]
            test_idx = idx[t0:t1]
            if len(test_idx) == 0:
                continue
            test_start, test_end = t0, t1 - 1

            train_mask = np.ones(n, dtype=bool)
            train_mask[t0:t1] = False
            # purge: drop train samples whose horizon overlaps the test window
            overlap = (idx <= test_end) & (end >= test_start)
            train_mask &= ~overlap
            # embargo: drop a band right after the test fold
            if emb > 0:
                emb_end = min(n, test_end + 1 + emb)
                train_mask[test_end + 1: emb_end] = False
            train_idx = idx[train_mask]
            yield train_idx, test_idx

    def fold_report(self, X, event_end_idx, time_index) -> pd.DataFrame:
        rows = []
        for k, (tr, te) in enumerate(self.split(X, event_end_idx)):
            end = np.asarray(event_end_idx)
            # leakage check: no train horizon may reach into the test window
            te_start, te_end = te[0], te[-1]
            tr_overlap = int(((tr <= te_end) & (end[tr] >= te_start)).sum())
            rows.append({
                "fold": k,
                "train_n": len(tr),
                "test_n": len(te),
                "test_start": str(time_index[te_start]),
                "test_end": str(time_index[te_end]),
                "train_horizon_overlap": tr_overlap,
                "status": "PASS" if tr_overlap == 0 else "FAIL",
            })
        return pd.DataFrame(rows)


class CombinatorialPurgedCV:
    """CPCV (López de Prado Ch.12). Splits the timeline into ``n_groups``
    contiguous groups, chooses every combination of ``n_test_groups`` as the
    test set, purges + embargoes the rest for training. Each combination is a
    distinct, overlapping backtest path.

    NOTE: the splitter and per-combination paths are fully implemented here.
    Full path-recombination into N(φ) clean backtest paths is left as a TODO
    (Notebook 3 uses the per-combination test returns for the PBO/Sharpe spread).
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def n_paths(self) -> int:
        from math import comb
        # number of paths each group participates in (LdP): C(N-1, k-1)
        return int(comb(self.n_groups - 1, self.n_test_groups - 1))

    def split(self, X, event_end_idx):
        n = len(X)
        idx = np.arange(n)
        end = np.asarray(event_end_idx)
        emb = _embargo_count(n, self.embargo_pct)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [idx[bounds[g]:bounds[g + 1]] for g in range(self.n_groups)]

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in combo])
            test_idx.sort()
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            # purge against each contiguous test block
            for g in combo:
                tb = groups[g]
                if len(tb) == 0:
                    continue
                ts, teh = tb[0], tb[-1]
                overlap = (idx <= teh) & (end >= ts)
                train_mask &= ~overlap
                if emb > 0:
                    train_mask[teh + 1: min(n, teh + 1 + emb)] = False
            yield idx[train_mask], test_idx, combo
