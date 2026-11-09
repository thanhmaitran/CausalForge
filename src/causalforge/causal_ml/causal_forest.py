"""Honest causal forest (research implementation).

R-learner style splitting: with cross-fitted residuals ry = Y - m(X), rt = T - e(X), each tree
is grown on half of its subsample by weighted regression of ry/rt with weights rt^2 (so the
split criterion is the R-loss sum (ry - tau*rt)^2); leaf effects are then re-estimated on the
OTHER half (honesty): tau_leaf = sum(rt*ry)/sum(rt^2).

Uncertainty is a *heuristic* little-bags-inspired estimate (between-group spread of group-mean
predictions, minus Monte-Carlo noise). It is NOT the Wager-Athey theory-backed CI; its
empirical coverage is measured in EXP-03b. For a validated implementation use EconML's
CausalForestDML as the reference.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


class HonestCausalForest:
    name = "honest_causal_forest"

    def __init__(
        self,
        n_groups=20,
        trees_per_group=10,
        sample_fraction=0.35,
        min_samples_leaf=10,
        min_honest=5,
        max_depth=None,
        seed=0,
        n_folds=5,
    ):
        self.G, self.B = n_groups, trees_per_group
        self.sf, self.msl, self.min_honest = sample_fraction, min_samples_leaf, min_honest
        self.max_depth, self.seed, self.n_folds = max_depth, seed, n_folds

    def _residuals(self, X, t, y, e):
        ry, rt = np.zeros(len(y)), np.zeros(len(y))
        gb = HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=8, min_samples_leaf=20, random_state=0
        )
        for tr, te in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(X):
            ry[te] = y[te] - clone(gb).fit(X[tr], y[tr]).predict(X[te])
            if e is None:
                ph = LogisticRegression(max_iter=2000).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
            else:
                ph = e[te]
            rt[te] = t[te] - ph
        return ry, rt

    def fit(self, X, t, y, propensity=None):
        X, t, y = np.asarray(X, float), np.asarray(t, float), np.asarray(y, float)
        n = len(y)
        ry, rt = self._residuals(X, t, y, propensity)
        self.theta_ = float((rt * ry).sum() / (rt**2).sum())
        rng = np.random.default_rng(self.seed)
        self.groups_: list[list] = []
        m = int(self.sf * n)
        for _ in range(self.G):
            half = rng.choice(n, n // 2, replace=False)
            trees = []
            for _ in range(self.B):
                sub = rng.choice(half, min(m, len(half)), replace=False)
                a, b = sub[: len(sub) // 2], sub[len(sub) // 2 :]  # build / honest halves
                rta = np.where(np.abs(rt[a]) < 1e-3, 1e-3, rt[a])
                tree = DecisionTreeRegressor(
                    min_samples_leaf=self.msl,
                    max_depth=self.max_depth,
                    max_features=max(1, int(np.sqrt(X.shape[1])) + 1),
                    random_state=int(rng.integers(1 << 30)),
                )
                tree.fit(X[a], ry[a] / rta, sample_weight=rta**2)
                leaf = tree.apply(X[b])
                nl = tree.tree_.node_count
                num = np.bincount(leaf, rt[b] * ry[b], nl)
                den = np.bincount(leaf, rt[b] ** 2, nl)
                cnt = np.bincount(leaf, minlength=nl)
                tau = np.where(
                    (cnt >= self.min_honest) & (den > 1e-8), num / np.maximum(den, 1e-12), np.nan
                )
                trees.append((tree, tau))
            self.groups_.append(trees)
        return self

    def _group_preds(self, X):
        X = np.asarray(X, float)
        out = []
        for trees in self.groups_:
            P = np.column_stack([tau[tree.apply(X)] for tree, tau in trees])
            out.append(P)
        return out  # list of (n, B)

    @overload
    def predict_cate(
        self,
        X: np.ndarray,
        return_std: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def predict_cate(
        self,
        X: np.ndarray,
        return_std: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_cate(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        gp = self._group_preds(X)
        allp = np.concatenate(gp, axis=1)
        with np.errstate(all="ignore"):
            mean = np.nanmean(allp, axis=1)
        mean = np.where(np.isnan(mean), self.theta_, mean)
        if not return_std:
            return mean
        with np.errstate(all="ignore"):
            gm = np.column_stack([np.nanmean(P, axis=1) for P in gp])
            within = np.column_stack([np.nanvar(P, axis=1, ddof=1) / P.shape[1] for P in gp])
        gm = np.where(np.isnan(gm), mean[:, None], gm)
        raw = np.var(gm, axis=1, ddof=1)
        mc = np.nanmean(np.nan_to_num(within), axis=1)
        return mean, np.sqrt(np.maximum(raw - mc, 1e-6))
