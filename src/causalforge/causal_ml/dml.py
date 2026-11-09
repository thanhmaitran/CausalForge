"""Double Machine Learning, partially linear model, with true K-fold cross-fitting."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from causalforge.core.types import EstimateResult, normal_ci


def dml_plr(ds, model_y=None, model_t=None, n_folds: int = 5, seed: int = 0) -> EstimateResult:
    """Y = theta*T + g(X) + e ; T = m(X) + v.   theta = sum(rT*rY)/sum(rT^2).

    Nuisances E[Y|X], E[T|X] are fit on training folds and predicted on the held-out
    fold (out-of-fold residuals only). SE from the orthogonal score.
    Note: under heterogeneous effects theta is a variance-weighted average of CATEs,
    not necessarily the ATE; use cross-fitted AIPW for the ATE.
    """
    mk = lambda: HistGradientBoostingRegressor(
        max_iter=80,
        max_leaf_nodes=8,  # noqa: E731
        min_samples_leaf=20,
        random_state=0,
    )
    my = mk() if model_y is None else model_y
    mt = mk() if model_t is None else model_t
    X, t, y = ds.X, ds.treatment.astype(float), ds.outcome
    ry, rt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        ry[te] = y[te] - clone(my).fit(X[tr], y[tr]).predict(X[te])
        rt[te] = t[te] - clone(mt).fit(X[tr], t[tr]).predict(X[te])
    theta = float((rt * ry).sum() / (rt**2).sum())
    n = len(y)
    psi = (ry - theta * rt) * rt
    se = float(np.sqrt(np.mean(psi**2) / np.mean(rt**2) ** 2 / n))
    return EstimateResult(
        "dml_plr",
        "ATE(PLR)",
        theta,
        se,
        normal_ci(theta, se),
        {"n_folds": n_folds, "residual_t_var": float(rt.var())},
    )
