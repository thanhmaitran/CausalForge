"""AIPW (doubly robust) estimator with optional cross-fitting."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from causalforge.core.types import EstimateResult, normal_ci


def aipw_ate(
    ds,
    outcome_model=None,
    propensity_model=None,
    n_folds: int = 5,
    seed: int = 0,
    clip: float = 0.01,
    return_psi: bool = False,
):
    """AIPW ATE; n_folds>=2 uses cross-fitting, n_folds=1 fits in-sample.

    psi_i = mu1 - mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e);  ATE = mean(psi);
    SE = sd(psi)/sqrt(n). Propensity is clipped to [clip, 1-clip]; the clipped fraction
    is reported in diagnostics so positivity problems are visible.
    """
    om = LinearRegression() if outcome_model is None else outcome_model
    pm = LogisticRegression(max_iter=2000) if propensity_model is None else propensity_model
    X, t, y = ds.X, ds.treatment, ds.outcome
    n = len(y)
    mu1, mu0, e = np.zeros(n), np.zeros(n), np.zeros(n)
    if n_folds >= 2:
        splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, t)
    else:
        splits = [(np.arange(n), np.arange(n))]
    for tr, te in splits:
        m1 = clone(om).fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1])
        m0 = clone(om).fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0])
        mu1[te], mu0[te] = m1.predict(X[te]), m0.predict(X[te])
        e[te] = clone(pm).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    frac = float(np.mean((e < clip) | (e > 1 - clip)))
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    est, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n))
    res = EstimateResult(
        "aipw", "ATE", est, se, normal_ci(est, se), {"clipped_fraction": frac, "n_folds": n_folds}
    )
    return (res, psi) if return_psi else res
