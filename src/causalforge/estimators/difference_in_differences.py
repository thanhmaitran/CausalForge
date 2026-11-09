"""Difference-in-differences with cluster-robust SEs, event study and pre-trend test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from causalforge.core.linalg import cluster_vcov, ols
from causalforge.core.types import EstimateResult


def did_2x2(unit, group, post, y, X=None, level: float = 0.95) -> EstimateResult:
    """y = a + b1*group + b2*post + b3*group*post (+ Xg) + e ; DiD = b3 (an ATT under
    parallel trends). SEs clustered by unit; t critical value with G-1 d.o.f."""
    unit, group, post, y = map(np.asarray, (unit, group, post, y))
    cols = [np.ones(len(y)), group, post, group * post]
    if X is not None:
        cols += list(np.asarray(X, float).reshape(len(y), -1).T)
    D = np.column_stack(cols).astype(float)
    beta, res = ols(D, y.astype(float))
    V = cluster_vcov(D, res, unit)
    se = float(np.sqrt(V[3, 3]))
    G = len(np.unique(unit))
    q = stats.t.ppf(0.5 + level / 2, G - 1)
    return EstimateResult(
        "did",
        "ATT",
        float(beta[3]),
        se,
        (float(beta[3] - q * se), float(beta[3] + q * se)),
        {"n_clusters": G, "identifying_assumption": "parallel trends"},
    )


def event_study(unit, group, period, y, ref_period) -> tuple[pd.DataFrame, dict]:
    """Group x period interactions relative to `ref_period`; Wald test that all PRE-period
    interactions (period < ref_period) are jointly zero (a diagnostic, not a proof, of
    parallel trends)."""
    unit, group, period, y = map(np.asarray, (unit, group, period, y))
    periods = [p for p in np.unique(period) if p != ref_period]
    P = np.column_stack([(period == p).astype(float) for p in periods])
    D = np.column_stack([np.ones(len(y)), group, P, P * group[:, None]]).astype(float)
    beta, res = ols(D, y.astype(float))
    V = cluster_vcov(D, res, unit)
    k0 = 2 + len(periods)
    b, se = beta[k0:], np.sqrt(np.diag(V)[k0:])
    tab = pd.DataFrame(
        {"period": periods, "coef": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se}
    )
    pre = [i for i, p in enumerate(periods) if p < ref_period]
    out = {"pretrend_wald_p": np.nan, "n_pre": len(pre)}
    if pre:
        idx = np.array(pre)
        bw, Vw = b[idx], V[k0:, k0:][np.ix_(idx, idx)]
        w = float(bw @ np.linalg.solve(Vw, bw))
        out["pretrend_wald_p"] = float(1 - stats.chi2.cdf(w, len(idx)))
    return tab, out
