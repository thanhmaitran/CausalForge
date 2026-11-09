"""Nearest-neighbour propensity score matching (with replacement, optional caliper)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from causalforge.core.types import EstimateResult
from causalforge.estimators.propensity import estimate_propensity


def psm(ds, estimand: str = "ATT", caliper: float | None = None, propensity=None) -> EstimateResult:
    """1-NN matching on the logit propensity score, with replacement.

    estimand='ATT': match each treated unit to a control.
    estimand='ATE': impute the missing potential outcome for every unit.
    No analytic SE is reported: the plain bootstrap is invalid for NN matching
    (Abadie & Imbens 2008), so coverage is deliberately not claimed.
    """
    t, y = ds.treatment, ds.outcome
    e = estimate_propensity(ds.X, t)[1] if propensity is None else propensity
    e = np.clip(e, 1e-6, 1 - 1e-6)
    score = np.log(e / (1 - e)).reshape(-1, 1)

    def match(src: np.ndarray, dst: np.ndarray):
        nn = NearestNeighbors(n_neighbors=1).fit(score[dst])
        dist, j = nn.kneighbors(score[src])
        return dist.ravel(), dst[j.ravel()]

    i1, i0 = np.where(t == 1)[0], np.where(t == 0)[0]
    d1, m1 = match(i1, i0)  # treated -> control
    keep1 = np.ones(len(i1), bool) if caliper is None else d1 <= caliper
    effects = (y[i1] - y[m1])[keep1]
    n_unmatched = int((~keep1).sum())
    if estimand == "ATE":
        d0, m0 = match(i0, i1)  # control -> treated
        keep0 = np.ones(len(i0), bool) if caliper is None else d0 <= caliper
        effects = np.concatenate([effects, (y[m0] - y[i0])[keep0]])
        n_unmatched += int((~keep0).sum())
    elif estimand != "ATT":
        raise ValueError("estimand must be 'ATT' or 'ATE'")
    return EstimateResult(
        "psm",
        estimand,
        float(effects.mean()),
        None,
        None,
        {
            "n_matched": int(len(effects)),
            "n_unmatched": n_unmatched,
            "mean_match_distance": float(d1.mean()),
            "controls_reused": int(len(i1) - len(np.unique(m1))),
        },
    )
