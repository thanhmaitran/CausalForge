"""Inverse probability weighting (Horvitz-Thompson and Hajek forms)."""

from __future__ import annotations

import numpy as np

from causalforge.core.types import EstimateResult, normal_ci
from causalforge.estimators.propensity import (
    effective_sample_size,
    estimate_propensity,
    overlap_report,
)


def ipw_ate(
    ds, propensity=None, normalized: bool = True, trim: float | None = None
) -> EstimateResult:
    """IPW ATE.

    normalized=True : Hajek (self-normalised) estimator; False : Horvitz-Thompson.
    trim: if given, units with e outside [trim, 1-trim] are DROPPED (changes the estimand
          to the trimmed population; reported in diagnostics) - never silent clipping.
    SE: influence-function SE treating e-hat as known (ignores propensity estimation error).
    """
    t, y = ds.treatment.astype(float), ds.outcome
    e = estimate_propensity(ds.X, ds.treatment)[1] if propensity is None else propensity
    keep = np.ones(len(y), bool) if trim is None else (e >= trim) & (e <= 1 - trim)
    t, y, e = t[keep], y[keep], e[keep]
    n = len(y)
    w1, w0 = t / e, (1 - t) / (1 - e)
    if normalized:
        mu1, mu0 = (w1 * y).sum() / w1.sum(), (w0 * y).sum() / w0.sum()
        psi = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean() + (mu1 - mu0)
    else:
        mu1, mu0 = (w1 * y).mean(), (w0 * y).mean()
        psi = w1 * y - w0 * y
    est = float(mu1 - mu0)
    se = (
        float(np.std(psi - psi.mean(), ddof=1) / np.sqrt(n))
        if normalized
        else float(np.std(psi, ddof=1) / np.sqrt(n))
    )
    diag = overlap_report(e, ds.treatment[keep]) | {
        "n_used": int(n),
        "n_dropped": int((~keep).sum()),
        "ess_total": effective_sample_size(np.where(t == 1, 1 / e, 1 / (1 - e))),
        "max_weight": float(np.where(t == 1, 1 / e, 1 / (1 - e)).max()),
    }
    return EstimateResult(
        "ipw_hajek" if normalized else "ipw_ht", "ATE", est, se, normal_ci(est, se), diag
    )


def ipw_att(ds, propensity=None) -> EstimateResult:
    """IPW for the ATT: treated unweighted, controls weighted by e/(1-e). SE by influence
    function treating e-hat as known (approximate)."""
    t, y = ds.treatment, ds.outcome
    e = estimate_propensity(ds.X, t)[1] if propensity is None else propensity
    e = np.clip(e, 1e-6, 1 - 1e-6)
    w = np.where(t == 0, e / (1 - e), 0.0)
    mu1 = y[t == 1].mean()
    mu0 = (w * y).sum() / w.sum()
    n1 = (t == 1).sum()
    psi = np.where(t == 1, y - mu1, 0) / n1 * len(y) - w * (y - mu0) / w.sum() * len(y)
    se = float(np.sqrt(np.sum(psi**2)) / len(y))
    est = float(mu1 - mu0)
    return EstimateResult(
        "ipw_att",
        "ATT",
        est,
        se,
        normal_ci(est, se),
        {"ess_controls": effective_sample_size(w[t == 0])},
    )
