"""Omitted-variable-bias sensitivity (Cinelli & Hazlett 2020) for OLS treatment effects."""

from __future__ import annotations

import numpy as np
from scipy import stats


def ols_treatment_summary(X, t, y) -> dict:
    """OLS of y on [1, t, X]; returns estimate, se, t-stat, dof, partial R2 of treatment."""
    n = len(y)
    D = np.column_stack([np.ones(n), t, X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    dof = n - D.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(D.T @ D)
    se = float(np.sqrt(cov[1, 1]))
    tstat = float(beta[1] / se)
    return {
        "estimate": float(beta[1]),
        "se": se,
        "t": tstat,
        "dof": int(dof),
        "partial_r2_yd": tstat**2 / (tstat**2 + dof),
    }


def robustness_value(tstat: float, dof: int, q: float = 1.0, alpha: float | None = None) -> float:
    """RV_q (alpha=None) or RV_{q,alpha}: the strength of association (partial R2) an
    unobserved confounder would need with BOTH treatment and outcome to reduce the estimate
    by 100q% (or make it insignificant at level alpha)."""
    f = abs(tstat) / np.sqrt(dof)
    fq = q * f
    if alpha is not None:
        fq -= stats.t.ppf(1 - alpha / 2, dof - 1) / np.sqrt(dof - 1)
    if fq <= 0:
        return 0.0
    return float(0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2))


def adjusted_estimate(estimate: float, se: float, dof: int, r2_dz: float, r2_yz: float) -> float:
    """Estimate after hypothetically removing bias from a confounder with partial R2
    r2_dz (with treatment) and r2_yz (with outcome), worst-case sign."""
    bias = np.sqrt(r2_yz * r2_dz / (1 - r2_dz)) * se * np.sqrt(dof)
    return float(estimate - np.sign(estimate) * bias)


def sensitivity_report(X, t, y, q: float = 1.0, alpha: float = 0.05) -> dict:
    s = ols_treatment_summary(X, t, y)
    return s | {
        "rv_q": robustness_value(s["t"], s["dof"], q),
        "rv_q_alpha": robustness_value(s["t"], s["dof"], q, alpha),
    }
