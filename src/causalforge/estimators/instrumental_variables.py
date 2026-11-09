"""Two-stage least squares with robust SEs and weak-instrument diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from causalforge.core.types import EstimateResult, normal_ci


def tsls(y, d, Z, X=None) -> EstimateResult:
    """Y = b0 + b1*D + X g + e, instrumenting D with Z (n x q).

    Diagnostics: first-stage F for the excluded instruments (rule of thumb F<10 = weak).
    Relevance is testable; the exclusion restriction and independence are NOT testable
    and must be argued from domain knowledge. With a binary treatment and heterogeneous
    effects the estimate is a LATE (requires monotonicity).
    """
    y, d = np.asarray(y, float), np.asarray(d, float)
    n = len(y)
    Z = np.asarray(Z, float).reshape(n, -1)
    Xc = np.empty((n, 0)) if X is None else np.asarray(X, float).reshape(n, -1)
    Xm = np.column_stack([np.ones(n), d, Xc])
    Zm = np.column_stack([np.ones(n), Z, Xc])
    Xhat = Zm @ np.linalg.lstsq(Zm, Xm, rcond=None)[0]
    beta = np.linalg.solve(Xhat.T @ Xm, Xhat.T @ y)
    u = y - Xm @ beta  # residuals use the ACTUAL treatment, not fitted values
    A = np.linalg.inv(Xhat.T @ Xm)
    V = A @ (Xhat.T * u**2) @ Xhat @ A.T * n / (n - Xm.shape[1])
    se = float(np.sqrt(V[1, 1]))
    # first-stage F for excluded instruments
    r_u = d - Zm @ np.linalg.lstsq(Zm, d, rcond=None)[0]
    Zr = np.column_stack([np.ones(n), Xc])
    r_r = d - Zr @ np.linalg.lstsq(Zr, d, rcond=None)[0]
    q = Z.shape[1]
    F = float(((r_r @ r_r - r_u @ r_u) / q) / (r_u @ r_u / (n - Zm.shape[1])))
    diag = {
        "first_stage_F": F,
        "weak_instrument": F < 10,
        "n_instruments": q,
        "first_stage_F_pvalue": float(1 - stats.f.cdf(F, q, n - Zm.shape[1])),
    }
    return EstimateResult(
        "tsls", "LATE/ATE (needs IV assumptions)", float(beta[1]), se, normal_ci(beta[1], se), diag
    )
