"""Uplift / Qini evaluation for randomized data (e.g. Criteo, assignment as treatment)."""

from __future__ import annotations

import numpy as np


def qini_curve(score, t, y, n_points: int = 100):
    """Qini(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k) over the top-k units by score."""
    order = np.argsort(-np.asarray(score))
    t, y = np.asarray(t)[order], np.asarray(y, float)[order]
    n = len(y)
    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    ct, cy = np.cumsum(t), np.cumsum(y * t)
    cc, cyc = np.cumsum(1 - t), np.cumsum(y * (1 - t))
    q = np.array(
        [cy[k - 1] - (cyc[k - 1] * ct[k - 1] / cc[k - 1] if cc[k - 1] > 0 else 0) for k in ks]
    )
    return ks / n, q


def auuc(score, t, y, n_points: int = 100) -> float:
    """Area between the Qini curve and the random-targeting diagonal (normalised by n)."""
    f, q = qini_curve(score, t, y, n_points)
    random_line = q[-1] * f
    return float(np.trapezoid(q - random_line, f))
