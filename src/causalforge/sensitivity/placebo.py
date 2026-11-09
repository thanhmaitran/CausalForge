"""Permutation placebo-treatment test."""

from __future__ import annotations

import numpy as np


def placebo_treatment_test(ds, estimator, n_perm: int = 100, seed: int = 0) -> dict:
    """Permute treatment labels (breaking any T-Y link by construction) and re-estimate.

    A well-calibrated estimator should reject (|est|/se > 1.96) in about 5% of runs.
    A passing placebo does NOT prove absence of confounding.
    """
    rng = np.random.default_rng(seed)
    ests, rej = [], []
    for _ in range(n_perm):
        p = ds.subset(np.arange(ds.n))
        p.treatment = rng.permutation(ds.treatment)
        r = estimator(p)
        ests.append(r.estimate)
        rej.append(abs(r.estimate / r.se) > 1.96)
    return {
        "mean_placebo_estimate": float(np.mean(ests)),
        "sd": float(np.std(ests)),
        "rejection_rate": float(np.mean(rej)),
        "n_perm": n_perm,
    }
