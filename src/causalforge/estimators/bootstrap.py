"""Generic i.i.d. bootstrap for any estimator f(ds) -> EstimateResult."""

from __future__ import annotations

import numpy as np

from causalforge.core.types import EstimateResult


def bootstrap_ci(fn, ds, n_boot: int = 500, seed: int = 0, level: float = 0.95) -> EstimateResult:
    """Percentile bootstrap resampling i.i.d. units. For clustered/panel data a
    cluster/block bootstrap is required instead (not implemented here)."""
    rng = np.random.default_rng(seed)
    base = fn(ds)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, ds.n, ds.n)
        sub = ds.subset(idx)
        if sub.treatment.min() == sub.treatment.max():
            continue
        vals.append(fn(sub).estimate)
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return EstimateResult(
        base.name,
        base.estimand,
        base.estimate,
        float(np.std(vals, ddof=1)),
        (float(lo), float(hi)),
        {"n_boot": len(vals)},
    )
