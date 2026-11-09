"""Cluster (unit-level) bootstrap for panel / clustered data."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap(
    fn, data: dict, cluster_key: str, n_boot: int = 500, seed: int = 0, level: float = 0.95
) -> dict:
    """Resample whole clusters with replacement; `fn(data_dict) -> float`."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(data[cluster_key])
    ids = np.unique(cl)
    rows = {i: np.where(cl == i)[0] for i in ids}
    est = fn(data)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([rows[i] for i in pick])
        vals.append(
            fn(
                {
                    k: np.asarray(v)[idx] if hasattr(v, "__len__") and len(v) == len(cl) else v
                    for k, v in data.items()
                }
            )
        )
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {
        "estimate": float(est),
        "se": float(np.std(vals, ddof=1)),
        "ci": (float(lo), float(hi)),
        "n_boot": n_boot,
    }
