"""Dataset validation and data-quality report for causal analyses."""

from __future__ import annotations

import numpy as np

from causalforge.estimators.propensity import (
    estimate_propensity,
    overlap_report,
    standardized_mean_difference,
)


def validate_dataset(ds, post_treatment: list[str] | None = None) -> dict:
    """Return a data-quality report; raises ValueError only on hard failures.

    `post_treatment` lists columns known to be post-treatment (e.g. Criteo `exposure`);
    if any appear among covariates a leakage warning is raised.
    """
    if ds.n == 0:
        raise ValueError("dataset has zero rows")
    t, y = ds.treatment, ds.outcome
    warnings: list[str] = []
    leak = sorted(set(post_treatment or []) & set(ds.feature_names))
    if leak:
        warnings.append(f"possible leakage: post-treatment columns used as covariates: {leak}")
    if t.min() == t.max():
        raise ValueError("treatment has a single value")
    binary_y = set(np.unique(y)) <= {0, 1}
    e = estimate_propensity(ds.X, t)[1]
    smd = standardized_mean_difference(ds.X, t)
    ov = overlap_report(e, t)
    if ov["frac_extreme"] > 0.05:
        warnings.append(
            f"{ov['frac_extreme']:.1%} of units have extreme propensity (<0.05 or >0.95)"
        )
    if np.abs(smd).max() > 0.1:
        warnings.append("covariate imbalance (|SMD|>0.1) before adjustment")
    return {
        "n_rows": int(ds.n),
        "n_features": int(ds.X.shape[1]),
        "missing_by_column": {
            n: int(np.isnan(ds.X[:, j]).sum()) for j, n in enumerate(ds.feature_names)
        },
        "missing_outcome": int(np.isnan(y).sum()),
        "treatment_ratio": float(t.mean()),
        "outcome_type": "binary" if binary_y else "continuous",
        "outcome_mean_treated": float(y[t == 1].mean()),
        "outcome_mean_control": float(y[t == 0].mean()),
        "duplicate_rows": int(ds.n - len(np.unique(np.column_stack([ds.X, t, y]), axis=0))),
        "smd_max_abs": float(np.abs(smd).max()),
        "smd": {n: float(s) for n, s in zip(ds.feature_names, smd)},
        "overlap": ov,
        "warnings": warnings,
    }
