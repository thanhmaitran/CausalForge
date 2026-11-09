"""Negative-control outcome check.

A negative control outcome (NCO) is a variable that, by domain knowledge, is NOT caused by the
treatment but shares the unmeasured confounders of the real outcome. A non-null estimated
'effect' on an NCO signals residual confounding. Choosing an arbitrary feature as an NCO is
invalid: the user must justify it.
"""

from __future__ import annotations

import numpy as np

from causalforge.estimators.doubly_robust import aipw_ate


def negative_control_outcome_test(ds, nco: np.ndarray, estimator=aipw_ate) -> dict:
    sub = ds.subset(np.arange(ds.n))
    sub.outcome = np.asarray(nco, float)
    r = estimator(sub)
    z = r.estimate / r.se if r.se else np.nan
    return {"estimate": r.estimate, "se": r.se, "z": float(z), "reject_null": bool(abs(z) > 1.96)}
