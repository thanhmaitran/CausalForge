"""Core result types shared by all estimators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EstimateResult:
    """Result of a point/interval estimate of a causal parameter."""

    name: str
    estimand: str
    estimate: float
    se: float | None = None
    ci: tuple[float, float] | None = None
    diagnostics: dict = field(default_factory=dict)

    def covers(self, truth: float) -> bool | None:
        if self.ci is None:
            return None
        return bool(self.ci[0] <= truth <= self.ci[1])


def normal_ci(est: float, se: float, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    return (float(est - z * se), float(est + z * se))


def as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()
