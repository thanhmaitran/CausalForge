"""Metrics matched to causal estimands."""

from __future__ import annotations

import numpy as np


def ate_error(est: float, truth: float) -> float:
    return abs(est - truth)


def rmse(est, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(truth)) ** 2)))


def bias(est, truth) -> float:
    return float(np.mean(np.asarray(est) - np.asarray(truth)))


def pehe(tau_hat, tau_true) -> float:
    """sqrt-PEHE; only valid when true ITE/CATE is known (simulation / semi-synthetic)."""
    return rmse(tau_hat, tau_true)


def coverage(cis, truths) -> float:
    ok = [lo <= tr <= hi for (lo, hi), tr in zip(cis, truths) if lo == lo]
    return float(np.mean(ok)) if ok else float("nan")
