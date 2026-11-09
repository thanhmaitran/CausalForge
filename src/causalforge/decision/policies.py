"""Budget-constrained treatment policies and policy value."""

from __future__ import annotations

import numpy as np


def top_k_policy(tau_hat: np.ndarray, budget: float, require_positive: bool = True) -> np.ndarray:
    """Treat the `budget` fraction with highest predicted CATE (optionally only if >0)."""
    n = len(tau_hat)
    k = int(np.floor(budget * n))
    pi = np.zeros(n, int)
    if k > 0:
        pi[np.argsort(-tau_hat)[:k]] = 1
    if require_positive:
        pi[tau_hat <= 0] = 0
    return pi


def true_policy_value(ds, pi: np.ndarray) -> float:
    """Exact V(pi)=E[Y(pi)] from simulated potential outcomes (ground truth only)."""
    return float(np.mean(np.where(pi == 1, ds.y1, ds.y0)))


def ipw_policy_value(t, y, pi, e) -> float:
    """IPW estimate of V(pi) = E[1{T=pi(X)} Y / P(T=pi(X)|X)]; valid when e is known
    (randomized) or consistently estimated, plus consistency & ignorability."""
    p_match = np.where(pi == 1, e, 1 - e)
    return float(np.mean((t == pi) * y / p_match))


def aipw_policy_value(t, y, pi, e, mu1, mu0) -> float:
    """Doubly robust policy value estimate."""
    mu_pi = np.where(pi == 1, mu1, mu0)
    p_match = np.where(pi == 1, e, 1 - e)
    return float(np.mean(mu_pi + (t == pi) * (y - mu_pi) / p_match))
