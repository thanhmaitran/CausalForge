"""Small OLS helpers with cluster-robust inference."""

from __future__ import annotations

import numpy as np


def ols(D: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return beta, y - D @ beta


def cluster_vcov(D: np.ndarray, resid: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """CR1 cluster-robust covariance of OLS coefficients."""
    n, k = D.shape
    bread = np.linalg.inv(D.T @ D)
    ids, inv = np.unique(clusters, return_inverse=True)
    G = len(ids)
    scores = np.zeros((G, k))
    np.add.at(scores, inv, D * resid[:, None])
    meat = scores.T @ scores
    c = G / (G - 1) * (n - 1) / (n - k)
    return c * bread @ meat @ bread
