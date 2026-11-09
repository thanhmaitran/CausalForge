"""Propensity score estimation and diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def estimate_propensity(X, t, C: float = 1.0):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(X, t)
    return model, model.predict_proba(X)[:, 1]


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w**2).sum())


def standardized_mean_difference(X, t, w=None) -> np.ndarray:
    """Weighted SMD per column: (mean1-mean0)/sqrt((var1+var0)/2)."""
    X = np.asarray(X, float)
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    out = []
    for j in range(X.shape[1]):
        m, v = [], []
        for g in (1, 0):
            mask = t == g
            wg = w[mask]
            mu = np.average(X[mask, j], weights=wg)
            var = np.average((X[mask, j] - mu) ** 2, weights=wg)
            m.append(mu)
            v.append(var)
        pooled = np.sqrt((v[0] + v[1]) / 2)
        out.append(0.0 if pooled == 0 else (m[0] - m[1]) / pooled)
    return np.array(out)


def overlap_report(e: np.ndarray, t: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> dict:
    """Overlap diagnostics. Reports, does not silently fix, positivity problems."""
    return {
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "frac_extreme": float(np.mean((e < lo) | (e > hi))),
        "common_support": [
            float(max(e[t == 1].min(), e[t == 0].min())),
            float(min(e[t == 1].max(), e[t == 0].max())),
        ],
        "ess_treated": effective_sample_size(1 / e[t == 1]),
        "ess_control": effective_sample_size(1 / (1 - e[t == 0])),
    }
