"""Synthetic panel (DiD) and instrument (IV) generators with exact truth."""

from __future__ import annotations

import numpy as np


def make_panel(
    n_units=400, periods=6, treat_start=3, effect=2.0, trend_violation=0.0, seed=0
) -> dict:
    """y_it = alpha_i + 0.3 t + violation*group*t + effect*group*1{t>=start} + eps.
    trend_violation != 0 breaks parallel trends. true ATT = effect."""
    rng = np.random.default_rng(seed)
    group = rng.binomial(1, 0.5, n_units)
    alpha = rng.normal(size=n_units) + 0.5 * group
    unit = np.repeat(np.arange(n_units), periods)
    t = np.tile(np.arange(periods), n_units)
    g = group[unit]
    post = (t >= treat_start).astype(int)
    y = (
        alpha[unit]
        + 0.3 * t
        + trend_violation * g * t
        + effect * g * post
        + rng.normal(size=len(t))
    )
    return dict(
        unit=unit, group=g, period=t, post=post, y=y, true_att=effect, treat_start=treat_start
    )


def make_iv(
    n=2000, seed=0, strength=1.0, tau=1.0, confounding=1.0, exclusion_violation=0.0
) -> dict:
    """D = strength*Z + confounding*U + v ; Y = tau*D + confounding*U + viol*Z + e.
    OLS of Y on D is biased by U; valid IV (viol=0) recovers tau."""
    rng = np.random.default_rng(seed)
    Z, U = rng.normal(size=n), rng.normal(size=n)
    D = strength * Z + confounding * U + rng.normal(size=n)
    Y = tau * D + confounding * U + exclusion_violation * Z + rng.normal(size=n)
    return dict(y=Y, d=D, z=Z, true_effect=tau)
