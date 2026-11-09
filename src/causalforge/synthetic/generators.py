"""Synthetic structural causal models with exact ground truth.

Ground truth is computed from the structural equations (never from an estimator).

    X ~ N(0, I_d),  U ~ N(0, 1)           (U hidden, not part of X)
    logit P(T=1|X,U) = confounding * (0.8 X0 - 0.5 X1) + hidden_t * U   [0.5 if randomized]
    Y(0) = m(X) + hidden_y * U + eps,   Y(1) = Y(0) + tau(X)
    ITE_i = tau(X_i)  =>  true ATE = mean(tau(X_i))  (sample ATE)
"""

from __future__ import annotations

import numpy as np

from causalforge.data.schemas import CausalDataset


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_scm(
    n: int = 1000,
    seed: int = 0,
    d: int = 5,
    confounding: float = 1.0,
    hidden_t: float = 0.0,
    hidden_y: float = 0.0,
    nonlinear: bool = False,
    heterogeneous: bool = True,
    randomized: bool = False,
    tau: float = 1.0,
    noise: float = 1.0,
    keep_hidden: bool = False,
) -> CausalDataset:
    """Generate a synthetic dataset. `keep_hidden=True` appends U as a covariate (oracle)."""
    if d < 3:
        raise ValueError("d must be >= 3")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    U = rng.normal(size=n)

    if randomized:
        e = np.full(n, 0.5)
    else:
        e = _sigmoid(confounding * (0.8 * X[:, 0] - 0.5 * X[:, 1]) + hidden_t * U)
        e = np.clip(e, 1e-3, 1 - 1e-3)
    T = rng.binomial(1, e).astype(int)

    if nonlinear:
        m = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 0]
        tau_x = tau + ((1.0 * (X[:, 0] > 0) - 0.5) if heterogeneous else 0.0)
    else:
        m = X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2]
        tau_x = tau + (0.5 * X[:, 0] if heterogeneous else 0.0)
    tau_x = np.asarray(tau_x) * np.ones(n)

    eps = rng.normal(scale=noise, size=n)
    y0 = m + hidden_y * U + eps
    y1 = y0 + tau_x
    Y = np.where(T == 1, y1, y0)

    names = [f"x{i}" for i in range(d)]
    Xo = X
    if keep_hidden:
        Xo = np.column_stack([X, U])
        names = names + ["U"]
    return CausalDataset(
        X=Xo,
        treatment=T,
        outcome=Y,
        feature_names=names,
        dataset_name="synthetic_scm",
        propensity=e,
        true_ite=tau_x,
        true_cate=tau_x.copy(),
        true_ate=float(tau_x.mean()),
        y0=y0,
        y1=y1,
        metadata=dict(
            seed=seed,
            n=n,
            d=d,
            confounding=confounding,
            hidden_t=hidden_t,
            hidden_y=hidden_y,
            nonlinear=nonlinear,
            heterogeneous=heterogeneous,
            randomized=randomized,
            tau=tau,
            noise=noise,
            U=U,
            dag_edges=[["X", "T"], ["X", "Y"], ["T", "Y"], ["U", "T"], ["U", "Y"]],
        ),
    )
