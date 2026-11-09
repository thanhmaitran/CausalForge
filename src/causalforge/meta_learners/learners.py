"""S-, T- and X-learners for CATE. Interface: fit(X, t, y) -> self; predict_cate(X)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def default_base():
    return HistGradientBoostingRegressor(
        max_iter=80, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=20, random_state=0
    )


class SLearner:
    name = "s_learner"

    def __init__(self, base=None):
        self.base = default_base() if base is None else base

    def fit(self, X, t, y):
        self.m_ = clone(self.base).fit(np.column_stack([X, t]), y)
        return self

    def predict_cate(self, X):
        n = len(X)
        return self.m_.predict(np.column_stack([X, np.ones(n)])) - self.m_.predict(
            np.column_stack([X, np.zeros(n)])
        )


class TLearner:
    name = "t_learner"

    def __init__(self, base=None):
        self.base = default_base() if base is None else base

    def fit(self, X, t, y):
        self.m1_ = clone(self.base).fit(X[t == 1], y[t == 1])
        self.m0_ = clone(self.base).fit(X[t == 0], y[t == 0])
        return self

    def predict_cate(self, X):
        return self.m1_.predict(X) - self.m0_.predict(X)


class XLearner:
    """Kunzel et al. (2019): impute ITEs per arm, model them, blend with propensity."""

    name = "x_learner"

    def __init__(self, base=None, effect_base=None):
        self.base = default_base() if base is None else base
        self.effect_base = self.base if effect_base is None else effect_base

    def fit(self, X, t, y):
        X1, X0, y1, y0 = X[t == 1], X[t == 0], y[t == 1], y[t == 0]
        m1 = clone(self.base).fit(X1, y1)
        m0 = clone(self.base).fit(X0, y0)
        d1 = y1 - m0.predict(X1)  # imputed effects, treated
        d0 = m1.predict(X0) - y0  # imputed effects, control
        self.tau1_ = clone(self.effect_base).fit(X1, d1)
        self.tau0_ = clone(self.effect_base).fit(X0, d0)
        self.prop_ = LogisticRegression(max_iter=2000).fit(X, t)
        return self

    def predict_cate(self, X):
        g = np.clip(self.prop_.predict_proba(X)[:, 1], 0.01, 0.99)
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)
