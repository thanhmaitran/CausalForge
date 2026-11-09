"""Statistical correctness: estimators recover truth across seeds within MC error."""

import numpy as np

from causalforge.causal_ml.dml import dml_plr
from causalforge.estimators.doubly_robust import aipw_ate
from causalforge.estimators.regression import naive_ate, regression_adjustment
from causalforge.estimators.weighting import ipw_ate
from causalforge.meta_learners.learners import TLearner, XLearner
from causalforge.synthetic.generators import make_scm


def _mc(fn, R=40, **kw):
    err, cov = [], []
    for s in range(R):
        ds = make_scm(1500, 1000 + s, **kw)
        r = fn(ds)
        err.append(r.estimate - ds.true_ate)
        c = r.covers(ds.true_ate)
        if c is not None:
            cov.append(c)
    return np.mean(err), (np.mean(cov) if cov else float("nan"))


def test_randomized_all_unbiased():
    for fn in (naive_ate, regression_adjustment, ipw_ate, aipw_ate):
        b, _ = _mc(fn, randomized=True)
        assert abs(b) < 0.08, fn


def test_naive_is_biased_under_confounding_adjusted_is_not():
    b_naive, _ = _mc(naive_ate)
    assert b_naive > 0.3
    for fn in (regression_adjustment, ipw_ate, aipw_ate):
        b, _ = _mc(fn)
        assert abs(b) < 0.08, fn


def test_aipw_coverage_reasonable():
    _, cov = _mc(aipw_ate, R=60)
    assert 0.85 <= cov <= 1.0


def test_hidden_confounding_biases_observed_adjustment():
    b, _ = _mc(aipw_ate, hidden_t=1.0, hidden_y=1.0, R=20)
    assert b > 0.4  # estimators cannot fix unobserved confounding


def test_zero_effect_scm_gives_zero_estimate():
    b, _ = _mc(aipw_ate, tau=0.0, heterogeneous=False, R=30)
    assert abs(b) < 0.08


def test_row_permutation_invariance():
    ds = make_scm(1000, 5)
    perm = np.random.default_rng(0).permutation(ds.n)
    a, b = regression_adjustment(ds).estimate, regression_adjustment(ds.subset(perm)).estimate
    assert abs(a - b) < 1e-8


def test_meta_learners_recover_heterogeneity():
    ds = make_scm(3000, 11, randomized=True)
    for L in (TLearner, XLearner):
        p = L().fit(ds.X, ds.treatment, ds.outcome).predict_cate(ds.X)
        assert np.corrcoef(p, ds.true_cate)[0, 1] > 0.5


def test_dml_recovers_constant_effect():
    errs = [dml_plr(make_scm(2000, 50 + s, heterogeneous=False)).estimate - 1.0 for s in range(8)]
    assert abs(np.mean(errs)) < 0.1
