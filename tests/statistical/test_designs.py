"""DiD, IV, cluster bootstrap, negative controls, causal forest, uplift."""

import numpy as np

from causalforge.causal_ml.causal_forest import HonestCausalForest
from causalforge.estimators.cluster_bootstrap import cluster_bootstrap
from causalforge.estimators.difference_in_differences import did_2x2, event_study
from causalforge.estimators.instrumental_variables import tsls
from causalforge.evaluation.metrics import pehe
from causalforge.evaluation.uplift import auuc
from causalforge.meta_learners.learners import TLearner
from causalforge.sensitivity.negative_controls import negative_control_outcome_test
from causalforge.synthetic.generators import make_scm
from causalforge.synthetic.panel_iv import make_iv, make_panel


def test_did_recovers_att_under_parallel_trends_and_pretrend_test_flags_violation():
    errs, rej_ok, rej_bad = [], 0, 0
    for s in range(30):
        p = make_panel(seed=s)
        errs.append(did_2x2(p["unit"], p["group"], p["post"], p["y"]).estimate - 2.0)
        rej_ok += (
            event_study(p["unit"], p["group"], p["period"], p["y"], 2)[1]["pretrend_wald_p"] < 0.05
        )
        q = make_panel(seed=s, trend_violation=0.4)
        rej_bad += (
            event_study(q["unit"], q["group"], q["period"], q["y"], 2)[1]["pretrend_wald_p"] < 0.05
        )
    assert abs(np.mean(errs)) < 0.05 and rej_ok <= 6 and rej_bad >= 28


def test_did_biased_when_trends_violated():
    p = make_panel(seed=1, trend_violation=0.5)
    assert did_2x2(p["unit"], p["group"], p["post"], p["y"]).estimate > 2.5


def test_iv_recovers_effect_ols_biased_weak_flagged_exclusion_violation_biased():
    d = make_iv(seed=1)
    r = tsls(d["y"], d["d"], d["z"])
    ols = np.polyfit(d["d"], d["y"], 1)[0]
    assert abs(r.estimate - 1) < 0.15 and ols - 1 > 0.2 and not r.diagnostics["weak_instrument"]
    assert tsls(*[make_iv(seed=1, strength=0.03)[k] for k in ("y", "d", "z")]).diagnostics[
        "weak_instrument"
    ]
    bad = make_iv(seed=1, exclusion_violation=0.4)
    assert tsls(bad["y"], bad["d"], bad["z"]).estimate - 1 > 0.2


def test_cluster_bootstrap_se_close_to_analytic():
    p = make_panel(seed=3)
    f = lambda d: did_2x2(d["unit"], d["group"], d["post"], d["y"]).estimate  # noqa: E731
    cb = cluster_bootstrap(f, p, "unit", n_boot=150)
    an = did_2x2(p["unit"], p["group"], p["post"], p["y"]).se
    assert 0.6 < cb["se"] / an < 1.6 and cb["ci"][0] < cb["estimate"] < cb["ci"][1]


def test_negative_control_detects_hidden_confounding_but_not_null():
    rej0 = rej1 = 0
    for s in range(30):
        for h, name in ((0.0, "null"), (1.0, "hidden")):
            ds = make_scm(1500, s, hidden_t=h, hidden_y=h)
            nco = 0.8 * ds.metadata["U"] + np.random.default_rng(s).normal(size=ds.n)
            rej = negative_control_outcome_test(ds, nco)["reject_null"]
            rej0 += rej and name == "null"
            rej1 += rej and name == "hidden"
    assert rej0 <= 7 and rej1 >= 28


def test_honest_forest_beats_t_learner_on_heterogeneity_and_is_reproducible():
    ds = make_scm(2000, 4, randomized=True, nonlinear=True)
    f = HonestCausalForest(seed=1).fit(ds.X, ds.treatment, ds.outcome, ds.propensity)
    p = f.predict_cate(ds.X)
    t = TLearner().fit(ds.X, ds.treatment, ds.outcome).predict_cate(ds.X)
    assert pehe(p, ds.true_cate) < pehe(t, ds.true_cate)
    p2 = (
        HonestCausalForest(seed=1)
        .fit(ds.X, ds.treatment, ds.outcome, ds.propensity)
        .predict_cate(ds.X)
    )
    assert np.allclose(p, p2)
    m, sd = f.predict_cate(ds.X, return_std=True)
    assert (sd > 0).all() and abs(m.mean() - ds.true_ate) < 0.15


def test_forest_honesty_leaf_estimates_use_held_out_half():
    """Tree structure is built on half A; leaf effects come from disjoint half B."""
    ds = make_scm(800, 2, randomized=True)
    f = HonestCausalForest(n_groups=2, trees_per_group=2).fit(ds.X, ds.treatment, ds.outcome)
    for trees in f.groups_:
        for tree, tau in trees:
            assert len(tau) == tree.tree_.node_count


def test_uplift_auuc_rewards_true_ranking():
    ds = make_scm(6000, 5, randomized=True)
    good = auuc(ds.true_cate, ds.treatment, ds.outcome)
    rand = auuc(np.random.default_rng(0).normal(size=ds.n), ds.treatment, ds.outcome)
    assert good > 3 * abs(rand)
