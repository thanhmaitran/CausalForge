import numpy as np
import pytest

from causalforge.causal_ml.dml import dml_plr
from causalforge.data.schemas import CausalDataset
from causalforge.data.validators.causal_validator import validate_dataset
from causalforge.decision.policies import ipw_policy_value, top_k_policy, true_policy_value
from causalforge.estimators.matching import psm
from causalforge.estimators.propensity import effective_sample_size, standardized_mean_difference
from causalforge.graphs.dag import CausalDAG
from causalforge.sensitivity.partial_r2 import robustness_value
from causalforge.synthetic.generators import make_scm


def _dag():
    g = CausalDAG()
    for n, r in [("X", "confounder"), ("T", "treatment"), ("Y", "outcome")]:
        g.add_variable(n, r)
    g.add_causal_edge("X", "T")
    g.add_causal_edge("X", "Y")
    g.add_causal_edge("T", "Y")
    return g


def test_dag_cycle_detection():
    g = _dag()
    g.add_causal_edge("Y", "X")
    with pytest.raises(ValueError):
        g.validate()


def test_backdoor_confounder():
    g = _dag()
    assert g.backdoor_paths("T", "Y") == [["T", "X", "Y"]]
    assert g.adjustment_sets("T", "Y") == [{"X"}]
    assert not g.is_valid_adjustment_set(set(), "T", "Y")


def test_mediator_and_collider_not_adjusted():
    g = _dag()
    g.add_variable("M", "mediator")
    g.add_variable("C", "collider")
    g.graph.remove_edge("T", "Y")
    g.add_causal_edge("T", "M")
    g.add_causal_edge("M", "Y")
    g.add_causal_edge("T", "C")
    g.add_causal_edge("Y", "C")
    assert not g.is_valid_adjustment_set({"X", "M"}, "T", "Y")  # descendant of T
    assert not g.is_valid_adjustment_set({"X", "C"}, "T", "Y")  # collider/descendant
    assert g.is_valid_adjustment_set({"X"}, "T", "Y")
    assert g.mediators("T", "Y") == {"M"}


def test_unrelated_node_does_not_change_adjustment_sets():
    g = _dag()
    before = g.adjustment_sets("T", "Y")
    g.add_variable("Z", "covariate")
    after = g.adjustment_sets("T", "Y")
    assert before == [s for s in after if "Z" not in s]


def test_dag_json_roundtrip():
    g = _dag()
    g2 = CausalDAG.from_json(g.to_json())
    assert set(g2.graph.edges) == set(g.graph.edges)


def test_schema_validation():
    with pytest.raises(ValueError):
        CausalDataset(np.zeros((3, 2)), np.array([0, 1, 2]), np.zeros(3), ["a", "b"], "x")
    with pytest.raises(ValueError):
        CausalDataset(np.zeros((3, 2)), np.array([0, 1, 0]), np.zeros(3), ["a"], "x")


def test_ground_truth_from_structural_equations():
    ds = make_scm(2000, 1)
    assert np.isclose(ds.true_ate, np.mean(ds.y1 - ds.y0))
    assert np.allclose(ds.outcome, np.where(ds.treatment == 1, ds.y1, ds.y0))


def test_zero_effect_scm_has_zero_truth():
    ds = make_scm(500, 2, tau=0.0, heterogeneous=False)
    assert ds.true_ate == 0.0


def test_generator_is_reproducible():
    a, b = make_scm(300, 7), make_scm(300, 7)
    assert np.array_equal(a.outcome, b.outcome)


def test_randomized_assignment_balances_covariates():
    ds = make_scm(20000, 3, randomized=True)
    assert np.abs(standardized_mean_difference(ds.X, ds.treatment)).max() < 0.05


def test_psm_att_vs_ate_labels():
    ds = make_scm(1000, 4)
    assert psm(ds, "ATT").estimand == "ATT" and psm(ds, "ATE").estimand == "ATE"
    with pytest.raises(ValueError):
        psm(ds, "LATE")


def test_ess_bounds():
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert effective_sample_size(w) == 4
    assert effective_sample_size(np.array([10.0, 1, 1, 1])) < 4


def test_robustness_value_properties():
    assert robustness_value(0.0, 100) == 0.0
    assert 0 < robustness_value(5.0, 200) < 1
    assert robustness_value(5.0, 200, alpha=0.05) < robustness_value(5.0, 200)


def test_policy_budget_and_values():
    ds = make_scm(2000, 5, randomized=True)
    pi = top_k_policy(ds.true_cate, 0.3)
    assert pi.sum() <= int(0.3 * len(pi))
    assert true_policy_value(ds, pi) >= true_policy_value(ds, np.zeros(ds.n, int))
    est = ipw_policy_value(ds.treatment, ds.outcome, pi, ds.propensity)
    assert abs(est - true_policy_value(ds, pi)) < 0.3


def test_validator_flags_leakage_and_imbalance():
    ds = make_scm(1000, 6)
    ds.feature_names[0] = "exposure"
    rep = validate_dataset(ds, post_treatment=["exposure"])
    assert any("leakage" in w for w in rep["warnings"])
    assert rep["overlap"]["e_min"] > 0


def test_dml_residuals_are_out_of_fold(monkeypatch):
    """Cross-fitting: no nuisance model is ever evaluated on rows it was trained on."""
    seen = []
    from sklearn.linear_model import LinearRegression

    class Spy(LinearRegression):
        def fit(self, X, y):
            self._train = {tuple(r) for r in X[:, :3].round(6)}
            return super().fit(X, y)

        def predict(self, X):
            seen.append(len(self._train & {tuple(r) for r in X[:, :3].round(6)}))
            return super().predict(X)

    dml_plr(make_scm(500, 8), Spy(), Spy())
    assert sum(seen) == 0
