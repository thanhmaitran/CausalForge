"""Service layer: the ONLY place the API touches the causal engine. Pure Python, no web
framework, so it is unit-testable without FastAPI."""
from __future__ import annotations

import numpy as np

from causalforge.data.schemas import CausalDataset
from causalforge.data.validators.causal_validator import validate_dataset
from causalforge.decision.policies import top_k_policy, true_policy_value
from causalforge.estimators.doubly_robust import aipw_ate
from causalforge.estimators.matching import psm
from causalforge.estimators.regression import naive_ate, regression_adjustment
from causalforge.estimators.weighting import ipw_ate
from causalforge.graphs.dag import CausalDAG
from causalforge.sensitivity.partial_r2 import sensitivity_report
from causalforge.synthetic.generators import make_scm

ESTIMATORS = {
    "naive": naive_ate,
    "regression_adjustment": lambda ds: regression_adjustment(ds, n_boot=200),
    "ipw": ipw_ate,
    "aipw": aipw_ate,
    "psm_att": lambda ds: psm(ds, "ATT"),
    "psm_ate": lambda ds: psm(ds, "ATE"),
}


def dataset_from_spec(spec: dict) -> CausalDataset:
    """Either {'synthetic': {make_scm kwargs}} or {'X': [[..]], 'treatment': [..], 'outcome': [..]}."""
    if "synthetic" in spec:
        return make_scm(**spec["synthetic"])
    X = np.asarray(spec["X"], float)
    return CausalDataset(
        X=X, treatment=np.asarray(spec["treatment"], int), outcome=np.asarray(spec["outcome"], float),
        feature_names=spec.get("feature_names") or [f"x{i}" for i in range(X.shape[1])],
        dataset_name=spec.get("name", "uploaded"))


def list_datasets() -> list[dict]:
    return [{"name": "synthetic_scm", "ground_truth": True, "license": "MIT (generated)"},
            {"name": "ihdp", "ground_truth": True, "license": "see source"},
            {"name": "acic2016", "ground_truth": True, "license": "see Zenodo record"},
            {"name": "criteo_uplift_v2.1", "ground_truth": False, "license": "CC BY-NC-SA 4.0"},
            {"name": "lalonde", "ground_truth": False, "license": "see source"}]


def validate(spec: dict, post_treatment: list[str] | None = None) -> dict:
    return validate_dataset(dataset_from_spec(spec), post_treatment)


def identify(graph_json: str, treatment: str, outcome: str) -> dict:
    g = CausalDAG.from_json(graph_json)
    g.validate()
    return {"backdoor_paths": g.backdoor_paths(treatment, outcome),
            "adjustment_sets": [sorted(s) for s in g.adjustment_sets(treatment, outcome)],
            "confounder_candidates": sorted(g.confounder_candidates(treatment, outcome)),
            "mediators": sorted(g.mediators(treatment, outcome)),
            "colliders": g.colliders(),
            "note": "Valid only if the supplied graph is causally correct."}


def estimate(spec: dict, estimator: str) -> dict:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; choose from {sorted(ESTIMATORS)}")
    ds = dataset_from_spec(spec)
    r = ESTIMATORS[estimator](ds)
    out = {"estimator": r.name, "estimand": r.estimand, "estimate": r.estimate, "se": r.se,
           "ci": r.ci, "diagnostics": r.diagnostics}
    if ds.true_ate is not None:
        out["true_ate"] = ds.true_ate
    return out


def sensitivity(spec: dict, q: float = 1.0, alpha: float = 0.05) -> dict:
    ds = dataset_from_spec(spec)
    return sensitivity_report(ds.X, ds.treatment, ds.outcome, q, alpha) | {
        "note": "Robustness value measures how strong a confounder must be to overturn the "
                "estimate; it cannot detect whether confounding exists."}


def evaluate_policy(spec: dict, budget: float) -> dict:
    """Policy value is only computable exactly on simulated data with potential outcomes."""
    from causalforge.meta_learners.learners import XLearner
    ds = dataset_from_spec(spec)
    if ds.y0 is None:
        raise ValueError("policy value needs potential outcomes (synthetic data only here); "
                         "use IPW/AIPW policy evaluation for randomized real data")
    idx = np.random.default_rng(0).permutation(ds.n)
    tr, te = ds.subset(idx[: ds.n // 2]), ds.subset(idx[ds.n // 2:])
    m = XLearner().fit(tr.X, tr.treatment, tr.outcome)
    pi = top_k_policy(m.predict_cate(te.X), budget)
    return {"budget": budget, "treated_fraction": float(pi.mean()),
            "policy_value": true_policy_value(te, pi),
            "treat_none": true_policy_value(te, np.zeros(te.n, int)),
            "treat_all": true_policy_value(te, np.ones(te.n, int))}
