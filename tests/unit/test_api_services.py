import json
import sys

sys.path.insert(0, "api")
from causalforge_api.services import engine  # noqa: E402
from causalforge_api.services.store import Store  # noqa: E402

G = json.dumps(
    {
        "nodes": [
            {"name": "X", "role": "confounder"},
            {"name": "T", "role": "treatment"},
            {"name": "Y", "role": "outcome"},
        ],
        "edges": [["X", "T"], ["X", "Y"], ["T", "Y"]],
    }
)


def test_identify_service():
    r = engine.identify(G, "T", "Y")
    assert r["adjustment_sets"] == [["X"]] and r["confounder_candidates"] == ["X"]


def test_estimate_and_store_roundtrip():
    spec = {"synthetic": {"n": 1500, "seed": 2}}
    res = engine.estimate(spec, "aipw")
    assert abs(res["estimate"] - res["true_ate"]) < 0.3
    s = Store()
    eid = s.add_experiment("aipw", res["estimand"], spec, 0, res)
    got = s.get_experiment(eid)
    assert got["ate"] == res["estimate"] and s.count_experiments() == 1


def test_uploaded_data_spec_and_bad_estimator():
    spec = engine.dataset_from_spec({"synthetic": {"n": 200, "seed": 1}})
    up = {
        "X": spec.X.tolist(),
        "treatment": spec.treatment.tolist(),
        "outcome": spec.outcome.tolist(),
    }
    assert "true_ate" not in engine.estimate(up, "naive")
    try:
        engine.estimate(up, "magic")
    except ValueError:
        return
    raise AssertionError("unknown estimator accepted")


def test_policy_requires_potential_outcomes():
    spec = engine.dataset_from_spec({"synthetic": {"n": 200, "seed": 1}})
    up = {
        "X": spec.X.tolist(),
        "treatment": spec.treatment.tolist(),
        "outcome": spec.outcome.tolist(),
    }
    try:
        engine.evaluate_policy(up, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("policy value computed without potential outcomes")
    r = engine.evaluate_policy({"synthetic": {"n": 2000, "seed": 3, "randomized": True}}, 0.3)
    assert r["policy_value"] >= r["treat_none"]


def test_fastapi_app_if_installed():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return  # fastapi not installed in this environment: API layer untested here
    from causalforge_api.main import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "healthy"
    r = c.post(
        "/causal/estimate", json={"spec": {"synthetic": {"n": 800, "seed": 1}}, "estimator": "aipw"}
    )
    assert (
        r.status_code == 200
        and c.get(f"/experiments/{r.json()['experiment_id']}").status_code == 200
    )
    assert (
        c.post(
            "/causal/estimate", json={"spec": {"synthetic": {"n": 800}}, "estimator": "bogus"}
        ).status_code
        == 422
    )
