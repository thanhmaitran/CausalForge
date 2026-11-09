"""Experiment suite EXP-01..EXP-08 on synthetic SCMs with exact ground truth.

Everything reported is produced by running this code; nothing is hard-coded.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from causalforge.causal_ml.dml import dml_plr
from causalforge.decision.policies import ipw_policy_value, top_k_policy, true_policy_value
from causalforge.estimators.doubly_robust import aipw_ate
from causalforge.estimators.matching import psm
from causalforge.estimators.regression import naive_ate, regression_adjustment
from causalforge.estimators.weighting import ipw_ate
from causalforge.evaluation.metrics import bias, coverage, pehe, rmse
from causalforge.meta_learners.learners import SLearner, TLearner, XLearner
from causalforge.sensitivity.partial_r2 import sensitivity_report
from causalforge.sensitivity.placebo import placebo_treatment_test
from causalforge.synthetic.generators import make_scm

CLASSICAL = {
    "naive": lambda ds: naive_ate(ds),
    "regression_adj": lambda ds: regression_adjustment(ds),
    "ipw_hajek": lambda ds: ipw_ate(ds),
    "aipw_linear": lambda ds: aipw_ate(ds),
    "psm_ate": lambda ds: psm(ds, "ATE"),
}


def _summ(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        cis = [(a, b) for a, b in zip(g.lo, g.hi)]
        out.append(
            dict(zip(keys, k))
            | {
                "bias": bias(g.est, g.truth),
                "rmse": rmse(g.est, g.truth),
                "sd": float(np.std(g.est, ddof=1)),
                "coverage": coverage(cis, g.truth.tolist()),
                "ci_width": float((g.hi - g.lo).mean()) if g.lo.notna().all() else np.nan,
                "runtime_s": float(g.rt.mean()),
                "n_reps": len(g),
            }
        )
    return pd.DataFrame(out)


def _run(est, ds):
    t0 = time.perf_counter()
    r = est(ds)
    ci = r.ci or (np.nan, np.nan)
    return dict(est=r.estimate, lo=ci[0], hi=ci[1], truth=ds.true_ate, rt=time.perf_counter() - t0)


def exp01_classical(R=100, n=1000, seed=42):
    rows = []
    for sc, kw in {"linear": {}, "nonlinear": {"nonlinear": True}}.items():
        for r in range(R):
            ds = make_scm(n, seed + r, **kw)
            for name, est in CLASSICAL.items():
                rows.append({"scenario": sc, "estimator": name} | _run(est, ds))
    return _summ(rows, ["scenario", "estimator"])


def exp04_randomized_vs_observational(R=100, n=1000, seed=142):
    rows = []
    for sc, kw in {
        "randomized": {"randomized": True},
        "observational": {"confounding": 1.0},
    }.items():
        for r in range(R):
            ds = make_scm(n, seed + r, **kw)
            for name in ["naive", "regression_adj", "ipw_hajek", "aipw_linear"]:
                rows.append({"assignment": sc, "estimator": name} | _run(CLASSICAL[name], ds))
    return _summ(rows, ["assignment", "estimator"])


def exp05_confounding(R=100, n=2000, seed=242, levels=(0.0, 0.5, 1.0, 2.0, 3.0)):
    rows = []
    for c in levels:
        for r in range(R):
            ds = make_scm(n, seed + r, confounding=c)
            for name in ["naive", "regression_adj", "ipw_hajek", "aipw_linear"]:
                rows.append({"confounding": c, "estimator": name} | _run(CLASSICAL[name], ds))
    return _summ(rows, ["confounding", "estimator"])


def exp02_meta_learners(R=30, n=2000, seed=342):
    rows = []
    for sc, kw in {"linear": {}, "nonlinear": {"nonlinear": True}}.items():
        for r in range(R):
            ds = make_scm(n, seed + r, **kw)
            for L in (SLearner, TLearner, XLearner):
                t0 = time.perf_counter()
                m = L().fit(ds.X, ds.treatment, ds.outcome)
                p = m.predict_cate(ds.X)
                rows.append(
                    dict(
                        scenario=sc,
                        learner=L.name,
                        pehe=pehe(p, ds.true_cate),
                        ate_err=abs(p.mean() - ds.true_ate),
                        runtime_s=time.perf_counter() - t0,
                    )
                )
    return (
        pd.DataFrame(rows)
        .groupby(["scenario", "learner"], sort=False)
        .agg(
            pehe=("pehe", "mean"),
            pehe_sd=("pehe", "std"),
            ate_abs_err=("ate_err", "mean"),
            runtime_s=("runtime_s", "mean"),
            n_reps=("pehe", "size"),
        )
        .reset_index()
    )


def exp03_dml_vs_aipw(R=30, n=2000, seed=442):
    gb = lambda: HistGradientBoostingRegressor(
        max_iter=80,
        max_leaf_nodes=8,  # noqa: E731
        min_samples_leaf=20,
        random_state=0,
    )
    ests = {
        "dml_plr_gb": lambda ds: dml_plr(ds),
        "aipw_crossfit_gb": lambda ds: aipw_ate(ds, outcome_model=gb()),
        "aipw_crossfit_linear": lambda ds: aipw_ate(ds),
    }
    rows = []
    for sc, kw in {"linear": {}, "nonlinear": {"nonlinear": True}}.items():
        for r in range(R):
            ds = make_scm(n, seed + r, **kw)
            for name, e in ests.items():
                rows.append({"scenario": sc, "estimator": name} | _run(e, ds))
    return _summ(rows, ["scenario", "estimator"])


def exp06_hidden_confounding(R=100, n=2000, seed=542, strengths=(0.0, 0.5, 1.0, 1.5)):
    rows, rv = [], []
    for s in strengths:
        for r in range(R):
            ds = make_scm(n, seed + r, hidden_t=s, hidden_y=s)
            orc = make_scm(n, seed + r, hidden_t=s, hidden_y=s, keep_hidden=True)
            for name, (est, d) in {
                "aipw_observed_X": (CLASSICAL["aipw_linear"], ds),
                "aipw_oracle_with_U": (CLASSICAL["aipw_linear"], orc),
                "naive": (CLASSICAL["naive"], ds),
            }.items():
                rows.append({"hidden_strength": s, "estimator": name} | _run(est, d))
            if r < 30:
                rep = sensitivity_report(ds.X, ds.treatment, ds.outcome)
                rv.append(
                    {"hidden_strength": s, "rv_q": rep["rv_q"], "rv_q_alpha": rep["rv_q_alpha"]}
                )
    summ = _summ(rows, ["hidden_strength", "estimator"])
    rvdf = pd.DataFrame(rv).groupby("hidden_strength").mean().reset_index()
    return summ, rvdf


def exp06b_placebo(n=2000, seed=642):
    out = []
    for s in (0.0, 1.5):
        ds = make_scm(n, seed, hidden_t=s, hidden_y=s)
        res = placebo_treatment_test(ds, CLASSICAL["aipw_linear"], n_perm=200, seed=seed)
        out.append({"hidden_strength": s} | res)
    return pd.DataFrame(out)


def exp07_sample_size(
    seed=742, sizes=((500, 50), (1000, 50), (5000, 30), (10000, 20), (50000, 10))
):
    rows = []
    for n, R in sizes:
        for r in range(R):
            ds = make_scm(n, seed + r)
            for name in ["regression_adj", "ipw_hajek", "aipw_linear"]:
                rows.append({"n": n, "estimator": name} | _run(CLASSICAL[name], ds))
    return _summ(rows, ["n", "estimator"])


def exp08_policy(R=20, n=4000, seed=842, budgets=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0)):
    rows, ipwchk = [], []
    for r in range(R):
        ds = make_scm(n, seed + r, randomized=True, nonlinear=True)
        rng = np.random.default_rng(seed + r)
        idx = rng.permutation(n)
        tr, te = ds.subset(idx[: n // 2]), ds.subset(idx[n // 2 :])
        models = {
            "x_learner": XLearner().fit(tr.X, tr.treatment, tr.outcome),
            "t_learner": TLearner().fit(tr.X, tr.treatment, tr.outcome),
        }
        for b in budgets:
            for name, m in models.items():
                pi = top_k_policy(m.predict_cate(te.X), b)
                rows.append(dict(budget=b, policy=name, value=true_policy_value(te, pi)))
            rows.append(
                dict(
                    budget=b,
                    policy="oracle",
                    value=true_policy_value(te, top_k_policy(te.true_cate, b)),
                )
            )
            rnd = np.zeros(len(te.X), int)
            rnd[rng.permutation(len(rnd))[: int(b * len(rnd))]] = 1
            rows.append(dict(budget=b, policy="random", value=true_policy_value(te, rnd)))
        pi = top_k_policy(models["x_learner"].predict_cate(te.X), 0.3)
        ipwchk.append(
            dict(
                true=true_policy_value(te, pi),
                ipw=ipw_policy_value(te.treatment, te.outcome, pi, te.propensity),
            )
        )
    pol = pd.DataFrame(rows).groupby(["budget", "policy"]).value.agg(["mean", "std"]).reset_index()
    chk = pd.DataFrame(ipwchk)
    chk_s = {
        "budget": 0.3,
        "mean_true_value": chk.true.mean(),
        "mean_ipw_estimate": chk.ipw.mean(),
        "mean_abs_gap": float((chk.true - chk.ipw).abs().mean()),
        "n_reps": len(chk),
    }
    return pol, pd.DataFrame([chk_s])


def exp03b_forest(R=20, n=2000, seed=1042):
    from causalforge.causal_ml.causal_forest import HonestCausalForest

    rows = []
    for sc, kw in {
        "randomized_nonlinear": {"randomized": True, "nonlinear": True},
        "observational_nonlinear": {"nonlinear": True},
    }.items():
        for r in range(R):
            ds = make_scm(n, seed + r, **kw)
            for name, m in {
                "t_learner": TLearner(),
                "x_learner": XLearner(),
                "honest_forest": HonestCausalForest(seed=r),
            }.items():
                t0 = time.perf_counter()
                m.fit(ds.X, ds.treatment, ds.outcome)
                if name == "honest_forest":
                    p, sd = m.predict_cate(ds.X, return_std=True)
                    cov = float(np.mean(np.abs(p - ds.true_cate) <= 1.96 * sd))
                else:
                    p, cov = m.predict_cate(ds.X), np.nan
                rows.append(
                    dict(
                        scenario=sc,
                        learner=name,
                        pehe=pehe(p, ds.true_cate),
                        ate_abs_err=abs(p.mean() - ds.true_ate),
                        pointwise_cov=cov,
                        runtime_s=time.perf_counter() - t0,
                    )
                )
    return (
        pd.DataFrame(rows)
        .groupby(["scenario", "learner"], sort=False)
        .agg(
            pehe=("pehe", "mean"),
            pehe_sd=("pehe", "std"),
            ate_abs_err=("ate_abs_err", "mean"),
            pointwise_coverage=("pointwise_cov", "mean"),
            runtime_s=("runtime_s", "mean"),
            n_reps=("pehe", "size"),
        )
        .reset_index()
    )


def exp09_negative_control(R=50, n=2000, seed=1142, strengths=(0.0, 0.5, 1.0)):
    from causalforge.sensitivity.negative_controls import negative_control_outcome_test

    rows = []
    for s in strengths:
        for r in range(R):
            ds = make_scm(n, seed + r, hidden_t=s, hidden_y=s)
            rng = np.random.default_rng(seed + 10_000 + r)
            nco = 0.8 * ds.metadata["U"] + rng.normal(size=n)  # shares U, not caused by T
            out = negative_control_outcome_test(ds, nco)
            true_bias = ds.metadata["hidden_t"] > 0
            rows.append(dict(hidden_strength=s, est=out["estimate"], reject=out["reject_null"]))
    d = pd.DataFrame(rows)
    return (
        d.groupby("hidden_strength")
        .agg(
            mean_nco_estimate=("est", "mean"),
            rejection_rate=("reject", "mean"),
            n_reps=("est", "size"),
        )
        .reset_index()
    )


def exp10_did(R=100, seed=1242):
    from causalforge.estimators.difference_in_differences import did_2x2, event_study
    from causalforge.synthetic.panel_iv import make_panel

    rows = []
    for sc, v in {"parallel_trends": 0.0, "violated_trends(0.3)": 0.3}.items():
        for r in range(R):
            p = make_panel(seed=seed + r, trend_violation=v)
            res = did_2x2(p["unit"], p["group"], p["post"], p["y"])
            _, o = event_study(p["unit"], p["group"], p["period"], p["y"], p["treat_start"] - 1)
            rows.append(
                dict(
                    scenario=sc,
                    est=res.estimate,
                    lo=res.ci[0],
                    hi=res.ci[1],
                    truth=p["true_att"],
                    rt=0.0,
                    pretrend_reject=o["pretrend_wald_p"] < 0.05,
                )
            )
    df = pd.DataFrame(rows)
    out = _summ(df.drop(columns="pretrend_reject").to_dict("records"), ["scenario"])
    out["pretrend_rejection_rate"] = (
        df.groupby("scenario", sort=False).pretrend_reject.mean().values
    )
    return out.drop(columns="runtime_s")


def exp11_iv(R=100, seed=1342):
    from causalforge.estimators.instrumental_variables import tsls
    from causalforge.synthetic.panel_iv import make_iv

    cfg = {
        "ols_confounded": None,
        "2sls_strong(1.0)": dict(strength=1.0),
        "2sls_weak(0.08)": dict(strength=0.08),
        "2sls_exclusion_violated(0.3)": dict(strength=1.0, exclusion_violation=0.3),
    }
    rows = []
    for name, kw in cfg.items():
        for r in range(R):
            d = make_iv(n=1000, seed=seed + r, **(kw or {}))
            if kw is None:
                D = np.column_stack([np.ones(1000), d["d"]])
                b = np.linalg.lstsq(D, d["y"], rcond=None)[0]
                res = D @ b - d["y"]
                se = np.sqrt(res @ res / 998 * np.linalg.inv(D.T @ D)[1, 1])
                est, ci, F = b[1], (b[1] - 1.96 * se, b[1] + 1.96 * se), np.nan
            else:
                o = tsls(d["y"], d["d"], d["z"])
                est, ci, F = o.estimate, o.ci, o.diagnostics["first_stage_F"]
            rows.append(dict(method=name, est=est, lo=ci[0], hi=ci[1], truth=1.0, rt=0.0, F=F))
    df = pd.DataFrame(rows)
    out = _summ(df.drop(columns="F").to_dict("records"), ["method"]).drop(columns="runtime_s")
    out["mean_first_stage_F"] = df.groupby("method", sort=False).F.mean().values
    return out


def environment() -> dict:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    }


def run_all(out_dir="artifacts/results", quick=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = 0.2 if quick else 1.0
    R = lambda x: max(3, int(x * f))  # noqa: E731
    t0 = time.perf_counter()
    res = {
        "exp01_classical": exp01_classical(R(100)),
        "exp02_meta_learners": exp02_meta_learners(R(30)),
        "exp03_dml_vs_aipw": exp03_dml_vs_aipw(R(30)),
        "exp04_randomized_vs_observational": exp04_randomized_vs_observational(R(100)),
        "exp05_confounding": exp05_confounding(R(100)),
    }
    res["exp06_hidden_confounding"], res["exp06_robustness_value"] = exp06_hidden_confounding(
        R(100)
    )
    res["exp06b_placebo"] = exp06b_placebo()
    res["exp07_sample_size"] = exp07_sample_size()
    res["exp08_policy_value"], res["exp08_ipw_check"] = exp08_policy(R(20))
    res["exp03b_forest"] = exp03b_forest(R(20))
    res["exp09_negative_control"] = exp09_negative_control(R(50))
    res["exp10_did"] = exp10_did(R(100))
    res["exp11_iv"] = exp11_iv(R(100))
    for k, df in res.items():
        df.to_csv(out / f"{k}.csv", index=False)
    (out / "environment.json").write_text(
        json.dumps(
            environment() | {"quick": quick, "total_runtime_s": time.perf_counter() - t0}, indent=2
        )
    )
    return res
