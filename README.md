# CausalForge

Causal inference, treatment-effect estimation and decision intelligence — implemented from
scratch in Python and benchmarked on simulations with exact ground truth.

> **Status: v1.0.0 software complete; real-dataset results pending your run.** Every module is
> implemented and tested (44 tests, 8 notebooks execute). IHDP, ACIC 2016, Criteo and LaLonde
> benchmarks are implemented but were **not run** (the development sandbox had no network). Run
> `docs/RUN_ON_YOUR_MACHINE.md`. All numbers below come from `scripts/run_experiments.py`.

## Research questions

| ID | Question | Where answered |
|----|----------|----------------|
| RQ1 | How do classical estimators differ in ATE accuracy? | EXP-01, EXP-04, EXP-05, EXP-07 |
| RQ2 | How well do causal ML models recover heterogeneity? | EXP-02, EXP-03, EXP-03b |
| RQ3 | How does confounding affect estimation? | EXP-04, EXP-05 |
| RQ4 | Can sensitivity analysis flag untrustworthy conclusions? | EXP-06, EXP-09 |
| RQ5 | How do effect estimates become budget-constrained policies? | EXP-08 |

## What is implemented

| Module | Contents |
|---|---|
| `graphs/dag.py` | DAG validation, roles, confounders/mediators/colliders, backdoor-path enumeration, backdoor-criterion check (via d-separation), brute-force minimal adjustment sets, JSON serialization |
| `data/` | Unified `CausalDataset` schema, data-quality validator (balance, overlap, leakage warnings), streaming Criteo ingestion (ITT design, `exposure` excluded) |
| `synthetic/generators.py` | SCMs with hidden confounding, nonlinearity, heterogeneity, randomized/observational assignment; truth from structural equations |
| `estimators/` | Naive, regression adjustment, propensity + diagnostics (SMD, ESS, overlap), IPW (Hajek/HT, explicit trimming), AIPW with cross-fitting, 1-NN PSM (ATT/ATE, caliper), generic bootstrap |
| `meta_learners/` | S-, T-, X-learners |
| `causal_ml/dml.py` | DML (partially linear) with genuine K-fold cross-fitting |
| `sensitivity/` | Cinelli–Hazlett robustness values and bias-adjusted estimates; permutation placebo test |
| `evaluation/`, `decision/` | ATE error, bias, RMSE, PEHE, CI coverage; top-k budget policies, exact / IPW / AIPW policy value |
| `estimators/` (cont.) | DiD (cluster-robust, event study, pre-trend test), 2SLS (first-stage F), cluster bootstrap, IPW-ATT |
| `causal_ml/causal_forest.py` | Honest causal forest (R-loss splits, honest leaves, heuristic SE) |
| `sensitivity/negative_controls.py` | Negative-control-outcome test |
| `data/loaders/benchmarks.py` | IHDP, ACIC 2016, LaLonde loaders (format-tested on fixtures) |
| `experiments/` | EXP-01…EXP-11 runner, real-data runner, file-based tracking (MLflow mirror if installed) |
| `api/`, `frontend/`, `docker/` | FastAPI service layer + SQLite store + PostgreSQL DDL; React dashboard; Docker Compose |
| `reporting/` | Figures with written captions and an auto-generated benchmark report |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest tests -v                                        # or: PYTHONPATH=src python tests/run_tests.py
PYTHONPATH=src python scripts/run_experiments.py       # ~2 min on one CPU core
PYTHONPATH=src python scripts/generate_report.py       # figures + reports/benchmark_report/
python scripts/download_data.py --dataset all          # needs internet; records SHA-256
```

## Measured results (synthetic SCMs, n and replications as stated)

**Confounding (EXP-05, n=2000, 100 reps).** Naive bias grows with confounding strength
(0.35 → 0.63 → 0.94 → 1.07 at strengths 0.5, 1, 2, 3) while AIPW stays at |bias| ≤ 0.007.

**Classical estimators (EXP-01, n=1000, 100 reps).** Regression adjustment, IPW, AIPW and PSM
are all near-unbiased under observed confounding; naive bias is ≈0.6. IPW intervals covered
100% in several settings: its SE ignores propensity-estimation error and is conservative.

**Meta-learners (EXP-02, n=2000, 30 reps, gradient-boosting base).** √PEHE — linear SCM:
S 0.252, X 0.325, T 0.487; nonlinear: S 0.289, X 0.326, T 0.537. This reflects the smooth
heterogeneity of *this* simulator and should not be read as a general ranking.

**Hidden confounding (EXP-06, n=2000, 100 reps).** AIPW on observed covariates has bias
0.23 / 0.83 / 1.62 at hidden strengths 0.5 / 1.0 / 1.5 (95% CI coverage 3% / 0% / 0%), while
an oracle AIPW given U stays unbiased. Two important negative findings:
* The mean robustness value *rises* (0.36 → 0.52) as hidden confounding worsens, because the
  biased estimate looks larger and more "robust". RV quantifies how much confounding would
  overturn an estimate; **it cannot detect confounding**.
* The permutation placebo test rejected at 5.5% and 3.5% even under strong hidden
  confounding: passing it is not evidence of unconfoundedness.

**Policy (EXP-08, randomized nonlinear SCM, n=4000, 20 reps).** X- and T-learner top-k
policies approach the oracle and beat random targeting at budgets 0.1–0.7. The IPW policy-value
estimate (0.945) matched the exact value (0.943), mean absolute gap 0.034.

Full tables: [`reports/benchmark_report/benchmark_report.md`](reports/benchmark_report/benchmark_report.md);
figures (each with a description) in `reports/benchmark_report/figures/`.

## Project status

| Area | State |
|---|---|
| DAG, schema, validator, SCMs, estimators, learners, DML, forest, sensitivity, DiD, IV, policy | Done, tested |
| EXP-01…EXP-11 on synthetic data | Done, executed (`artifacts/results/`) |
| 8 notebooks | Done; code executed and verified |
| Technical + research report | Written (real-data section marked pending) |
| IHDP / ACIC / LaLonde / Criteo benchmarks | Implemented, format-tested on fixtures; **needs your run** |
| FastAPI app, React dashboard, Docker Compose, MLflow server | Written; service layer tested; **not executed here** |
| EconML / DoWhy cross-check | Script written; **not executed here** |
| ruff, mypy, Hypothesis | Configured; **not run here** |
| Screenshots | Generated evidence figures included; UI/Docker/MLflow screenshots need your machine |

## Limitations

* Results come from one family of low-dimensional synthetic SCMs; they say little about
  performance on real or high-dimensional data.
* PSM reports no SE (the plain bootstrap is invalid for matching); coverage is not claimed.
* Honest-forest intervals are heuristic (measured pointwise coverage 0.67–0.69).
* DML-PLR targets a variance-weighted effect under heterogeneity, not the ATE.
* Coverage estimates from 20–30 replications have Monte-Carlo error of about ±0.05.
* Identification rests on consistency, conditional exchangeability, positivity and SUTVA;
  no algorithm here verifies them. See `docs/methodology/`.

## License and data

Code: MIT. Criteo data: CC BY-NC-SA 4.0 (non-commercial); cite the source. Raw data is never committed.

## Additional measured results
* **Forest (EXP-03b):** √PEHE 0.166/0.180 vs X-learner 0.331/0.329, T-learner 0.472/0.536 (randomized/observational nonlinear); pointwise coverage 0.690/0.665.
* **Negative control (EXP-09):** rejects 100% under hidden confounding vs 12% under none (valid-by-construction NCO).
* **DiD (EXP-10):** coverage 0.97 under parallel trends; bias 0.90 and 100% pre-trend rejection when violated.
* **IV (EXP-11):** 2SLS unbiased with a strong valid instrument (F≈505); weak IV (F≈4.3) RMSE 2.18; exclusion violation gives bias 0.30 undetected by F.
