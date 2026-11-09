# CausalForge — Technical Report (v1.0.0)

## 1. Introduction
CausalForge is a research-engineering platform for causal effect estimation, heterogeneous
effects, sensitivity analysis and budget-constrained decisions. Core algorithms are implemented
from scratch (NumPy/SciPy/scikit-learn/NetworkX); DoWhy/EconML are reserved for cross-checks.

## 2. Requirements
Python 3.12; runtime deps in `pyproject.toml`; optional extras `reference` (DoWhy, EconML,
statsmodels), `tracking` (MLflow), `api` (FastAPI), `data` (Polars, PyArrow).

## 3. Architecture
`CausalDataset` is the single data interface; estimators return `EstimateResult`. Layers:
data → graphs / estimators / meta_learners / causal_ml → sensitivity → evaluation / decision →
experiments → reporting. The API (`api/causalforge_api`) is a thin routing layer over
`services/engine.py`; it contains no statistics.

## 4. Data architecture
Raw files are never modified or committed. `scripts/download_data.py` records HTTP status, size
and SHA-256. Criteo is ingested in chunks (`criteo_stream.py`), validated, and analysed as
intention-to-treat; `exposure` is post-treatment and excluded from covariates. PostgreSQL stores
metadata only (`docker/init.sql`); a SQLite store with the same tables runs locally/tests.

## 5. Causal graph engine
Cycle detection, role validation, confounder/mediator/collider detection, backdoor-path
enumeration, the backdoor criterion implemented through d-separation in the graph with outgoing
treatment edges removed, brute-force minimal adjustment sets (small graphs), JSON serialization.
The engine reasons conditional on the supplied graph and cannot verify its causal correctness.

## 6. Classical estimators
Naive, regression adjustment (bootstrap SE), propensity + diagnostics (SMD, ESS, overlap),
IPW (Hajek/HT, explicit trimming, influence-function SE ignoring propensity estimation),
AIPW with K-fold cross-fitting and reported clipping fraction, 1-NN PSM (ATT/ATE, caliper, no SE),
IPW-ATT, DiD with CR1 cluster-robust SEs + event study + joint pre-trend Wald test, 2SLS with
robust SEs and first-stage F, generic i.i.d. and cluster bootstrap.

## 7. Causal ML
S/T/X-learners; DML (partially linear) with genuine out-of-fold residuals (verified by a spy-model
test); honest causal forest: R-loss splitting on one half of each subsample, leaf effects from the
other half; heuristic little-bags-style variance (not the Wager–Athey CI). EconML's CausalForestDML
is the reference for validated inference (`scripts/crosscheck_reference.py`, not yet run).

## 8. Sensitivity engine
Cinelli–Hazlett robustness values and bias-adjusted estimates, permutation placebo test,
negative-control-outcome test, hidden-confounding simulations. Measured limits: RV and placebo
tests do not detect confounding; a *valid* negative control does (EXP-09).

## 9. API design
Endpoints: `/health, /datasets, /datasets/validate, /graphs, /graphs/{id}, /causal/identify,
/causal/estimate, /causal/sensitivity, /experiments, /experiments/{id}(/results),
/policies/evaluate, /reports/{id}`. Validation errors return HTTP 422. The service layer is
unit-tested; the FastAPI app itself was **not executed** during development (FastAPI unavailable).

## 10. Frontend
React + TypeScript (Vite), seven pages (Overview, Datasets, Causal Graphs, Estimators,
Experiments, Sensitivity, Reports & Decisions). No placeholder statistics are rendered. Not built
during development (no npm access).

## 11. Testing
44 tests (unit, statistical Monte-Carlo, property-style, loader format fixtures, reproducibility).
Statistical tests check recovery of truth, null calibration, and detection of assumption violations.
Hypothesis-based tests, ruff and mypy are configured but were not run (not installed).

## 12. Deployment
`docker/docker-compose.yml`: PostgreSQL, MLflow, API, frontend. Not run during development.
CI (`.github/workflows/ci.yml`): ruff, pytest, quick synthetic experiments; no real data in CI.

## 13. Limitations
See `docs/methodology/limitations_and_validity.md`. Real-data benchmarks require the user's run
(`docs/RUN_ON_YOUR_MACHINE.md`).
