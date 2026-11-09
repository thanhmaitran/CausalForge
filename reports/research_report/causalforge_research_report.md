# CausalForge: A Reproducible Platform for Causal Effect Estimation, Heterogeneous Treatment Analysis and Robust Decision Intelligence

## Abstract
We present CausalForge, a from-scratch Python platform for ATE/ATT/CATE estimation, sensitivity
analysis and budget-constrained policy evaluation, and a benchmark on synthetic structural causal
models with exact ground truth. In controlled experiments, adjustment estimators remove observed
confounding bias (|bias| ≤ 0.007 for AIPW versus up to 1.07 for the naive contrast), but all fail
under hidden confounding (AIPW bias up to 1.62, CI coverage 0%). Robustness values and permutation
placebo tests did not detect hidden confounding, whereas a valid negative-control outcome did.
An honest causal forest achieved lower √PEHE (0.17–0.18) than X-learner (0.33) and T-learner
(0.47–0.54) on a nonlinear simulator, but its heuristic pointwise intervals covered only 67–69%.
Real-data benchmarks (IHDP, ACIC 2016, Criteo, LaLonde) are implemented but **not yet executed**;
their results are reported in §6.2 once run.

## 1. Introduction
Research question: how reliably can interventions' causal effects be estimated from observational
and experimental data while quantifying uncertainty, heterogeneity and robustness? RQ1–RQ5 follow
the project specification (estimator accuracy, heterogeneity recovery, confounding, sensitivity,
policy construction).

## 2. Related work
Potential outcomes (Rubin); structural causal models and the backdoor criterion (Pearl); propensity
scores (Rosenbaum & Rubin, 1983); doubly robust estimation; meta-learners (Künzel et al., 2019);
double/debiased ML (Chernozhukov et al., 2018); causal forests (Wager & Athey, 2018; Athey,
Tibshirani & Wager, 2019); omitted-variable-bias sensitivity (Cinelli & Hazlett, 2020); policy
learning. *The original papers should be read and cited in full before submission; this draft
lists them as pointers and does not paraphrase their results.*

## 3. Background
Estimands: ITE, ATE, CATE, ATT. Identification assumptions: consistency, conditional
exchangeability, positivity, SUTVA. A numerically accurate estimator does not imply identification.

## 4. Methodology
Implemented estimators and diagnostics are listed in the technical report (§6–§8). Ground truth in
simulation is computed from structural equations. Policy value is exact in simulation and
IPW-estimated in randomized data.

## 5. Experimental design
Synthetic SCMs (linear/nonlinear, heterogeneous, randomized/observational, hidden confounder),
synthetic panels (DiD) and instruments (IV). Sample sizes, replications and seeds are in
`artifacts/results/*.csv`; the full suite is `scripts/run_experiments.py` (≈2 minutes, one CPU core).

## 6. Results
### 6.1 Synthetic benchmarks (measured)
* **Confounding (EXP-05).** Naive bias 0.35 / 0.63 / 0.94 / 1.07 at strengths 0.5 / 1 / 2 / 3; AIPW |bias| ≤ 0.007.
* **Sample size (EXP-07).** AIPW RMSE 0.118, 0.069, 0.035, 0.024, 0.010 at n = 500, 1k, 5k, 10k, 50k (≈1/√n).
* **Meta-learners (EXP-02), √PEHE.** Linear: S 0.252, X 0.325, T 0.487. Nonlinear: S 0.289, X 0.326, T 0.537.
* **DML vs AIPW (EXP-03), 30 reps.** Coverage 0.90–0.97 (Monte-Carlo error ≈ ±0.05); RMSE 0.04–0.07.
* **Honest forest (EXP-03b), 20 reps.** √PEHE 0.166 (randomized) and 0.180 (observational) versus X-learner 0.331/0.329 and T-learner 0.472/0.536; pointwise coverage 0.690/0.665.
* **Hidden confounding (EXP-06).** AIPW (observed X) bias 0.23 / 0.83 / 1.62 at U-strength 0.5 / 1.0 / 1.5, coverage 3% / 0% / 0%; oracle AIPW with U unbiased. Mean RV rose 0.36 → 0.52. Permutation placebo rejected 5.5% / 3.5% (strength 0 / 1.5).
* **Negative control outcome (EXP-09).** Rejection 12% at no hidden confounding (nominal 5%, 50 reps) and 100% at strengths 0.5 and 1.0. The NCO here is valid by construction (shares U, unaffected by T).
* **DiD (EXP-10).** Parallel trends: bias −0.004, coverage 0.97, pre-trend rejection 2%. Violated trends: bias 0.90, coverage 0%, pre-trend rejection 100%.
* **IV (EXP-11).** OLS bias 0.33. Strong valid IV (F≈505): bias −0.001, coverage 1.00. Weak IV (F≈4.3): RMSE 2.18. Exclusion violated: bias 0.30 (F is large, so F cannot diagnose this).
* **Policy (EXP-08).** Learned top-k policies approach the oracle and beat random at budgets 0.1–0.7; IPW policy value 0.945 vs exact 0.943.

### 6.2 Real-data benchmarks
**Pending.** Run `scripts/run_real_benchmarks.py` (see `docs/RUN_ON_YOUR_MACHINE.md`); results are
written to `artifacts/real/` and appended to `reports/benchmark_report/benchmark_report.md`.
No real-data claim is made in this draft.

## 7. Discussion
Adjustment fixes only measured confounding. Diagnostics differ in what they can see: RV quantifies
fragility but rises with a biased estimate; placebo tests are blind to confounding that does not
distort the permuted contrast; negative controls and pre-trend tests detect specific violations when
their own assumptions hold; first-stage F detects weak instruments but not exclusion violations.
S-learner's advantage in this simulator reflects smooth heterogeneity and shrinkage toward a
constant effect; it should not be generalised.

## 8. Limitations
Low-dimensional well-specified simulators; heuristic forest intervals; PSM without SE; IPW SE
conservative; DML-PLR weighted target; real-data and reference-library cross-checks outstanding.

## 9. Conclusion
CausalForge provides a tested, reproducible implementation of estimation, heterogeneity,
sensitivity and policy tools, and quantifies where each tool fails. Contributions: unified
framework, DAG/adjustment engine, reproducible multi-DGP benchmark, sensitivity-method comparison,
uncertainty-aware policy evaluation (on simulations so far).

## References
Rosenbaum & Rubin (1983), The central role of the propensity score in observational studies for causal effects.
Imbens & Rubin (2015), Causal Inference for Statistics, Social, and Biomedical Sciences.
Chernozhukov et al. (2018), Double/debiased machine learning for treatment and structural parameters.
Künzel et al. (2019), Metalearners for estimating heterogeneous treatment effects using machine learning.
Wager & Athey (2018), Estimation and inference of heterogeneous treatment effects using random forests.
Athey, Tibshirani & Wager (2019), Generalized random forests.
Cinelli & Hazlett (2020), Making sense of sensitivity: extending omitted variable bias.
