# CausalForge benchmark report (auto-generated)

Generated from `artifacts/results/*.csv` by `scripts/generate_report.py`. Environment: Python 3.12.3, numpy 2.5.4, scikit-learn 1.9.1. Total runtime 122s. All data are **synthetic SCMs with exact ground truth**; no real dataset (IHDP, ACIC, Criteo, LaLonde) has been run yet.

## EXP-01 Classical estimators (n=1000)

| scenario   | estimator      |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|:-----------|:---------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
| linear     | naive          |  0.626 |  0.637 | 0.122 |       0    |      0.417 |       0     |      100 |
| linear     | regression_adj |  0     |  0.064 | 0.065 |     nan    |    nan     |       0.001 |      100 |
| linear     | ipw_hajek      |  0.005 |  0.08  | 0.081 |       1    |      0.511 |       0.002 |      100 |
| linear     | aipw_linear    |  0.003 |  0.067 | 0.069 |       0.97 |      0.295 |       0.014 |      100 |
| linear     | psm_ate        |  0.006 |  0.101 | 0.104 |     nan    |    nan     |       0.004 |      100 |
| nonlinear  | naive          |  0.56  |  0.568 | 0.094 |       0    |      0.392 |       0     |      100 |
| nonlinear  | regression_adj |  0.019 |  0.077 | 0.078 |     nan    |    nan     |       0.001 |      100 |
| nonlinear  | ipw_hajek      |  0.004 |  0.089 | 0.092 |       1    |      0.478 |       0.002 |      100 |
| nonlinear  | aipw_linear    | -0.002 |  0.107 | 0.11  |       0.99 |      0.475 |       0.012 |      100 |
| nonlinear  | psm_ate        | -0.004 |  0.108 | 0.111 |     nan    |    nan     |       0.003 |      100 |

## EXP-02 Meta-learners (n=2000)

| scenario   | learner   |   pehe |   pehe_sd |   ate_abs_err |   runtime_s |   n_reps |
|:-----------|:----------|-------:|----------:|--------------:|------------:|---------:|
| linear     | s_learner |  0.252 |     0.034 |         0.078 |       0.038 |       30 |
| linear     | t_learner |  0.487 |     0.026 |         0.053 |       0.065 |       30 |
| linear     | x_learner |  0.325 |     0.026 |         0.051 |       0.127 |       30 |
| nonlinear  | s_learner |  0.289 |     0.038 |         0.067 |       0.036 |       30 |
| nonlinear  | t_learner |  0.537 |     0.039 |         0.052 |       0.063 |       30 |
| nonlinear  | x_learner |  0.326 |     0.023 |         0.049 |       0.123 |       30 |

## EXP-03 (partial) DML-PLR vs cross-fitted AIPW

| scenario   | estimator            |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|:-----------|:---------------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
| linear     | dml_plr_gb           | -0.014 |  0.051 | 0.047 |      0.967 |      0.2   |       0.291 |       30 |
| linear     | aipw_crossfit_gb     | -0.009 |  0.052 | 0.052 |      0.933 |      0.223 |       0.309 |       30 |
| linear     | aipw_crossfit_linear | -0.011 |  0.043 | 0.042 |      0.933 |      0.205 |       0.016 |       30 |
| nonlinear  | dml_plr_gb           | -0.019 |  0.054 | 0.049 |      0.9   |      0.203 |       0.288 |       30 |
| nonlinear  | aipw_crossfit_gb     | -0.001 |  0.05  | 0.05  |      0.967 |      0.232 |       0.305 |       30 |
| nonlinear  | aipw_crossfit_linear |  0.005 |  0.073 | 0.073 |      0.967 |      0.316 |       0.016 |       30 |

## EXP-04 Randomized vs observational

| assignment    | estimator      |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|:--------------|:---------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
| randomized    | naive          | -0.004 |  0.106 | 0.106 |       0.95 |      0.427 |       0     |      100 |
| randomized    | regression_adj | -0.003 |  0.073 | 0.074 |     nan    |    nan     |       0.001 |      100 |
| randomized    | ipw_hajek      | -0.003 |  0.073 | 0.074 |       0.99 |      0.428 |       0.002 |      100 |
| randomized    | aipw_linear    | -0.003 |  0.073 | 0.075 |       0.92 |      0.258 |       0.013 |      100 |
| observational | naive          |  0.643 |  0.651 | 0.108 |       0    |      0.419 |       0     |      100 |
| observational | regression_adj |  0.011 |  0.06  | 0.061 |     nan    |    nan     |       0.001 |      100 |
| observational | ipw_hajek      |  0.015 |  0.07  | 0.07  |       1    |      0.512 |       0.002 |      100 |
| observational | aipw_linear    |  0.009 |  0.063 | 0.065 |       0.97 |      0.294 |       0.013 |      100 |

## EXP-05 Confounding sweep

|   confounding | estimator      |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|--------------:|:---------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
|           0   | naive          |  0     |  0.073 | 0.073 |       0.97 |      0.303 |       0     |      100 |
|           0   | regression_adj | -0     |  0.046 | 0.048 |     nan    |    nan     |       0.001 |      100 |
|           0   | ipw_hajek      | -0     |  0.046 | 0.048 |       1    |      0.303 |       0.002 |      100 |
|           0   | aipw_linear    | -0     |  0.045 | 0.047 |       0.96 |      0.182 |       0.014 |      100 |
|           0.5 | naive          |  0.354 |  0.362 | 0.08  |       0    |      0.301 |       0     |      100 |
|           0.5 | regression_adj | -0.001 |  0.045 | 0.045 |     nan    |    nan     |       0.002 |      100 |
|           0.5 | ipw_hajek      |  0.001 |  0.044 | 0.045 |       1    |      0.316 |       0.003 |      100 |
|           0.5 | aipw_linear    | -0.001 |  0.045 | 0.045 |       0.95 |      0.187 |       0.018 |      100 |
|           1   | naive          |  0.627 |  0.631 | 0.066 |       0    |      0.297 |       0     |      100 |
|           1   | regression_adj | -0.002 |  0.04  | 0.041 |     nan    |    nan     |       0.002 |      100 |
|           1   | ipw_hajek      | -0.001 |  0.047 | 0.047 |       1    |      0.364 |       0.003 |      100 |
|           1   | aipw_linear    | -0.002 |  0.038 | 0.038 |       1    |      0.206 |       0.018 |      100 |
|           2   | naive          |  0.938 |  0.941 | 0.068 |       0    |      0.291 |       0     |      100 |
|           2   | regression_adj |  0.005 |  0.055 | 0.056 |     nan    |    nan     |       0.001 |      100 |
|           2   | ipw_hajek      |  0.027 |  0.132 | 0.129 |       0.95 |      0.59  |       0.003 |      100 |
|           2   | aipw_linear    |  0.007 |  0.083 | 0.083 |       0.97 |      0.307 |       0.016 |      100 |
|           3   | naive          |  1.072 |  1.074 | 0.065 |       0    |      0.287 |       0     |      100 |
|           3   | regression_adj | -0.001 |  0.062 | 0.063 |     nan    |    nan     |       0.001 |      100 |
|           3   | ipw_hajek      | -0.021 |  0.426 | 0.427 |       0.84 |      0.99  |       0.003 |      100 |
|           3   | aipw_linear    | -0.003 |  0.101 | 0.101 |       0.96 |      0.376 |       0.015 |      100 |

## EXP-06 Hidden confounding

|   hidden_strength | estimator          |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|------------------:|:-------------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
|               0   | aipw_observed_X    | -0.004 |  0.049 | 0.049 |       0.96 |      0.205 |       0.014 |      100 |
|               0   | aipw_oracle_with_U | -0.004 |  0.049 | 0.049 |       0.96 |      0.206 |       0.014 |      100 |
|               0   | naive              |  0.622 |  0.627 | 0.077 |       0    |      0.297 |       0     |      100 |
|               0.5 | aipw_observed_X    |  0.232 |  0.239 | 0.058 |       0.03 |      0.225 |       0.015 |      100 |
|               0.5 | aipw_oracle_with_U | -0.005 |  0.054 | 0.055 |       0.94 |      0.216 |       0.015 |      100 |
|               0.5 | naive              |  0.804 |  0.807 | 0.074 |       0    |      0.306 |       0     |      100 |
|               1   | aipw_observed_X    |  0.827 |  0.83  | 0.074 |       0    |      0.266 |       0.015 |      100 |
|               1   | aipw_oracle_with_U | -0.001 |  0.066 | 0.066 |       0.94 |      0.243 |       0.016 |      100 |
|               1   | naive              |  1.273 |  1.276 | 0.086 |       0    |      0.331 |       0     |      100 |
|               1.5 | aipw_observed_X    |  1.616 |  1.618 | 0.082 |       0    |      0.308 |       0.015 |      100 |
|               1.5 | aipw_oracle_with_U | -0.001 |  0.071 | 0.071 |       0.97 |      0.283 |       0.016 |      100 |
|               1.5 | naive              |  1.938 |  1.94  | 0.091 |       0    |      0.363 |       0     |      100 |

## EXP-06 Mean robustness values (OLS, 30 reps)

|   hidden_strength |   rv_q |   rv_q_alpha |
|------------------:|-------:|-------------:|
|               0   |  0.359 |        0.331 |
|               0.5 |  0.391 |        0.365 |
|               1   |  0.459 |        0.435 |
|               1.5 |  0.523 |        0.503 |

## EXP-06b Permutation placebo test

|   hidden_strength |   mean_placebo_estimate |    sd |   rejection_rate |   n_perm |
|------------------:|------------------------:|------:|-----------------:|---------:|
|               0   |                  -0.004 | 0.049 |            0.055 |      200 |
|               1.5 |                  -0.006 | 0.088 |            0.035 |      200 |

## EXP-07 Sample size

|     n | estimator      |   bias |   rmse |    sd |   coverage |   ci_width |   runtime_s |   n_reps |
|------:|:---------------|-------:|-------:|------:|-----------:|-----------:|------------:|---------:|
|   500 | regression_adj |  0.009 |  0.111 | 0.114 |    nan     |    nan     |       0.001 |       50 |
|   500 | ipw_hajek      |  0.031 |  0.123 | 0.122 |      1     |      0.702 |       0.002 |       50 |
|   500 | aipw_linear    |  0.005 |  0.118 | 0.122 |      0.92  |      0.419 |       0.012 |       50 |
|  1000 | regression_adj | -0.012 |  0.066 | 0.068 |    nan     |    nan     |       0.001 |       50 |
|  1000 | ipw_hajek      | -0.018 |  0.086 | 0.088 |      1     |      0.534 |       0.002 |       50 |
|  1000 | aipw_linear    | -0.014 |  0.069 | 0.072 |      0.98  |      0.301 |       0.013 |       50 |
|  5000 | regression_adj |  0.005 |  0.034 | 0.037 |    nan     |    nan     |       0.002 |       30 |
|  5000 | ipw_hajek      |  0.003 |  0.037 | 0.039 |      1     |      0.228 |       0.004 |       30 |
|  5000 | aipw_linear    |  0.003 |  0.035 | 0.038 |      0.967 |      0.13  |       0.019 |       30 |
| 10000 | regression_adj |  0.006 |  0.022 | 0.023 |    nan     |    nan     |       0.002 |       20 |
| 10000 | ipw_hajek      |  0.011 |  0.029 | 0.028 |      1     |      0.159 |       0.006 |       20 |
| 10000 | aipw_linear    |  0.008 |  0.024 | 0.024 |      0.95  |      0.09  |       0.027 |       20 |
| 50000 | regression_adj |  0.001 |  0.011 | 0.011 |    nan     |    nan     |       0.007 |       10 |
| 50000 | ipw_hajek      |  0.003 |  0.01  | 0.01  |      1     |      0.072 |       0.022 |       10 |
| 50000 | aipw_linear    |  0.002 |  0.01  | 0.011 |      1     |      0.041 |       0.09  |       10 |

## EXP-03b Honest causal forest vs T/X-learner

| scenario                | learner       |   pehe |   pehe_sd |   ate_abs_err |   pointwise_coverage |   runtime_s |   n_reps |
|:------------------------|:--------------|-------:|----------:|--------------:|---------------------:|------------:|---------:|
| randomized_nonlinear    | t_learner     |  0.472 |     0.03  |         0.035 |              nan     |       0.064 |       20 |
| randomized_nonlinear    | x_learner     |  0.331 |     0.026 |         0.034 |              nan     |       0.124 |       20 |
| randomized_nonlinear    | honest_forest |  0.166 |     0.031 |         0.031 |                0.69  |       0.356 |       20 |
| observational_nonlinear | t_learner     |  0.536 |     0.04  |         0.038 |              nan     |       0.064 |       20 |
| observational_nonlinear | x_learner     |  0.329 |     0.027 |         0.035 |              nan     |       0.123 |       20 |
| observational_nonlinear | honest_forest |  0.18  |     0.029 |         0.045 |                0.665 |       0.358 |       20 |

## EXP-09 Negative-control outcome test

|   hidden_strength |   mean_nco_estimate |   rejection_rate |   n_reps |
|------------------:|--------------------:|-----------------:|---------:|
|               0   |               0.001 |             0.12 |       50 |
|               0.5 |               0.385 |             1    |       50 |
|               1   |               0.675 |             1    |       50 |

## EXP-10 Difference-in-differences (synthetic panel)

| scenario             |   bias |   rmse |    sd |   coverage |   ci_width |   n_reps |   pretrend_rejection_rate |
|:---------------------|-------:|-------:|------:|-----------:|-----------:|---------:|--------------------------:|
| parallel_trends      | -0.004 |  0.088 | 0.089 |       0.97 |      0.323 |      100 |                      0.02 |
| violated_trends(0.3) |  0.896 |  0.901 | 0.089 |       0    |      0.323 |      100 |                      1    |

## EXP-11 Instrumental variables (synthetic)

| method                       |   bias |   rmse |    sd |   coverage |   ci_width |   n_reps |   mean_first_stage_F |
|:-----------------------------|-------:|-------:|------:|-----------:|-----------:|---------:|---------------------:|
| ols_confounded               |  0.334 |  0.335 | 0.021 |       0    |      0.092 |      100 |              nan     |
| 2sls_strong(1.0)             | -0.001 |  0.039 | 0.04  |       1    |      0.174 |      100 |              505.145 |
| 2sls_weak(0.08)              | -0.28  |  2.18  | 2.173 |       0.98 |     13.348 |      100 |                4.315 |
| 2sls_exclusion_violated(0.3) |  0.298 |  0.3   | 0.035 |       0    |      0.154 |      100 |              505.145 |

## EXP-08 Policy value

|   budget | policy    |   mean |   std |
|---------:|:----------|-------:|------:|
|      0.1 | oracle    |  0.645 | 0.031 |
|      0.1 | random    |  0.595 | 0.032 |
|      0.1 | t_learner |  0.643 | 0.032 |
|      0.1 | x_learner |  0.644 | 0.031 |
|      0.2 | oracle    |  0.795 | 0.031 |
|      0.2 | random    |  0.695 | 0.034 |
|      0.2 | t_learner |  0.79  | 0.033 |
|      0.2 | x_learner |  0.794 | 0.031 |
|      0.3 | oracle    |  0.945 | 0.031 |
|      0.3 | random    |  0.796 | 0.034 |
|      0.3 | t_learner |  0.933 | 0.033 |
|      0.3 | x_learner |  0.943 | 0.032 |
|      0.5 | oracle    |  1.24  | 0.034 |
|      0.5 | random    |  0.993 | 0.036 |
|      0.5 | t_learner |  1.185 | 0.038 |
|      0.5 | x_learner |  1.216 | 0.037 |
|      0.7 | oracle    |  1.345 | 0.037 |
|      0.7 | random    |  1.195 | 0.036 |
|      0.7 | t_learner |  1.332 | 0.038 |
|      0.7 | x_learner |  1.342 | 0.038 |
|      1   | oracle    |  1.495 | 0.037 |
|      1   | random    |  1.495 | 0.037 |
|      1   | t_learner |  1.462 | 0.042 |
|      1   | x_learner |  1.482 | 0.038 |

## EXP-08 IPW policy value vs truth

|   budget |   mean_true_value |   mean_ipw_estimate |   mean_abs_gap |   n_reps |
|---------:|------------------:|--------------------:|---------------:|---------:|
|      0.3 |             0.943 |               0.945 |          0.034 |       20 |

## Real-data results

_Not yet run: see docs/RUN_ON_YOUR_MACHINE.md._

## Figures

![EXP-05. Bias and RMSE of the ATE estimate versus observed-confounding strength on a synthetic linear SCM (n=2000, 100 replications per point). The naive difference in means drifts away from the truth as confounding grows, while regression adjustment, IPW and AIPW stay close to zero bias.](figures/fig01_confounding_bias.png)

*EXP-05. Bias and RMSE of the ATE estimate versus observed-confounding strength on a synthetic linear SCM (n=2000, 100 replications per point). The naive difference in means drifts away from the truth as confounding grows, while regression adjustment, IPW and AIPW stay close to zero bias.*

![EXP-06. ATE bias as a hidden confounder U gets stronger (n=2000, 100 replications). AIPW on observed covariates becomes strongly biased; the oracle AIPW that is given U stays unbiased. No estimator can fix a confounder it never sees.](figures/fig02_hidden_confounding.png)

*EXP-06. ATE bias as a hidden confounder U gets stronger (n=2000, 100 replications). AIPW on observed covariates becomes strongly biased; the oracle AIPW that is given U stays unbiased. No estimator can fix a confounder it never sees.*

![EXP-07. ATE RMSE versus sample size on the synthetic SCM (500 to 50,000; 10 to 50 replications depending on n). RMSE decreases at roughly the 1/sqrt(n) rate for all three adjusted estimators.](figures/fig03_sample_size.png)

*EXP-07. ATE RMSE versus sample size on the synthetic SCM (500 to 50,000; 10 to 50 replications depending on n). RMSE decreases at roughly the 1/sqrt(n) rate for all three adjusted estimators.*

![EXP-08. Exact policy value (from simulated potential outcomes) of top-k CATE policies versus budget on a randomized nonlinear SCM (n=4000, models trained on one half, evaluated on the other). Error bars are SD across replications. Learned policies track the oracle and beat random targeting at low and mid budgets.](figures/fig04_policy_value.png)

*EXP-08. Exact policy value (from simulated potential outcomes) of top-k CATE policies versus budget on a randomized nonlinear SCM (n=4000, models trained on one half, evaluated on the other). Error bars are SD across replications. Learned policies track the oracle and beat random targeting at low and mid budgets.*

![EXP-02. CATE error (sqrt-PEHE) of S-, T- and X-learners with a gradient-boosting base model on linear and nonlinear synthetic SCMs (n=2000, 30 replications).](figures/fig05_meta_learners_pehe.png)

*EXP-02. CATE error (sqrt-PEHE) of S-, T- and X-learners with a gradient-boosting base model on linear and nonlinear synthetic SCMs (n=2000, 30 replications).*

![DAG with treatment T, outcome Y, confounder X1, mediator M and collider C. The engine finds backdoor paths [['T', 'X1', 'Y'], ['T', 'X2', 'X1', 'Y']] and the minimal valid adjustment set(s) [['X1']]; the mediator M and collider C are excluded because adjusting for them would block the causal path or open a spurious one.](figures/ev01_dag_backdoor.png)

*DAG with treatment T, outcome Y, confounder X1, mediator M and collider C. The engine finds backdoor paths [['T', 'X1', 'Y'], ['T', 'X2', 'X1', 'Y']] and the minimal valid adjustment set(s) [['X1']]; the mediator M and collider C are excluded because adjusting for them would block the causal path or open a spurious one.*

![Estimated propensity score distributions for treated and control units in a synthetic observational dataset (n=3000, confounding strength 1). Good overlap means both groups have mass over the same range; thin tails signal positivity problems.](figures/ev02_propensity_overlap.png)

*Estimated propensity score distributions for treated and control units in a synthetic observational dataset (n=3000, confounding strength 1). Good overlap means both groups have mass over the same range; thin tails signal positivity problems.*

![Absolute standardized mean differences per covariate before and after inverse-probability weighting (same dataset as the overlap plot). The dashed line marks the common 0.1 threshold; weighting should move points left of it.](figures/ev03_covariate_balance.png)

*Absolute standardized mean differences per covariate before and after inverse-probability weighting (same dataset as the overlap plot). The dashed line marks the common 0.1 threshold; weighting should move points left of it.*

![Honest causal forest on a randomized nonlinear synthetic SCM (n=3000). Left: estimated versus true CATE (dashed line = perfect recovery). Right: distribution of estimated and true effects. True effects take two levels (step function in x0), which the forest smooths.](figures/ev04_cate_forest.png)

*Honest causal forest on a randomized nonlinear synthetic SCM (n=3000). Left: estimated versus true CATE (dashed line = perfect recovery). Right: distribution of estimated and true effects. True effects take two levels (step function in x0), which the forest smooths.*

![Cinelli–Hazlett style contour: OLS ATE estimate (1.05) after removing the bias of a hypothetical confounder with the stated partial R² values. The red line is where the estimate reaches zero. Robustness value RV=0.38. This quantifies how much confounding would overturn the result; it does not show whether such confounding exists.](figures/ev05_sensitivity_contour.png)

*Cinelli–Hazlett style contour: OLS ATE estimate (1.05) after removing the bias of a hypothetical confounder with the stated partial R² values. The red line is where the estimate reaches zero. Robustness value RV=0.38. This quantifies how much confounding would overturn the result; it does not show whether such confounding exists.*

![Event-study coefficients (95% cluster-robust CIs) on a synthetic panel, reference period just before treatment. With parallel trends the pre-period coefficients are near zero and the post-period coefficients show the true effect (2.0). With a built-in differential trend the pre-period coefficients drift and the Wald pre-trend test rejects.](figures/ev06_event_study.png)

*Event-study coefficients (95% cluster-robust CIs) on a synthetic panel, reference period just before treatment. With parallel trends the pre-period coefficients are near zero and the post-period coefficients show the true effect (2.0). With a built-in differential trend the pre-period coefficients drift and the Wald pre-trend test rejects.*
