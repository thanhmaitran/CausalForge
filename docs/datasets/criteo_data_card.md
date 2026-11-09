# Criteo Uplift v2.1 — data card (draft; unverified until downloaded)

Source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ · License: CC BY-NC-SA 4.0.
Columns: f0–f11 (covariates), treatment, conversion, visit, exposure.
Aggregated and resampled from multiple incrementality tests: do not assume known per-row
propensities. Use ITT on `treatment`; treat `exposure` analyses as a separate estimand with
noncompliance assumptions. Record SHA-256 with `scripts/download_data.py`.
