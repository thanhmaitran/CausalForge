# Architecture

`CausalDataset` is the single interface between data and estimators. Estimators return
`EstimateResult` (estimate, SE, CI, diagnostics, estimand). Layers: `data` → `graphs` /
`estimators` / `meta_learners` / `causal_ml` → `sensitivity` → `evaluation` / `decision` →
`experiments` → `reporting`. Algorithms live only in `src/causalforge`; any future API or UI
must call into it and contain no statistics of its own.
