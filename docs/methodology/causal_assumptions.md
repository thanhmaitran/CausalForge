# Causal assumptions and estimands

## Estimands
Potential outcomes Y(1), Y(0); ITE_i = Y_i(1) - Y_i(0); ATE = E[Y(1)-Y(0)];
CATE(x) = E[Y(1)-Y(0) | X=x]; ATT = E[Y(1)-Y(0) | T=1].

## Identification assumptions
1. **Consistency** — observed outcome equals the potential outcome of the received treatment.
2. **Conditional exchangeability** — (Y(0),Y(1)) independent of T given X. Untestable; violated in EXP-06 by construction.
3. **Positivity** — 0 < P(T=1|X) < 1. Diagnosed via overlap, ESS and extreme-weight reports; not silently clipped.
4. **SUTVA** — no interference, no hidden treatment versions.

A numerically accurate estimator does not establish identification.

## Dataset-specific rules
* **Criteo**: analyse assignment (`treatment`) as intention-to-treat; `exposure` is post-treatment and is never a covariate. No individual counterfactuals, so no PEHE.
* **IHDP / ACIC / synthetic**: ground truth available, so PEHE and ATE error are valid.
* **Synthetic SCM**: truth is computed from structural equations, never from an estimator.
