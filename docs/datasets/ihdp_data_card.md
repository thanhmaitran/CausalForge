# IHDP (NPCI semi-synthetic) — data card (unverified until downloaded)
Source: Fred Johansson's NPCI files (`ihdp_npci_1-100.{train,test}.npz`), 100 replications; real
covariates, simulated outcomes. Arrays: x (n,25,100), t, yf, ycf, mu0, mu1. Ground truth:
ITE = mu1 - mu0. Treatment is biased by construction (non-random removal), so it is an
observational benchmark. Do not enlarge the sample and still call it IHDP.
