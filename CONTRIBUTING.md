# Contributing
Run `pytest tests -v` and `ruff check .` before a PR. Commit style: feat/fix/test/docs/refactor/perf/research.
Statistical changes need a test that checks the mathematical behaviour (recovery of truth, null calibration), not just coverage.
Never commit raw datasets, credentials or hard-coded result numbers.
