# Evidence screenshots to capture on your machine (each needs a one-line description)

| # | Screenshot | How |
|---|---|---|
| 01 | Repository structure | VS Code explorer |
| 02 | Dataset validation | output of `scripts/validate_data.py` |
| 03–05 | DAG, backdoor, overlap | already generated: `reports/benchmark_report/figures/ev01..ev03` |
| 06–08 | Classical / meta-learner / DML results | `artifacts/real/*.csv` + report |
| 09 | Sensitivity curves | `ev05_sensitivity_contour.png` |
| 10 | Policy evaluation | `fig04_policy_value.png` / Criteo IPW policy table |
| 11 | MLflow UI | http://127.0.0.1:5000 |
| 12 | Dashboard overview | http://localhost:3000 (or `npm run dev`) |
| 13 | API Swagger | http://127.0.0.1:8000/docs |
| 14 | Docker services | `docker compose ps` |
| 15 | Passing tests + CI | `pytest` output; GitHub Actions page |
