test:
	PYTHONPATH=src python tests/run_tests.py
experiments:
	PYTHONPATH=src python scripts/run_experiments.py
report:
	PYTHONPATH=src python scripts/generate_report.py
