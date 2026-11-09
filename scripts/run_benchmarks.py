"""Convenience: synthetic suite, then real-data suite (skips missing datasets), then report."""

import subprocess
import sys

for s in ("run_experiments.py", "run_real_benchmarks.py", "generate_report.py"):
    subprocess.run([sys.executable, f"scripts/{s}"], check=True)
