"""Run EXP-01..EXP-08. Usage: PYTHONPATH=src python scripts/run_experiments.py [--quick]"""

import sys

from causalforge.experiments.runner import run_all

if __name__ == "__main__":
    res = run_all(quick="--quick" in sys.argv)
    for k, v in res.items():
        print(f"\n== {k} ==\n{v.round(3).to_string(index=False)}")
