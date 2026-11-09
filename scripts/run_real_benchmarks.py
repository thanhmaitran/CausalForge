"""Run benchmarks on whatever real datasets are present under data/raw/.

    PYTHONPATH=src python scripts/run_real_benchmarks.py [--ihdp-reps 100] [--acic-max 30]
                                                         [--criteo-rows 1000000]
Writes CSVs to artifacts/real/. Missing datasets are skipped with a clear message.
Expected files (see docs/RUN_ON_YOUR_MACHINE.md):
  data/raw/ihdp/ihdp_npci_1-100.train.npz (+ .test.npz)
  data/raw/acic2016/**/x.csv and zymu_*.csv
  data/raw/lalonde/nsw_experimental.csv (+ comparison.csv, e.g. CPS/PSID controls)
  data/raw/criteo/criteo-research-uplift-v2.1.csv.gz
"""

import argparse
import json
from pathlib import Path

from causalforge.experiments import real
from causalforge.experiments.runner import environment
from causalforge.experiments.tracking import file_sha256, track_run

ap = argparse.ArgumentParser()
ap.add_argument("--ihdp-reps", type=int, default=100)
ap.add_argument("--acic-max", type=int, default=30)
ap.add_argument("--criteo-rows", type=int, default=1_000_000)
a = ap.parse_args()
R = Path("data/raw")
OUT = Path("artifacts/real")
OUT.mkdir(parents=True, exist_ok=True)
done, skipped = [], []


def save(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n== {name} ==\n{df.round(3).to_string(index=False)}")


tr = R / "ihdp/ihdp_npci_1-100.train.npz"
if tr.exists():
    te = R / "ihdp/ihdp_npci_1-100.test.npz"
    with track_run("real-ihdp", {"reps": a.ihdp_reps}, dataset_files=[str(tr), str(te)]):
        ate, cate = real.run_ihdp(str(tr), str(te) if te.exists() else None, a.ihdp_reps)
    save("ihdp_ate", ate)
    save("ihdp_cate", cate)
    done.append("ihdp")
else:
    skipped.append(f"ihdp: {tr} not found")

acic = R / "acic2016"
if list(acic.rglob("zymu_*.csv")):
    with track_run("real-acic", {"max_files": a.acic_max}):
        ate, cate, n = real.run_acic(str(acic), a.acic_max)
    save("acic_ate", ate)
    save("acic_cate", cate)
    done.append(f"acic ({n} datasets)")
else:
    skipped.append("acic2016: no zymu_*.csv under data/raw/acic2016 (extract the Zenodo archive)")

nsw = R / "lalonde/nsw_experimental.csv"
if nsw.exists():
    comp = R / "lalonde/comparison.csv"
    save("lalonde", real.run_lalonde(str(nsw), str(comp) if comp.exists() else None))
    done.append("lalonde")
else:
    skipped.append(f"lalonde: {nsw} not found")

cr = R / "criteo/criteo-research-uplift-v2.1.csv.gz"
if cr.exists():
    with track_run("real-criteo", {"rows": a.criteo_rows}, dataset_files=[str(cr)]):
        itt, au, qi, po = real.run_criteo(str(cr), a.criteo_rows)
    save("criteo_itt", itt)
    save("criteo_auuc", au)
    save("criteo_qini", qi)
    save("criteo_policy_ipw", po)
    done.append("criteo")
else:
    skipped.append(f"criteo: {cr} not found")

(OUT / "summary.json").write_text(
    json.dumps(
        {
            "completed": done,
            "skipped": skipped,
            "environment": environment(),
            "checksums": {
                str(p): file_sha256(p) for p in R.rglob("*") if p.is_file() and p.name != ".gitkeep"
            },
        },
        indent=2,
    )
)
print("\nCompleted:", done, "\nSkipped:", skipped)
