"""Convert Dehejia-Wahba NSW/CPS files (.dta or .csv) to the CSV layout CausalForge expects.

    python scripts/prepare_lalonde.py nsw_dw.dta data/raw/lalonde/nsw_experimental.csv
    python scripts/prepare_lalonde.py cps_controls.dta data/raw/lalonde/comparison.csv

Verify the file names/URLs at the source (https://users.nber.org/~rdehejia/data/) and record
them in data/manifests/datasets.yaml. The comparison file's rows with treat==0 are used as
the observational control group.
"""

import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_stata(src) if src.endswith(".dta") else pd.read_csv(src)
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"education": "educ", "hispanic": "hispan"})
need = {"treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"}
missing = need - set(df.columns)
if missing:
    sys.exit(f"missing columns {sorted(missing)}; found {list(df.columns)}")
df[sorted(need)].to_csv(dst, index=False)
print("wrote", dst, len(df), "rows")
