"""Loaders for IHDP (NPCI .npz), ACIC 2016 (x.csv + zymu_*.csv) and LaLonde/NSW csv.

Formats are as published; each loader raises an informative error on schema mismatch rather
than guessing. Verify the ACIC file layout against the Zenodo record you downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causalforge.data.schemas import CausalDataset

# ---------------------------------------------------------------- IHDP
IHDP_KEYS = {"x", "t", "yf", "ycf", "mu0", "mu1"}


def load_ihdp(path: str | Path, replication: int = 0) -> CausalDataset:
    z = np.load(path)
    missing = IHDP_KEYS - set(z.files)
    if missing:
        raise ValueError(f"{path}: missing IHDP arrays {sorted(missing)}; found {z.files}")
    x = z["x"]
    if x.ndim == 2:  # single replication stored flat
        x, t, yf, ycf, mu0, mu1 = (
            a[..., None] for a in (x, z["t"], z["yf"], z["ycf"], z["mu0"], z["mu1"])
        )
    else:
        t, yf, ycf, mu0, mu1 = z["t"], z["yf"], z["ycf"], z["mu0"], z["mu1"]
    r = replication
    if not 0 <= r < x.shape[2]:
        raise IndexError(f"replication {r} out of range 0..{x.shape[2] - 1}")
    T = t[:, r].astype(int)
    y1 = np.where(T == 1, yf[:, r], ycf[:, r])
    y0 = np.where(T == 1, ycf[:, r], yf[:, r])
    ite = mu1[:, r] - mu0[:, r]
    return CausalDataset(
        X=x[:, :, r].astype(float),
        treatment=T,
        outcome=yf[:, r].astype(float),
        feature_names=[f"x{i}" for i in range(x.shape[1])],
        dataset_name="ihdp",
        true_ite=ite,
        true_cate=ite.copy(),
        true_ate=float(ite.mean()),
        y0=y0,
        y1=y1,
        metadata={"replication": r, "ground_truth": "mu1-mu0 (noiseless)"},
    )


# ---------------------------------------------------------------- ACIC 2016
def load_acic(x_csv: str | Path, zymu_csv: str | Path) -> CausalDataset:
    X = pd.read_csv(x_csv)
    zy = pd.read_csv(zymu_csv)
    need = {"z", "y0", "y1", "mu0", "mu1"}
    if not need <= set(zy.columns):
        raise ValueError(f"{zymu_csv}: expected columns {sorted(need)}, got {list(zy.columns)}")
    if len(X) != len(zy):
        raise ValueError("x.csv and zymu file have different row counts")
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    z = zy["z"].to_numpy(int)
    y = np.where(z == 1, zy["y1"], zy["y0"]).astype(float)
    ite = (zy["mu1"] - zy["mu0"]).to_numpy(float)
    return CausalDataset(
        X=X.to_numpy(float),
        treatment=z,
        outcome=y,
        feature_names=list(X.columns),
        dataset_name="acic2016",
        true_ite=ite,
        true_cate=ite.copy(),
        true_ate=float(ite.mean()),
        y0=zy["y0"].to_numpy(float),
        y1=zy["y1"].to_numpy(float),
        metadata={"zymu_file": Path(zymu_csv).name},
    )


# ---------------------------------------------------------------- LaLonde
LALONDE_COVS = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]


def read_lalonde_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "treat" not in df and "t" in df:
        df = df.rename(columns={"t": "treat"})
    df = df.rename(columns={"education": "educ", "hispanic": "hispan"})
    if "race" in df:
        df["black"] = (df["race"] == "black").astype(int)
        df["hispan"] = (df["race"].isin(["hispan", "hispanic"])).astype(int)
    if "hisp" in df and "hispan" not in df:
        df = df.rename(columns={"hisp": "hispan"})
    miss = {"treat", "re78", *LALONDE_COVS} - set(df.columns)
    if miss:
        raise ValueError(f"{path}: missing LaLonde columns {sorted(miss)}")
    return df


def lalonde_dataset(df: pd.DataFrame, name: str = "lalonde") -> CausalDataset:
    return CausalDataset(
        X=df[LALONDE_COVS].to_numpy(float),
        treatment=df["treat"].to_numpy(int),
        outcome=df["re78"].to_numpy(float),
        feature_names=list(LALONDE_COVS),
        dataset_name=name,
        outcome_name="re78",
        metadata={
            "pre_outcome": df["re75"].to_numpy(float).tolist()
            if len(df) < 5
            else "re75 column in frame"
        },
    )
