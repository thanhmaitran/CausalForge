"""Unified causal dataset schema (binary treatment, v1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CausalDataset:
    X: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    feature_names: list[str]
    dataset_name: str
    treatment_name: str = "treatment"
    outcome_name: str = "outcome"
    propensity: np.ndarray | None = None  # TRUE propensity, only if known by design
    true_ite: np.ndarray | None = None
    true_cate: np.ndarray | None = None
    true_ate: float | None = None
    y0: np.ndarray | None = None  # potential outcomes (simulation only)
    y1: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def validate(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        n = self.X.shape[0]
        for name in ("treatment", "outcome"):
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"{name} must be 1D with length {n}")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must equal number of columns of X")
        if not set(np.unique(self.treatment)) <= {0, 1}:
            raise ValueError("treatment must be binary {0,1}")
        for name in ("propensity", "true_ite", "true_cate", "y0", "y1"):
            arr = getattr(self, name)
            if arr is not None and arr.shape[0] != n:
                raise ValueError(f"{name} length mismatch")
        if self.propensity is not None and (
            self.propensity.min() <= 0 or self.propensity.max() >= 1
        ):
            raise ValueError("propensity must lie strictly in (0,1)")

    def subset(self, idx: np.ndarray) -> CausalDataset:
        def s(a):
            return None if a is None else a[idx]

        return CausalDataset(
            X=self.X[idx],
            treatment=self.treatment[idx],
            outcome=self.outcome[idx],
            feature_names=self.feature_names,
            dataset_name=self.dataset_name,
            treatment_name=self.treatment_name,
            outcome_name=self.outcome_name,
            propensity=s(self.propensity),
            true_ite=s(self.true_ite),
            true_cate=s(self.true_cate),
            true_ate=None if self.true_ite is None else float(self.true_ite[idx].mean()),
            y0=s(self.y0),
            y1=s(self.y1),
            metadata=dict(self.metadata),
        )
