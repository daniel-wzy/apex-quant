"""
quant/model.py — QuantModel wrapper (scaler + gradient-boosted estimator).

A small picklable bundle so the SAME object is used in training, backtest, and
the live decision path. Holds the fitted StandardScaler, the fitted estimator,
the exact feature-column order, and metadata (kind/target/metrics).

Using one class (rather than a raw sklearn Pipeline) keeps the column order
explicit and lets the live path pass a dict/DataFrame without worrying about
column ordering or scaling drift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd


@dataclass
class QuantModel:
    estimator: object                      # fitted XGB classifier or regressor
    scaler: object                         # fitted StandardScaler
    feature_columns: list                  # exact input order
    kind: str                              # "classifier" | "regressor"
    target: str                            # label column it was trained on
    metadata: dict = field(default_factory=dict)

    # ── inference ────────────────────────────────────────────────────────
    def _matrix(self, X) -> np.ndarray:
        """Coerce a dict/Series/DataFrame to the scaled feature matrix."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return self.scaler.transform(X.to_numpy(dtype=float))

    def predict_proba(self, X) -> np.ndarray:
        """P(positive class) for the classifier."""
        proba = self.estimator.predict_proba(self._matrix(X))
        return proba[:, 1]

    def predict(self, X) -> np.ndarray:
        """Regression output (e.g., expected forward PnL%)."""
        return self.estimator.predict(self._matrix(X))

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "QuantModel":
        return joblib.load(path)
