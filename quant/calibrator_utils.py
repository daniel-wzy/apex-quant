"""
quant/calibrator_utils.py — Shared calibrator load/predict utility.

Avoids __main__ pickle issues by loading the underlying sklearn model.
"""
from __future__ import annotations

import numpy as np
import joblib


def load_calibrator(path: str):
    """Load calibrator bundle from disk. Returns a callable that predicts calibrated probs."""
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        model = bundle["model"]
        cal_type = bundle.get("type", "unknown")
    else:
        # Legacy: plain sklearn object
        model = bundle
        cal_type = "legacy"

    class CalibratorWrapper:
        def __init__(self, m, t):
            self.model = m
            self.type = t

        def predict(self, probs: np.ndarray) -> np.ndarray:
            probs = np.asarray(probs, dtype=float)
            if self.type == "isotonic":
                return np.clip(self.model.predict(probs), 0.0, 1.0)
            elif self.type == "platt":
                return np.clip(self.model.predict_proba(probs.reshape(-1, 1))[:, 1], 0.0, 1.0)
            else:
                # Legacy isotonic
                try:
                    return np.clip(self.model.predict(probs), 0.0, 1.0)
                except Exception:
                    return np.clip(self.model.predict_proba(probs.reshape(-1, 1))[:, 1], 0.0, 1.0)

    return CalibratorWrapper(model, cal_type)
