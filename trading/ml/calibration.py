"""
Model Calibration & Overconfidence Analysis — Evaluates probability realism.

Computes:
- Brier Score (lower is better, 0 = perfect)
- Log Loss
- Calibration Curve & Accuracy by confidence bucket
- Overconfidence Penalty
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
try:
    from sklearn.metrics import brier_score_loss, log_loss
except ImportError:
    def brier_score_loss(y_true, y_prob):
        y_true = np.asarray(y_true).astype(float)
        y_prob = np.asarray(y_prob).astype(float)
        return float(np.mean((y_prob - y_true) ** 2))

    def log_loss(y_true, y_prob, eps=1e-15):
        y_true = np.asarray(y_true).astype(float)
        y_prob = np.clip(np.asarray(y_prob).astype(float), eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


class ModelCalibrator:
    """Evaluates whether model predicted probabilities match observed frequencies."""

    @staticmethod
    def evaluate_calibration(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate probability calibration across confidence buckets.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1.0 - 1e-6)

        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            return {
                "brier_score": 0.25,
                "log_loss": 0.693,
                "overconfidence_penalty": 0.0,
                "is_calibrated": False,
                "note": "Insufficient data for calibration curve",
            }

        brier = float(brier_score_loss(y_true, y_prob))
        ll = float(log_loss(y_true, y_prob))

        # Confidence buckets
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1

        bucket_reports = []
        calibration_gap = 0.0

        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                mean_pred = float(np.mean(y_prob[mask]))
                actual_acc = float(np.mean(y_true[mask]))
                gap = abs(mean_pred - actual_acc)
                calibration_gap += gap * (np.sum(mask) / len(y_true))
                bucket_reports.append({
                    "bucket": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                    "count": int(np.sum(mask)),
                    "mean_predicted": round(mean_pred, 3),
                    "actual_accuracy": round(actual_acc, 3),
                    "gap": round(gap, 3),
                })

        # Overconfidence penalty: when predicted prob is far higher than actual win rate
        overconfidence_penalty = round(max(0.0, float(calibration_gap - 0.08)), 3)
        is_calibrated = brier < 0.22 and overconfidence_penalty < 0.05

        return {
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "mean_calibration_gap": round(calibration_gap, 4),
            "overconfidence_penalty": overconfidence_penalty,
            "is_calibrated": is_calibrated,
            "buckets": bucket_reports,
        }
