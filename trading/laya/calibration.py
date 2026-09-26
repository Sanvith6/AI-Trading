"""
Laya Model Probability Calibration Engine.

Enforces Directives 8 & 17:
- Probabilities are stored as `decision_probability` / `selected_choice_probability`.
- Calibration is evaluated independently from trading profitability:
    1. Brier Score (mean squared error of probability forecasts)
    2. Multi-class / Binary Log Loss
    3. Expected Calibration Error (ECE)
    4. Reliability diagrams / Confidence buckets
- Calibration success is NOT treated as evidence of alpha/profitability.
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np


class LayaCalibrationEngine:
    """Rigorous statistical calibration evaluator for Laya probabilistic outputs."""

    @staticmethod
    def brier_score(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculates standard Brier Score: (1/N) * sum((prob - true)^2).
        Ranges from 0 (perfect calibration) to 1.
        """
        if not y_true or len(y_true) != len(y_prob):
            return 0.0

        n = len(y_true)
        sq_errs = [(p - y) ** 2 for y, p in zip(y_true, y_prob)]
        return round(float(sum(sq_errs) / n), 4)

    @staticmethod
    def log_loss(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
        """Calculates binary log loss (cross-entropy) with numerical clipping."""
        if not y_true or len(y_true) != len(y_prob):
            return 0.0

        n = len(y_true)
        clipped_probs = [min(max(p, eps), 1.0 - eps) for p in y_prob]
        losses = [
            -(y * math.log(p) + (1 - y) * math.log(1 - p))
            for y, p in zip(y_true, clipped_probs)
        ]
        return round(float(sum(losses) / n), 4)

    @staticmethod
    def reliability_buckets(
        y_true: List[int],
        y_prob: List[float],
        n_bins: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Partitions predictions into confidence buckets to evaluate calibration curves.
        Each bin reports:
          - bin_range: [low, high)
          - mean_predicted_prob: average model confidence
          - empirical_win_rate: actual positive outcome frequency
          - count: sample count
        """
        if not y_true or len(y_true) != len(y_prob):
            return []

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bucket_results = []

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        for i in range(n_bins):
            low = bins[i]
            high = bins[i + 1]

            if i == n_bins - 1:
                mask = (y_prob_arr >= low) & (y_prob_arr <= high)
            else:
                mask = (y_prob_arr >= low) & (y_prob_arr < high)

            count = int(np.sum(mask))
            if count > 0:
                mean_prob = float(np.mean(y_prob_arr[mask]))
                empirical_rate = float(np.mean(y_true_arr[mask]))
            else:
                mean_prob = round((low + high) / 2.0, 3)
                empirical_rate = 0.0

            bucket_results.append({
                "bin_index": i,
                "bin_range": f"[{low:.1f}, {high:.1f}]",
                "mean_predicted_prob": round(mean_prob, 4),
                "empirical_win_rate": round(empirical_rate, 4),
                "count": count,
                "calibration_gap": round(abs(mean_prob - empirical_rate), 4) if count > 0 else 0.0,
            })

        return bucket_results

    @classmethod
    def expected_calibration_error(
        cls,
        y_true: List[int],
        y_prob: List[float],
        n_bins: int = 10,
    ) -> float:
        """Calculates Expected Calibration Error (ECE) weighted by bin sample count."""
        buckets = cls.reliability_buckets(y_true, y_prob, n_bins=n_bins)
        total_samples = len(y_true)
        if total_samples == 0:
            return 0.0

        weighted_ece = sum(
            (b["count"] / total_samples) * b["calibration_gap"]
            for b in buckets if b["count"] > 0
        )
        return round(float(weighted_ece), 4)

    @classmethod
    def generate_calibration_report(
        cls,
        task: str,
        y_true: List[int],
        y_prob: List[float],
    ) -> Dict[str, Any]:
        """Produces full statistical calibration profile."""
        brier = cls.brier_score(y_true, y_prob)
        ll = cls.log_loss(y_true, y_prob)
        ece = cls.expected_calibration_error(y_true, y_prob)
        buckets = cls.reliability_buckets(y_true, y_prob)

        return {
            "task": task,
            "sample_count": len(y_true),
            "brier_score": brier,
            "log_loss": ll,
            "expected_calibration_error": ece,
            "is_well_calibrated": ece < 0.10,
            "confidence_buckets": buckets,
            "methodology_note": (
                "Calibration measures forecast accuracy of probabilities against empirical frequency; "
                "it is strictly independent of trading strategy PnL."
            ),
        }
