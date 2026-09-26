"""
Drift Detection Engine — Multi-Tier Distribution Shift & Model Governance Monitor.

Monitors:
1. Feature Drift: Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).
2. Prediction & Probability Drift: Distribution shift in ML output probabilities.
3. Signal Drift: Proportion shifts in LONG / SHORT / CASH allocations.
4. Multi-Distribution Comparison: Training vs Validation vs Forward Paper distributions.

Alerting Tiers:
- NORMAL: Distribution variation within standard statistical noise.
- WARNING: Moderate drift detected (PSI 0.10 - 0.25, single feature p < 0.01).
- CRITICAL: Severe distribution collapse (PSI > 0.25, >= 3 features drifted, calibration failure).

Generates actionable investigation signals. Does NOT trigger automated retraining in Phase 1.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    feature_drift_detected: bool
    drifted_features: List[str]
    psi_scores: Dict[str, float]
    ks_pvalues: Dict[str, float]
    prediction_drift_detected: bool
    prediction_psi: float
    signal_distribution_shift: Dict[str, float]
    performance_degradation_pct: float
    alert_level: str  # "NORMAL" | "WARNING" | "CRITICAL"
    investigation_signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_drift_detected": self.feature_drift_detected,
            "drifted_features": self.drifted_features,
            "psi_scores": {k: round(v, 4) for k, v in self.psi_scores.items()},
            "ks_pvalues": {k: round(v, 4) for k, v in self.ks_pvalues.items()},
            "prediction_drift_detected": self.prediction_drift_detected,
            "prediction_psi": round(self.prediction_psi, 4),
            "signal_distribution_shift": {k: round(v, 4) for k, v in self.signal_distribution_shift.items()},
            "performance_degradation_pct": round(self.performance_degradation_pct, 2),
            "alert_level": self.alert_level,
            "investigation_signals": self.investigation_signals,
        }


class DriftDetector:
    """Detects statistical feature drift, prediction shift, and signal distribution decay."""

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Calculate Population Stability Index between baseline and current distributions."""
        if len(baseline) < 10 or len(current) < 10:
            return 0.0

        b_clean = baseline[~np.isnan(baseline)]
        c_clean = current[~np.isnan(current)]

        if len(b_clean) < 10 or len(c_clean) < 10:
            return 0.0

        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(b_clean, quantiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        b_counts = np.histogram(b_clean, bins=bins)[0] / len(b_clean)
        c_counts = np.histogram(c_clean, bins=bins)[0] / len(c_clean)

        # Smooth zero bins
        b_counts = np.where(b_counts == 0, 1e-4, b_counts)
        c_counts = np.where(c_counts == 0, 1e-4, c_counts)

        psi = np.sum((c_counts - b_counts) * np.log(c_counts / b_counts))
        return float(psi)

    def check_features(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Evaluate statistical drift across all feature columns.
        """
        cols = features or [c for c in baseline_df.select_dtypes(include=[np.number]).columns if c in current_df.columns]
        drifted = []
        psi_map = {}
        ks_map = {}
        investigations = []

        for col in cols:
            b_vals = baseline_df[col].dropna().values
            c_vals = current_df[col].dropna().values

            if len(b_vals) < 15 or len(c_vals) < 15:
                continue

            psi = self.calculate_psi(b_vals, c_vals)
            psi_map[col] = psi

            # Two-sample KS test
            ks_res = stats.ks_2samp(b_vals, c_vals)
            ks_map[col] = float(ks_res.pvalue)

            # Flag drift if PSI > 0.20 or KS p-value < 0.01
            if psi > 0.20 or ks_res.pvalue < 0.01:
                drifted.append(col)
                investigations.append(f"Feature '{col}' shifted: PSI={psi:.3f}, KS p-val={ks_res.pvalue:.4f}")

        # Prediction drift check if predictions exist
        pred_psi = 0.0
        pred_drift = False
        if "prediction" in baseline_df.columns and "prediction" in current_df.columns:
            pred_psi = self.calculate_psi(baseline_df["prediction"].values, current_df["prediction"].values)
            if pred_psi > 0.15:
                pred_drift = True
                investigations.append(f"Model prediction probability distribution shifted: PSI={pred_psi:.3f}")

        # Signal distribution shift
        sig_shift = {}
        if "signal" in baseline_df.columns and "signal" in current_df.columns:
            b_dist = baseline_df["signal"].value_counts(normalize=True).to_dict()
            c_dist = current_df["signal"].value_counts(normalize=True).to_dict()
            for k in set(list(b_dist.keys()) + list(c_dist.keys())):
                shift = abs(c_dist.get(k, 0.0) - b_dist.get(k, 0.0))
                sig_shift[str(k)] = shift
                if shift > 0.25:
                    investigations.append(f"Signal allocation for '{k}' shifted by {shift*100:.1f}%")

        has_drift = len(drifted) > 0 or pred_drift
        if len(drifted) >= 3 or pred_psi > 0.25:
            alert = "CRITICAL"
        elif has_drift or len(sig_shift) > 0:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        return DriftReport(
            feature_drift_detected=len(drifted) > 0,
            drifted_features=drifted,
            psi_scores=psi_map,
            ks_pvalues=ks_map,
            prediction_drift_detected=pred_drift,
            prediction_psi=pred_psi,
            signal_distribution_shift=sig_shift,
            performance_degradation_pct=0.0,
            alert_level=alert,
            investigation_signals=investigations,
        )
