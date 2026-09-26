"""
Meta Model & Ensemble — Multi-Alpha, ML & Regime Synthesis with Disagreement Gate.

Combines:
- Technical features
- 12 Alpha signals
- ML calibrated predictions
- Market regime information
- Cross-sectional ranking

Circuit Breaker:
Measures model disagreement variance.
When alpha signals or ML models strongly clash:
- Forces decision to HOLD or scales down portfolio exposure
- Prevents manufactured confidence
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class MetaDecision:
    signal: str          # BUY | SELL | HOLD
    weight: float        # Allocation weight in [-1.0, 1.0]
    confidence: float    # Confidence in [0.0, 1.0]
    expected_return: float
    expected_risk: float
    disagreement_score: float  # Variance / divergence across signals
    exposure_scale: float      # Scale factor (1.0 = normal, < 1.0 = reduced, 0.0 = full stop)
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetaModel:
    """Ensemble synthesizing alpha families, ML output, and regime states."""

    def __init__(self, disagreement_threshold: float = 0.45):
        self.disagreement_threshold = disagreement_threshold

    def evaluate(
        self,
        alpha_signals: Dict[str, float],
        alpha_confidences: Dict[str, float],
        ml_prediction: Optional[Dict[str, Any]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        cross_sectional_rank: float = 0.50,
    ) -> MetaDecision:
        """
        Synthesize diverse quantitative inputs into a single risk-governed decision.
        """
        reasons: List[str] = []

        if not alpha_signals:
            return MetaDecision(
                signal="HOLD", weight=0.0, confidence=0.5,
                expected_return=0.0, expected_risk=0.02,
                disagreement_score=0.0, exposure_scale=0.0,
                reasons=["No alpha signals provided"],
            )

        # 1. Measure signal disagreement (dispersion among alpha signals)
        sig_values = list(alpha_signals.values())
        disagreement = float(np.std(sig_values)) if len(sig_values) > 1 else 0.0

        # 2. Weighted alpha composite
        weighted_sum = 0.0
        total_conf = 0.0
        for name, sig in alpha_signals.items():
            conf = alpha_confidences.get(name, 0.5)
            weighted_sum += sig * conf
            total_conf += conf

        composite_alpha = (weighted_sum / total_conf) if total_conf > 0 else 0.0

        # 3. Incorporate ML prediction if available
        ml_prob_up = 0.50
        ml_conf = 0.50
        if ml_prediction and ml_prediction.get("is_trained"):
            ml_prob_up = ml_prediction.get("probability_up", 0.50)
            ml_conf = ml_prediction.get("confidence", 0.50)
            # Blend alpha and ML: 60% alpha library, 40% ML
            ml_signal = (ml_prob_up - 0.50) * 2.0  # map to [-1, 1]
            blended_signal = composite_alpha * 0.60 + ml_signal * 0.40
            reasons.append(f"Blended Alpha ({composite_alpha:.2f}) + ML ({ml_signal:.2f})")
        else:
            blended_signal = composite_alpha
            reasons.append(f"Alpha composite: {composite_alpha:.2f}")

        # 4. Regime Adjustment
        regime_scale = 1.0
        if regime_info:
            regime = regime_info.get("regime", "SIDEWAYS")
            risk_mode = regime_info.get("risk_mode", "NEUTRAL")
            if risk_mode == "RISK_OFF" or "BEAR" in regime or "HIGH_VOLATILITY" in regime:
                regime_scale = 0.50  # Cut exposure in half during hostile regimes
                reasons.append(f"Hostile regime ({regime}) -> exposure scaled 50%")
            elif "BULL" in regime:
                regime_scale = 1.0
                reasons.append(f"Favorable regime ({regime})")

        # 5. Model Disagreement Gate (Circuit Breaker)
        exposure_scale = regime_scale
        if disagreement > self.disagreement_threshold:
            # Alpha models conflict heavily (e.g. Trend says BUY, Mean Reversion says SELL)
            exposure_scale *= max(0.20, 1.0 - (disagreement * 1.5))
            reasons.append(f"⚠️ High Model Disagreement ({disagreement:.2f} > {self.disagreement_threshold}) -> REDUCING EXPOSURE")

        # 6. Final Decision & Confidence
        effective_weight = blended_signal * exposure_scale
        mean_confidence = float(np.mean(list(alpha_confidences.values()))) if alpha_confidences else 0.5
        # Penalize confidence if disagreement is elevated
        calibrated_confidence = round(max(0.10, mean_confidence * (1.0 - disagreement * 0.5)), 3)

        if effective_weight > 0.20 and calibrated_confidence >= 0.55:
            decision_signal = "BUY"
        elif effective_weight < -0.20 and calibrated_confidence >= 0.55:
            decision_signal = "SELL"
        else:
            decision_signal = "HOLD"
            effective_weight = 0.0
            reasons.append("Hold condition: signal below threshold or insufficient confidence")

        expected_ret = round(effective_weight * 0.05, 4)
        expected_risk = round(0.015 + (disagreement * 0.02), 4)

        return MetaDecision(
            signal=decision_signal,
            weight=round(effective_weight, 3),
            confidence=calibrated_confidence,
            expected_return=expected_ret,
            expected_risk=expected_risk,
            disagreement_score=round(disagreement, 3),
            exposure_scale=round(exposure_scale, 3),
            reasons=reasons,
        )
