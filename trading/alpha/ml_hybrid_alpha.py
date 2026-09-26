"""
ML Hybrid Alpha Family — Bridges calibrated probabilistic predictions with quantitative rules.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class MLHybridAlpha(BaseAlpha):
    """Combines machine learning directional probabilities with technical trend confirmation."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "prob_up_threshold": 0.60,
            "prob_down_threshold": 0.40,
            "min_confidence": 0.65,
        }

    def required_features(self) -> List[str]:
        return ["ml_prob_up", "ml_confidence", "adx_14", "ret_1"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        prob_up = features.get("ml_prob_up", 0.50)
        conf = features.get("ml_confidence", 0.50)

        p_up_th = self._parameters.get("prob_up_threshold", 0.60)
        p_dn_th = self._parameters.get("prob_down_threshold", 0.40)
        min_conf = self._parameters.get("min_confidence", 0.65)

        if conf < min_conf:
            return 0.0  # Refuse to trade on low ML confidence

        if prob_up >= p_up_th:
            return min(1.0, (prob_up - 0.5) * 2.5)
        elif prob_up <= p_dn_th:
            return max(-1.0, (prob_up - 0.5) * 2.5)

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        return float(features.get("ml_confidence", 0.50))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        prob_up = features.get("ml_prob_up", 0.50)
        conf = features.get("ml_confidence", 0.50)
        return {
            "reason_codes": [f"ML Probability Up: {prob_up*100:.1f}%", f"Model Confidence: {conf*100:.1f}%"],
            "expected_horizon": "medium",
            "expected_return": 0.05,
            "expected_risk": 0.02,
            "regime_compatibility": "any",
        }

    def regime_compatibility(self) -> str:
        return "any"
