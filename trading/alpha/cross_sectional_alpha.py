"""
Cross-Sectional Alpha Family — Generates alpha by ranking assets cross-sectionally.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class CrossSectionalAlpha(BaseAlpha):
    """Generates signals based on cross-sectional return rank and relative momentum."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "top_percentile": 0.80,
            "bottom_percentile": 0.20,
        }

    def required_features(self) -> List[str]:
        return ["return_rank", "momentum_5_20", "ret_20"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        rank = features.get("return_rank", 0.5)
        top_p = self._parameters.get("top_percentile", 0.80)
        bot_p = self._parameters.get("bottom_percentile", 0.20)

        # Top quintile leader -> Long
        if rank >= top_p:
            return 1.0
        elif rank >= 0.65:
            return 0.5

        # Bottom quintile laggard -> Short / Underweight
        if rank <= bot_p:
            return -1.0
        elif rank <= 0.35:
            return -0.5

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        rank = features.get("return_rank", 0.5)
        dist_from_median = abs(rank - 0.5)
        return min(0.95, max(0.30, 0.50 + dist_from_median * 0.9))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        rank = features.get("return_rank", 0.5)
        return {
            "reason_codes": [f"Cross-sectional return rank: {rank*100:.1f}th percentile"],
            "expected_horizon": "medium_to_long",
            "expected_return": 0.06,
            "expected_risk": 0.03,
            "regime_compatibility": "BULL,RISK_ON",
        }

    def regime_compatibility(self) -> str:
        return "BULL"
