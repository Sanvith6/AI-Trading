"""
Relative Strength Alpha Family — Evaluates momentum persistence relative to benchmark and sector.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class RelativeStrengthAlpha(BaseAlpha):
    """Generates signals based on medium-term relative strength and momentum differentials."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "mom_diff_threshold": 0.03,  # 3% differential
            "min_persistence": 0.60,
        }

    def required_features(self) -> List[str]:
        return ["momentum_5_20", "trend_persistence", "ret_20"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        mom_diff = features.get("momentum_5_20", 0.0)
        persist = features.get("trend_persistence", 0.5)

        th = self._parameters.get("mom_diff_threshold", 0.03)
        min_p = self._parameters.get("min_persistence", 0.60)

        if mom_diff > th and persist >= min_p:
            return 1.0
        elif mom_diff > 0.01 and persist >= 0.55:
            return 0.5
        elif mom_diff < -th and persist <= (1.0 - min_p):
            return -1.0
        elif mom_diff < -0.01 and persist <= 0.45:
            return -0.5

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        persist = features.get("trend_persistence", 0.5)
        return min(0.95, max(0.35, 0.40 + abs(persist - 0.5) * 1.1))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        mom_diff = features.get("momentum_5_20", 0.0)
        persist = features.get("trend_persistence", 0.5)
        return {
            "reason_codes": [f"Momentum Differential 5-20d: {mom_diff*100:.2f}%", f"Trend Persistence: {persist*100:.1f}%"],
            "expected_horizon": "medium",
            "expected_return": 0.05,
            "expected_risk": 0.025,
            "regime_compatibility": "BULL,TRENDING",
        }

    def regime_compatibility(self) -> str:
        return "BULL"
