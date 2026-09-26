"""
Volatility Alpha Family — Exploits volatility compression cycles (squeezes) and explosive expansion.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class VolatilityAlpha(BaseAlpha):
    """Detects volatility compression (quiet market) preparing for expansion."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "compression_threshold": 1.25,
            "expansion_threshold": 1.15,
        }

    def required_features(self) -> List[str]:
        return ["vol_compression", "vol_expansion", "atr_ratio_10_20", "ret_1"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        comp = features.get("vol_compression", 1.0)
        exp = features.get("vol_expansion", 1.0)
        ret1 = features.get("ret_1", 0.0)

        # Extreme compression followed by directional poke
        if comp > self._parameters.get("compression_threshold", 1.25):
            if ret1 > 0.005:
                return 0.75
            elif ret1 < -0.005:
                return -0.75

        # Active volatility expansion
        if exp > self._parameters.get("expansion_threshold", 1.15):
            if ret1 > 0.01:
                return 1.0
            elif ret1 < -0.01:
                return -1.0

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        comp = features.get("vol_compression", 1.0)
        return min(0.90, max(0.35, 0.50 + abs(comp - 1.0) * 0.3))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        comp = features.get("vol_compression", 1.0)
        exp = features.get("vol_expansion", 1.0)
        return {
            "reason_codes": [f"Vol Compression ratio: {comp:.2f}", f"Vol Expansion ratio: {exp:.2f}"],
            "expected_horizon": "short",
            "expected_return": 0.04,
            "expected_risk": 0.02,
            "regime_compatibility": "HIGH_VOLATILITY,LOW_VOLATILITY",
        }

    def regime_compatibility(self) -> str:
        return "HIGH_VOLATILITY"
