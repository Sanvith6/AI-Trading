"""
Breakout Alpha Family — Capitalizes on multi-day range expansions and structural breakouts.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class BreakoutAlpha(BaseAlpha):
    """Detects breakouts to new 20-bar highs or lows confirmed by volume expansion."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "dist_high_threshold": -0.5,  # Within 0.5% of 20d high
            "dist_low_threshold": 0.5,    # Within 0.5% of 20d low
            "min_relative_volume": 1.2,
        }

    def required_features(self) -> List[str]:
        return ["dist_from_high_20", "dist_from_low_20", "relative_volume", "range_10"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        dh = features.get("dist_from_high_20", -5.0)
        dl = features.get("dist_from_low_20", 5.0)
        rvol = features.get("relative_volume", 1.0)

        min_rvol = self._parameters.get("min_relative_volume", 1.2)

        # Bullish breakout
        if dh >= self._parameters.get("dist_high_threshold", -0.5) and rvol >= min_rvol:
            return 1.0
        elif dh >= -1.5 and rvol >= 1.0:
            return 0.5

        # Bearish breakdown
        if dl <= self._parameters.get("dist_low_threshold", 0.5) and rvol >= min_rvol:
            return -1.0
        elif dl <= 1.5 and rvol >= 1.0:
            return -0.5

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        rvol = features.get("relative_volume", 1.0)
        return min(0.95, max(0.30, 0.50 + min(0.40, (rvol - 1.0) * 0.2)))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        dh = features.get("dist_from_high_20", 0.0)
        rvol = features.get("relative_volume", 1.0)
        reasons = []
        if dh >= -0.5:
            reasons.append(f"At/Near 20-day high (dist {dh:.2f}%)")
        if rvol > 1.2:
            reasons.append(f"Volume confirmation: {rvol:.2f}x average")

        return {
            "reason_codes": reasons or ["No breakout confirmed"],
            "expected_horizon": "medium",
            "expected_return": 0.05,
            "expected_risk": 0.025,
            "regime_compatibility": "STRONG_BULL,HIGH_VOLATILITY",
        }

    def regime_compatibility(self) -> str:
        return "STRONG_BULL"
