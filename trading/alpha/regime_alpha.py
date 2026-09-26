"""
Regime Alpha Family — Adapts signal generation dynamically to active market regimes.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class RegimeAlpha(BaseAlpha):
    """Adapts directional bias and risk tolerance to detected market regime."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "bull_bias": 0.8,
            "bear_bias": -0.8,
            "chop_bias": 0.0,
        }

    def required_features(self) -> List[str]:
        return ["trend_slope_20", "adx_14", "realized_vol_20"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        slope = features.get("trend_slope_20", 0.0)
        adx = features.get("adx_14", 20.0)
        vol = features.get("realized_vol_20", 20.0)

        # Strong trend regime
        if adx > 25.0 and slope > 5.0 and vol < 30.0:
            return self._parameters.get("bull_bias", 0.8)
        elif adx > 25.0 and slope < -5.0:
            return self._parameters.get("bear_bias", -0.8)

        # Choppy / sideways regime -> mean reversion posture (small bias)
        if adx < 18.0:
            return self._parameters.get("chop_bias", 0.0)

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        adx = features.get("adx_14", 20.0)
        return min(0.95, max(0.40, adx / 50.0))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        adx = features.get("adx_14", 20.0)
        slope = features.get("trend_slope_20", 0.0)
        return {
            "reason_codes": [f"ADX Trend Strength: {adx:.1f}", f"20d Slope: {slope:.2f}%"],
            "expected_horizon": "medium",
            "expected_return": 0.05,
            "expected_risk": 0.02,
            "regime_compatibility": "any",
        }

    def regime_compatibility(self) -> str:
        return "any"
