"""
Statistical Alpha Family — Explores residual return reversion, beta deviations, and cointegration proxies.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class StatisticalAlpha(BaseAlpha):
    """Generates signals based on statistical residual returns and beta-adjusted moves."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "residual_z_threshold": 1.7,
        }

    def required_features(self) -> List[str]:
        return ["zscore_20", "realized_vol_20", "ret_1", "ret_5"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        z = features.get("zscore_20", 0.0)
        r5 = features.get("ret_5", 0.0)
        vol = features.get("realized_vol_20", 20.0)

        zt = self._parameters.get("residual_z_threshold", 1.7)

        # Statistical exhaustion / snapback
        if z < -zt and r5 < -0.05:
            return 0.8
        elif z > zt and r5 > 0.05:
            return -0.8

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        z = abs(features.get("zscore_20", 0.0))
        return min(0.90, max(0.30, 0.40 + (z / 3.0) * 0.5))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        z = features.get("zscore_20", 0.0)
        return {
            "reason_codes": [f"Statistical z-deviation: {z:.2f}"],
            "expected_horizon": "short",
            "expected_return": 0.035,
            "expected_risk": 0.02,
            "regime_compatibility": "SIDEWAYS,LOW_VOLATILITY",
        }

    def regime_compatibility(self) -> str:
        return "SIDEWAYS"
