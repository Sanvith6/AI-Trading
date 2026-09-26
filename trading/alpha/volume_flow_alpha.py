"""
Volume Flow Alpha Family — Measures institutional volume spikes and price-volume pressure.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class VolumeFlowAlpha(BaseAlpha):
    """Signals based on relative volume spikes and price-volume interactions."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "spike_threshold": 1.75,
            "acceleration_threshold": 1.30,
        }

    def required_features(self) -> List[str]:
        return ["relative_volume", "volume_acceleration", "price_vol_interaction", "ret_1"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        rvol = features.get("relative_volume", 1.0)
        accel = features.get("volume_acceleration", 1.0)
        pvi = features.get("price_vol_interaction", 0.0)
        ret1 = features.get("ret_1", 0.0)

        spike = self._parameters.get("spike_threshold", 1.75)

        # Huge volume on positive return -> institutional accumulation
        if rvol > spike and ret1 > 0.01:
            return 1.0
        elif rvol > 1.3 and ret1 > 0.005:
            return 0.6

        # Huge volume on negative return -> institutional liquidation
        if rvol > spike and ret1 < -0.01:
            return -1.0
        elif rvol > 1.3 and ret1 < -0.005:
            return -0.6

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        rvol = features.get("relative_volume", 1.0)
        return min(0.95, max(0.40, 0.50 + (rvol - 1.0) * 0.25))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        rvol = features.get("relative_volume", 1.0)
        pvi = features.get("price_vol_interaction", 0.0)
        return {
            "reason_codes": [f"Relative volume: {rvol:.2f}x", f"Price-Volume Interaction: {pvi:.4f}"],
            "expected_horizon": "short_to_medium",
            "expected_return": 0.045,
            "expected_risk": 0.02,
            "regime_compatibility": "any",
        }

    def regime_compatibility(self) -> str:
        return "any"
