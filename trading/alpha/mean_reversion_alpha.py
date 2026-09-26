"""
Mean Reversion Alpha Family — Exploits statistical overextension via z-scores, RSI, and Bollinger bands.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class MeanReversionAlpha(BaseAlpha):
    """Generates signals based on distance from moving averages, z-score, and RSI extremities."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "zscore_threshold": 1.8,
            "rsi_oversold": 32.0,
            "rsi_overbought": 68.0,
            "bb_dev_threshold": 80.0,
        }

    def required_features(self) -> List[str]:
        return ["zscore_20", "rsi_14", "bb_deviation", "dist_from_sma_20"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        z = features.get("zscore_20", 0.0)
        rsi = features.get("rsi_14", 50.0)
        bb = features.get("bb_deviation", 0.0)

        zt = self._parameters.get("zscore_threshold", 1.8)
        rsi_low = self._parameters.get("rsi_oversold", 32.0)
        rsi_high = self._parameters.get("rsi_overbought", 68.0)

        # Strongly oversold -> Buy signal
        if z < -zt and rsi < rsi_low:
            return 1.0
        elif z < -1.2 or rsi < 35.0:
            return 0.5

        # Strongly overbought -> Sell signal
        if z > zt and rsi > rsi_high:
            return -1.0
        elif z > 1.2 or rsi > 65.0:
            return -0.5

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        z = abs(features.get("zscore_20", 0.0))
        rsi = features.get("rsi_14", 50.0)
        dist = abs(rsi - 50.0) / 50.0
        return min(0.95, max(0.20, (z / 3.0) * 0.5 + dist * 0.5))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        z = features.get("zscore_20", 0.0)
        rsi = features.get("rsi_14", 50.0)
        reasons = []
        if z < -1.5:
            reasons.append(f"Z-score extreme low ({z:.2f})")
        elif z > 1.5:
            reasons.append(f"Z-score extreme high ({z:.2f})")
        if rsi < 35:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:
            reasons.append(f"RSI overbought ({rsi:.1f})")

        return {
            "reason_codes": reasons or ["Within neutral mean-reversion boundary"],
            "expected_horizon": "short_to_medium",
            "expected_return": 0.03,
            "expected_risk": 0.02,
            "regime_compatibility": "SIDEWAYS,LOW_VOLATILITY",
        }

    def regime_compatibility(self) -> str:
        return "SIDEWAYS"
