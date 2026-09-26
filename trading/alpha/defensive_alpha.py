"""
Defensive Alpha Family — Prioritizes capital preservation, drawdown defense, and risk-off hedges.
"""

from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class DefensiveAlpha(BaseAlpha):
    """Issues protective SELL / HOLD signals when drawdown expands or volatility spikes."""

    def default_parameters(self) -> Dict[str, Any]:
        return {
            "max_drawdown_tolerance_pct": -5.0,
            "max_realized_vol_pct": 35.0,
        }

    def required_features(self) -> List[str]:
        return ["drawdown", "realized_vol_20", "atr_pct"]

    def generate_signal(self, features: Dict[str, Any]) -> float:
        dd = features.get("drawdown", 0.0)
        vol = features.get("realized_vol_20", 15.0)

        max_dd = self._parameters.get("max_drawdown_tolerance_pct", -5.0)
        max_vol = self._parameters.get("max_realized_vol_pct", 35.0)

        # Severe drawdown or explosive volatility -> De-risk immediately
        if dd < max_dd or vol > max_vol:
            return -1.0
        elif dd < (max_dd * 0.6) or vol > (max_vol * 0.8):
            return -0.5

        # Safe conditions -> Neutral / Modest Long
        if dd > -2.0 and vol < 20.0:
            return 0.5

        return 0.0

    def score(self, features: Dict[str, Any]) -> float:
        vol = features.get("realized_vol_20", 15.0)
        return min(0.95, max(0.40, 0.50 + (vol / 50.0) * 0.4))

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        dd = features.get("drawdown", 0.0)
        vol = features.get("realized_vol_20", 15.0)
        return {
            "reason_codes": [f"Current drawdown: {dd:.2f}%", f"Realized vol: {vol:.1f}%"],
            "expected_horizon": "short_to_medium",
            "expected_return": 0.01,
            "expected_risk": 0.01,
            "regime_compatibility": "BEAR,HIGH_VOLATILITY,RISK_OFF",
        }

    def regime_compatibility(self) -> str:
        return "BEAR"
