"""
Trend Alpha Family

Alpha that captures trend-following opportunities.
Uses EMA crossovers, ADX trend strength, and price position.

Signal: +1 = bullish trend, -1 = bearish trend, 0 = no strong trend
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class TrendAlpha(BaseAlpha):
    """Trend-following alpha family.
    
    Captures trends using moving average crossovers and ADX trend strength.
    Signal range: -1 (strong bearish trend) to +1 (strong bullish trend).
    """
    
    def default_parameters(self) -> Dict[str, Any]:
        return {
            "ema_short": 9,
            "ema_long": 21,
            "adx_period": 14,
            "adx_threshold": 20,
            "price_position_threshold": 0.05,  # 5% from moving average
            "use_adx_filter": True,
        }
    
    def required_features(self) -> List[str]:
        return [
            "ema_9", "ema_21",
            "adx", "close",
        ]
    
    def generate_signal(self, features: Dict[str, Any]) -> float:
        """Generate trend signal in [-1, +1].
        
        Logic:
        - EMA9 > EMA21 + filter: bullish (positive signal)
        - EMA9 < EMA21 - filter: bearish (negative signal)
        - ADX above threshold confirms trend strength
        - Price position relative to EMAs adjusts signal intensity
        """
        ema_short = features.get("ema_9", 0)
        ema_long = features.get("ema_21", 0)
        adx = features.get("adx", 0)
        close = features.get("close", 0)
        
        # Default: HOLD
        signal = 0.0
        
        # Check if we have enough data
        if ema_short is None or ema_long is None or adx is None:
            return signal
        
        # EMA crossover direction
        ema_bullish = ema_short > ema_long
        ema_bearish = ema_short < ema_long
        
        # Trend strength filter
        trend_strong = adx >= self._parameters.get("adx_threshold", 20)
        
        if ema_bullish and trend_strong:
            # Bullish trend - signal strength based on how far EMA9 is above EMA21
            epsilon = 1e-8
            distance_pct = (ema_short - ema_long) / (ema_long + epsilon)
            # Scale from 0 to +1 based on distance
            signal = min(1.0, max(0.0, distance_pct * 20))  # 20x scaling
        elif ema_bearish and trend_strong:
            # Bearish trend - signal strength based on how far EMA9 is below EMA21
            epsilon = 1e-8
            distance_pct = (ema_short - ema_long) / (ema_long + epsilon)
            signal = max(-1.0, min(0.0, distance_pct * 20))
        elif ema_bullish and not trend_strong:
            # Weak bullish - partial signal
            signal = 0.3
        elif ema_bearish and not trend_strong:
            # Weak bearish - partial signal
            signal = -0.3
        
        return round(signal, 4)
    
    def score(self, features: Dict[str, Any]) -> float:
        """Return confidence score in [0, 1].
        
        Higher ADX + stronger EMA separation = higher confidence.
        """
        adx = features.get("adx", 0)
        ema_short = features.get("ema_9", 0)
        ema_long = features.get("ema_21", 0)
        
        if adx is None or ema_short is None or ema_long is None:
            return 0.0
        
        # Base confidence from ADX (scaled: 20->0.5, 40->0.8, 60->1.0)
        adx_conf = min(1.0, adx / 60.0)
        
        # Separation confidence
        epsilon = 1e-8
        if ema_long != 0:
            separation = abs(ema_short - ema_long) / (ema_long + epsilon)
            sep_conf = min(1.0, separation * 50)
        else:
            sep_conf = 0.0
        
        # Combined confidence
        combined = adx_conf * 0.6 + sep_conf * 0.4
        return round(min(1.0, combined), 4)
    
    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Provide explanation for the trend signal."""
        adx = features.get("adx", 0)
        ema_short = features.get("ema_9", 0)
        ema_long = features.get("ema_21", 0)
        close = features.get("close", 0)
        
        reason_codes = []
        expected_horizon = "medium"
        expected_return = 0.0
        expected_risk = 0.0
        
        if adx >= 25:
            reason_codes.append("Strong trend (ADX > 25)")
        elif adx >= 20:
            reason_codes.append("Trend (ADX 20-25)")
        else:
            reason_codes.append("Weak or no trend")
        
        if ema_short and ema_long:
            if ema_short > ema_long:
                reason_codes.append("EMA9 above EMA21 (bullish)")
                expected_horizon = "longer-term"
                expected_return = 0.05  # 5% expected
            else:
                reason_codes.append("EMA9 below EMA21 (bearish)")
                expected_horizon = "longer-term"
                expected_return = -0.05
        
        regime = self.regime_compatibility()
        if regime != "any":
            reason_codes.append(f"Optimized for {regime} regime")
        
        return {
            "reason_codes": reason_codes,
            "expected_horizon": expected_horizon,
            "expected_return": round(expected_return, 4),
            "expected_risk": round(expected_risk, 4),
            "regime_compatibility": regime,
        }
    
    def regime_compatibility(self) -> str:
        """Trend alpha works best in trending markets."""
        return "TRENDING"