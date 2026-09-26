"""
Momentum Alpha Family

Alpha that captures medium-term momentum opportunities.
Uses ROC (Rate of Change) and RSI for momentum assessment.

Signal: +1 = strong momentum, -1 = strong reversal, 0 = no strong momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from trading.alpha.base_alpha import BaseAlpha


class MomentumAlpha(BaseAlpha):
    """Momentum alpha family.
    
    Captures medium-term momentum using rate-of-change and RSI.
    Signal range: -1 (strong reversal/momentum exhausted) to +1 (strong momentum).
    """
    
    def default_parameters(self) -> Dict[str, Any]:
        return {
            "roc_period": 10,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "momentum_threshold": 0.03,  # 3% threshold
            "use_rsi_filter": True,
        }
    
    def required_features(self) -> List[str]:
        return [
            "roc_10", "rsi_14",
        ]
    
    def generate_signal(self, features: Dict[str, Any]) -> float:
        """Generate momentum signal in [-1, +1].
        
        Logic:
        - ROC > threshold: bullish momentum (positive signal)
        - ROC < -threshold: bearish momentum (negative signal)
        - RSI filter: extreme RSI reduces confidence or reverses signal
        """
        roc_val = features.get("roc_10", 0)
        rsi_val = features.get("rsi_14", 50)
        
        # Default: HOLD
        signal = 0.0
        
        # ROC-based momentum
        threshold = self._parameters.get("momentum_threshold", 0.03)
        
        if roc_val > threshold:
            # Bullish momentum
            if self._parameters.get("use_rsi_filter", True):
                rsi_ov = self._parameters.get("rsi_overbought", 70)
                if rsi_val > rsi_ov:
                    # Overbought - reduce signal
                    signal = 0.5 * (roc_val / threshold)
                else:
                    signal = min(1.0, roc_val / threshold * 0.5 + 0.5)
            else:
                signal = min(1.0, roc_val / threshold * 0.5 + 0.5)
                
        elif roc_val < -threshold:
            # Bearish momentum
            if self._parameters.get("use_rsi_filter", True):
                rsi_ov = self._parameters.get("rsi_overbought", 70)
                rsi_os = self._parameters.get("rsi_oversold", 30)
                if rsi_val < rsi_os:
                    # Oversold - reduce bearish signal (reversal potential)
                    signal = -0.5 * (abs(roc_val) / threshold)
                else:
                    signal = max(-1.0, -((abs(roc_val) / threshold) * 0.5 + 0.5))
            else:
                signal = max(-1.0, -((abs(roc_val) / threshold) * 0.5 + 0.5))
        
        # borderline: mild momentum
        elif abs(roc_val) > 0.01:
            signal = np.sign(roc_val) * 0.2
        
        return round(signal, 4)
    
    def score(self, features: Dict[str, Any]) -> float:
        """Return confidence score in [0, 1]."""
        roc_val = features.get("roc_10", 0)
        rsi_val = features.get("rsi_14", 50)
        
        if abs(roc_val) < 0.01:
            return 0.0
        
        # Base confidence from ROC magnitude
        threshold = self._parameters.get("momentum_threshold", 0.03)
        roc_conf = min(1.0, abs(roc_val) / threshold)
        
        # RSI adjustment
        rsi_os = self._parameters.get("rsi_oversold", 30)
        rsi_ob = self._parameters.get("rsi_overbought", 70)
        rsi_adj = 1.0
        
        if rsi_val < rsi_os:
            # Oversold adds confidence for reversal, but might indicate
            # momentum is exhausted - moderate
            rsi_adj = 0.7
        elif rsi_val > rsi_ob:
            # Overbought - momentum may be exhausted
            rsi_adj = 0.7
        else:
            # Normal RSI - full confidence
            rsi_adj = 1.0
        
        combined = roc_conf * rsi_adj
        return round(min(1.0, combined), 4)
    
    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Provide explanation for the momentum signal."""
        roc_val = features.get("roc_10", 0)
        rsi_val = features.get("rsi_14", 50)
        
        reason_codes = []
        expected_horizon = "medium"
        expected_return = 0.0
        
        if abs(roc_val) > 0.05:
            reason_codes.append(f"Strong momentum (ROC {roc_val*100:.1f}%)")
        elif abs(roc_val) > 0.02:
            reason_codes.append(f"Moderate momentum (ROC {roc_val*100:.1f}%)")
        else:
            reason_codes.append("Weak momentum")
        
        if rsi_val < 30:
            reason_codes.append("RSI oversold (< 30)")
            expected_horizon = "short-term"
            expected_return = 0.03
        elif rsi_val > 70:
            reason_codes.append("RSI overbought (> 70)")
            expected_horizon = "short-term"
            expected_return = -0.03
        else:
            reason_codes.append("RSI normal range")
        
        return {
            "reason_codes": reason_codes,
            "expected_horizon": expected_horizon,
            "expected_return": round(expected_return, 4),
            "expected_risk": round(0.02, 4),
            "regime_compatibility": "any",
        }
    
    def regime_compatibility(self) -> str:
        """Momentum works in various regimes but prefers TRENDING."""
        return "TRENDING"