"""
Regime Engine — Multi-Regime Detection & Probability Modeling with Hysteresis Smoothing.

Detects:
- STRONG_BULL, WEAK_BULL, SIDEWAYS, WEAK_BEAR, STRONG_BEAR
- HIGH_VOLATILITY, LOW_VOLATILITY
- RISK_ON, RISK_OFF

Includes Hysteresis Confirmation Window (smoothing_window=3) to eliminate
false whipsaw regime switching and stabilize strategy allocations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


class RegimeType:
    STRONG_BULL     = "STRONG_BULL"
    WEAK_BULL       = "WEAK_BULL"
    SIDEWAYS        = "SIDEWAYS"
    WEAK_BEAR       = "WEAK_BEAR"
    STRONG_BEAR     = "STRONG_BEAR"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY  = "LOW_VOLATILITY"
    RISK_ON         = "RISK_ON"
    RISK_OFF        = "RISK_OFF"


@dataclass
class RegimeState:
    regime: str
    confidence: float
    duration: int
    features: Dict[str, float]
    transition_probabilities: Dict[str, float]
    risk_mode: str  # "RISK_ON" | "RISK_OFF" | "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegimeDetector:
    """Detects market regimes from time-series features without future leakage."""

    def __init__(self, lookback: int = 50, smoothing_window: int = 3):
        self.lookback = lookback
        self.smoothing_window = smoothing_window
        self._history: List[str] = []
        self._raw_history: List[str] = []

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Analyze an OHLCV DataFrame (minimum 20 bars) and detect the active regime.
        Applies hysteresis confirmation to prevent 1-bar noise whipsaw switching.
        """
        if df is None or len(df) < 20:
            return RegimeState(
                regime=RegimeType.SIDEWAYS,
                confidence=0.5,
                duration=1,
                features={},
                transition_probabilities={RegimeType.SIDEWAYS: 1.0},
                risk_mode="NEUTRAL",
            )

        c = df["close"]
        h = df["high"]
        l = df["low"]

        # Compute core indicators
        sma_20 = c.rolling(20).mean().iloc[-1]
        sma_50 = c.rolling(50).mean().iloc[-1] if len(c) >= 50 else sma_20
        curr_price = float(c.iloc[-1])

        # Trend slope
        slope_20 = float((c.iloc[-1] - c.iloc[-20]) / c.iloc[-20] * 100) if len(c) >= 20 else 0.0

        # ADX estimate
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        atr_14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(h.iloc[-1] - l.iloc[-1])
        atr_pct = (atr_14 / curr_price * 100) if curr_price > 0 else 0.0

        # Realized vol annualized
        ret_1 = c.pct_change(1).dropna()
        realized_vol = float(ret_1.tail(20).std() * np.sqrt(252) * 100) if len(ret_1) >= 20 else 15.0

        features = {
            "close": round(curr_price, 2),
            "slope_20d_pct": round(slope_20, 2),
            "atr_pct": round(atr_pct, 2),
            "realized_vol_annual_pct": round(realized_vol, 2),
            "price_above_sma20": bool(curr_price > sma_20),
            "price_above_sma50": bool(curr_price > sma_50),
        }

        # Rule-based raw regime determination
        raw_regime = RegimeType.SIDEWAYS
        confidence = 0.60
        risk_mode = "NEUTRAL"

        # Volatility classification
        if atr_pct > 3.5 or realized_vol > 32.0:
            raw_regime = RegimeType.HIGH_VOLATILITY
            confidence = 0.80
            risk_mode = "RISK_OFF"
        elif atr_pct < 1.2 and realized_vol < 12.0:
            raw_regime = RegimeType.LOW_VOLATILITY
            confidence = 0.70
            risk_mode = "RISK_ON"
        # Trend classification
        elif curr_price > sma_20 and sma_20 > sma_50 and slope_20 > 5.0:
            raw_regime = RegimeType.STRONG_BULL
            confidence = 0.85
            risk_mode = "RISK_ON"
        elif curr_price > sma_20 and slope_20 > 1.5:
            raw_regime = RegimeType.WEAK_BULL
            confidence = 0.70
            risk_mode = "RISK_ON"
        elif curr_price < sma_20 and sma_20 < sma_50 and slope_20 < -5.0:
            raw_regime = RegimeType.STRONG_BEAR
            confidence = 0.85
            risk_mode = "RISK_OFF"
        elif curr_price < sma_20 and slope_20 < -1.5:
            raw_regime = RegimeType.WEAK_BEAR
            confidence = 0.70
            risk_mode = "RISK_OFF"
        else:
            raw_regime = RegimeType.SIDEWAYS
            confidence = 0.65
            risk_mode = "NEUTRAL"

        self._raw_history.append(raw_regime)

        # Hysteresis smoothing: require confirmation before switching away from active regime
        regime = raw_regime
        if self.smoothing_window > 1 and len(self._history) > 0:
            active_regime = self._history[-1]
            if raw_regime != active_regime:
                recent_raw = self._raw_history[-self.smoothing_window:]
                if len(recent_raw) < self.smoothing_window or not all(r == raw_regime for r in recent_raw):
                    regime = active_regime

        self._history.append(regime)

        # Calculate duration of current regime
        duration = 0
        for past in reversed(self._history):
            if past == regime:
                duration += 1
            else:
                break

        # Estimated transition probabilities based on regime persistence
        transitions = self._compute_transitions(regime, duration)

        return RegimeState(
            regime=regime,
            confidence=round(confidence, 2),
            duration=duration,
            features=features,
            transition_probabilities=transitions,
            risk_mode=risk_mode,
        )

    def _compute_transitions(self, current_regime: str, duration: int) -> Dict[str, float]:
        """Estimate next-bar regime transitions with decaying persistence."""
        persistence = max(0.60, 0.85 - (duration * 0.005))
        leak = (1.0 - persistence) / 3.0

        trans = {current_regime: round(persistence, 3)}
        if "BULL" in current_regime:
            trans[RegimeType.SIDEWAYS] = round(leak * 1.5, 3)
            trans[RegimeType.WEAK_BEAR] = round(leak * 0.8, 3)
            trans[RegimeType.HIGH_VOLATILITY] = round(leak * 0.7, 3)
        elif "BEAR" in current_regime:
            trans[RegimeType.SIDEWAYS] = round(leak * 1.5, 3)
            trans[RegimeType.HIGH_VOLATILITY] = round(leak * 1.0, 3)
            trans[RegimeType.WEAK_BULL] = round(leak * 0.5, 3)
        else:
            trans[RegimeType.WEAK_BULL] = round(leak, 3)
            trans[RegimeType.WEAK_BEAR] = round(leak, 3)
            trans[RegimeType.HIGH_VOLATILITY] = round(leak, 3)

        tot = sum(trans.values())
        return {k: round(v / tot, 3) for k, v in trans.items()}
