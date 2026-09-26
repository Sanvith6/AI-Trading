"""
Baseline Strategies — Institutional benchmark hurdles.

Every evolved AI strategy must demonstrate incremental risk-adjusted value
over these classical baselines:
- BUY_AND_HOLD
- SMA_TREND
- EMA_TREND
- MOMENTUM
- MEAN_REVERSION
- BREAKOUT
- VOLATILITY_TARGET
- EQUAL_WEIGHT
- RISK_PARITY
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract interface for all baseline strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate decision signal from historical OHLCV data.
        Returns dict:
        {
            'signal': 'BUY' | 'SELL' | 'HOLD',
            'weight': float in [-1.0, 1.0],
            'confidence': float in [0.0, 1.0],
            'reason': str
        }
        """
        pass


class BuyAndHold(BaseStrategy):
    """Classical benchmark: Enter long on first bar and hold."""

    def __init__(self):
        super().__init__(name="BUY_AND_HOLD")

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "signal": "BUY",
            "weight": 1.0,
            "confidence": 1.0,
            "reason": "Always long buy and hold",
        }


class SmaTrend(BaseStrategy):
    """Dual Moving Average Trend (Fast SMA vs Slow SMA)."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(name="SMA_TREND")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.slow_period:
            return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Insufficient bars"}

        fast_sma = df["close"].rolling(self.fast_period).mean().iloc[-1]
        slow_sma = df["close"].rolling(self.slow_period).mean().iloc[-1]

        if fast_sma > slow_sma:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": 0.7, "reason": f"SMA{self.fast_period} > SMA{self.slow_period}"}
        elif fast_sma < slow_sma:
            return {"strategy": self.name, "signal": "SELL", "weight": -1.0, "confidence": 0.7, "reason": f"SMA{self.fast_period} < SMA{self.slow_period}"}
        return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Neutral SMA alignment"}


class EmaTrend(BaseStrategy):
    """Fast EMA vs Slow EMA (9 / 21)."""

    def __init__(self, fast_span: int = 9, slow_span: int = 21):
        super().__init__(name="EMA_TREND")
        self.fast_span = fast_span
        self.slow_span = slow_span

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.slow_span:
            return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Insufficient bars"}

        fast_ema = df["close"].ewm(span=self.fast_span, adjust=False).mean().iloc[-1]
        slow_ema = df["close"].ewm(span=self.slow_span, adjust=False).mean().iloc[-1]

        if fast_ema > slow_ema:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": 0.75, "reason": f"EMA{self.fast_span} > EMA{self.slow_span}"}
        elif fast_ema < slow_ema:
            return {"strategy": self.name, "signal": "SELL", "weight": -1.0, "confidence": 0.75, "reason": f"EMA{self.fast_span} < EMA{self.slow_span}"}
        return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Neutral EMA"}


class MomentumBaseline(BaseStrategy):
    """Classical rate of change (ROC-20) momentum."""

    def __init__(self, lookback: int = 20, threshold_pct: float = 2.0):
        super().__init__(name="MOMENTUM")
        self.lookback = lookback
        self.threshold = threshold_pct

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.lookback + 1:
            return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Insufficient bars"}

        roc = (df["close"].iloc[-1] / df["close"].iloc[-1 - self.lookback] - 1.0) * 100.0
        if roc > self.threshold:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": min(0.9, 0.5 + roc / 20.0), "reason": f"ROC {roc:.1f}% > {self.threshold}%"}
        elif roc < -self.threshold:
            return {"strategy": self.name, "signal": "SELL", "weight": -1.0, "confidence": min(0.9, 0.5 + abs(roc) / 20.0), "reason": f"ROC {roc:.1f}% < -{self.threshold}%"}
        return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Momentum within neutral threshold"}


class MeanReversionBaseline(BaseStrategy):
    """Bollinger Band deviation mean reversion."""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(name="MEAN_REVERSION")
        self.period = period
        self.num_std = num_std

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.period:
            return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Insufficient bars"}

        c = df["close"]
        sma = c.rolling(self.period).mean().iloc[-1]
        std = c.rolling(self.period).std().iloc[-1]
        curr = c.iloc[-1]

        lower_band = sma - self.num_std * std
        upper_band = sma + self.num_std * std

        if curr < lower_band:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": 0.8, "reason": f"Price ₹{curr:.2f} < Lower Bollinger ₹{lower_band:.2f}"}
        elif curr > upper_band:
            return {"strategy": self.name, "signal": "SELL", "weight": -1.0, "confidence": 0.8, "reason": f"Price ₹{curr:.2f} > Upper Bollinger ₹{upper_band:.2f}"}
        return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Price inside Bollinger envelope"}


class BreakoutBaseline(BaseStrategy):
    """Donchian channel 20-bar high/low breakout."""

    def __init__(self, period: int = 20):
        super().__init__(name="BREAKOUT")
        self.period = period

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.period + 1:
            return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Insufficient bars"}

        window = df.iloc[-1 - self.period : -1]
        highest = window["high"].max()
        lowest = window["low"].min()
        curr_close = df["close"].iloc[-1]

        if curr_close > highest:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": 0.85, "reason": f"New {self.period}-bar high breakout: ₹{curr_close:.2f} > ₹{highest:.2f}"}
        elif curr_close < lowest:
            return {"strategy": self.name, "signal": "SELL", "weight": -1.0, "confidence": 0.85, "reason": f"New {self.period}-bar low breakdown: ₹{curr_close:.2f} < ₹{lowest:.2f}"}
        return {"strategy": self.name, "signal": "HOLD", "weight": 0.0, "confidence": 0.5, "reason": "Inside Donchian channel"}


class VolatilityTargetBaseline(BaseStrategy):
    """Volatility targeting: scales position to maintain target annualized volatility."""

    def __init__(self, target_vol_annual: float = 0.15, lookback: int = 20):
        super().__init__(name="VOLATILITY_TARGET")
        self.target_vol = target_vol_annual
        self.lookback = lookback

    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.lookback + 1:
            return {"strategy": self.name, "signal": "BUY", "weight": 1.0, "confidence": 0.5, "reason": "Default allocation"}

        daily_returns = df["close"].pct_change().dropna()
        realized_vol = float(daily_returns.tail(self.lookback).std() * np.sqrt(252))

        if realized_vol > 0:
            target_weight = min(1.5, self.target_vol / realized_vol)
        else:
            target_weight = 1.0

        return {
            "strategy": self.name,
            "signal": "BUY" if target_weight > 0.1 else "HOLD",
            "weight": round(target_weight, 3),
            "confidence": 0.75,
            "reason": f"Realized vol {realized_vol*100:.1f}%, Target {self.target_vol*100:.1f}%, Scaled Weight {target_weight:.2f}",
        }


def get_all_baselines() -> List[BaseStrategy]:
    """Factory helper returning instance of all benchmark strategies."""
    return [
        BuyAndHold(),
        SmaTrend(),
        EmaTrend(),
        MomentumBaseline(),
        MeanReversionBaseline(),
        BreakoutBaseline(),
        VolatilityTargetBaseline(),
    ]
