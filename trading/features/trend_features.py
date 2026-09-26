"""
Trend Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- ema: Exponential Moving Average
- sma: Simple Moving Average
- adx: Average Directional Index (trend strength)
- trend_slope: Polynomial fit slope of price series
- trend_persistence: Consecutive bars above/below MA
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average.
    
    Time-safe: uses only past data via ewm adjust=False.
    """
    return close.ewm(span=span, adjust=False).mean()


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average.
    
    Time-safe: rolling window uses only past data.
    """
    return close.rolling(window=window).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength gauge.
    
    Time-safe: DI calculations use only current and prior bars.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).rolling(period).mean()
    
    plus_di = 100 * plus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.rolling(period).mean() / tr.replace(0, float("nan"))
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.rolling(period).mean().fillna(0)


def trend_slope(close: pd.Series, degree: int = 1, window: int = 20) -> pd.Series:
    """Linear regression slope of price over a rolling window.
    
    Returns slope per bar — positive = uptrend, negative = downtrend.
    Time-safe: fits using only past data within window.
    """
    slopes = []
    for i in range(window, len(close) + 1):
        y = close.iloc[i - window:i].values
        x = np.arange(len(y))
        # Polyfit: degree 1 gives slope
        coeff = np.polyfit(x, y, degree)
        slopes.append(coeff[0])
    # Return as Series aligned with input index
    result = pd.Series(np.nan, index=close.index)
    result.iloc[window-1:] = slopes
    return result


def trend_persistence(close: pd.Series, ma_window: int = 20, lookback: int = 5) -> Dict[str, pd.Series]:
    """Trend persistence features.
    
    Returns dict with:
    - above_ma: whether price is above MA
    - consecutive_above: bars consecutive above MA
    - consecutive_below: bars consecutive below MA
    """
    ma = sma(close, ma_window)
    above_ma = close > ma
    
    # Consecutive above/below using groupby
    groups = (~above_ma).cumsum()
    consecutive = above_ma.groupby(groups).cumcount() * (~above_ma).astype(int)
    # Actually let's compute differently
    
    # Simple: shift and compare
    prev_above = above_ma.shift(1)
    consecutive_above = ((above_ma) & (~prev_above)).cumsum()
    consecutive_below = ((~above_ma) & (prev_above)).cumsum()
    
    return {
        "above_ma": above_ma,
        "consecutive_above": consecutive_above,
        "consecutive_below": consecutive_below,
    }