"""
Price Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- returns: simple percentage return
- log_returns: log return
- rolling_returns: rolling simple return over N periods
- gap_return: return from previous close to current open
- range: high - low
- atr: Average True Range (volatility measure)
- realized_vol: rolling realized volatility
- drawdown: peak-to-trough drawdown
- distance_from_highs: price distance from rolling high
- distance_from_lows: price distance from rolling low
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Simple percentage return: (close_t / close_{t-period}) - 1"""
    return close.pct_change(period)


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return: log(close_t / close_{t-period})"""
    return np.log(close / close.shift(period))


def rolling_returns(close: pd.Series, window: int = 5) -> pd.Series:
    """Rolling simple return over window periods."""
    return close.pct_change(window)


def gap_return(open_prices: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Gap from previous close to current open: (open - prev_close) / prev_close"""
    return (open_prices - prev_close) / prev_close


def range_features(high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
    """Calculate range-related features."""
    return {
        "high_low_range": high - low,
        "high_low_pct": (high - low) / low,
    }


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — volatility measure.
    
    Time-safe: uses only current and prior bars, no future data.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility from log returns."""
    ret = log_returns(close)
    return ret.rolling(window).std()


def drawdown(close: pd.Series) -> pd.Series:
    """Running drawdown from peak: (current - peak) / peak."""
    peak = close.cummax()
    dd = (close - peak) / peak
    return dd


def distance_from_highs(close: pd.Series, window: int = 20) -> pd.Series:
    """Distance from rolling high as percentage."""
    peak = close.rolling(window).max()
    return (close - peak) / peak


def distance_from_lows(close: pd.Series, window: int = 20) -> pd.Series:
    """Distance from rolling low as percentage."""
    trough = close.rolling(window).min()
    return (close - trough) / trough