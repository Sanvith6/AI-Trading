"""
Mean Reversion Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- z_score: z-score from rolling mean/std
- bb_deviation: Bollinger Band deviation
- distance_from_ma: distance from moving average
- short_term_reversal: short-term reversal signal
- residual_reversion: residual from regression
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from trading.features.trend_features import sma


def z_score(close: pd.Series, window: int = 20) -> pd.Series:
    """Z-score: (price - rolling_mean) / rolling_std.
    
    Time-safe: uses only past data within window.
    Extremes (z > 2 or z < -2) may indicate mean reversion opportunities.
    """
    mean = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return (close - mean) / std.replace(0, np.nan)


def bb_deviation(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band deviation from middle band.
    
    Time-safe: SMA and std use only past data.
    """
    mid = sma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    # Deviation from mid as percentage of band width
    band_width = upper - lower
    return ((close - mid) / band_width).replace([np.inf, -np.inf], np.nan)


def distance_from_ma(close: pd.Series, ma_window: int = 20) -> pd.Series:
    """Distance from moving average as percentage."""
    ma = sma(close, ma_window)
    return (close - ma) / ma.replace(0, np.nan)


def short_term_reversal(close: pd.Series, window: int = 5) -> pd.Series:
    """Short-term reversal: price change over short window.
    
    Mean reversion logic: recent winners may reverse, recent losers may bounce.
    """
    return returns(close, window)


def residual_reversion(close: pd.Series, regression_window: int = 30) -> pd.Series:
    """Residual from rolling linear regression — mean reversion signal.
    
    Fits a line to the last N closes; the residual indicates how far
    price is from its predicted path.
    """
    residuals = []
    for i in range(regression_window, len(close) + 1):
        y = close.iloc[i - regression_window:i].values
        x = np.arange(regression_window)
        coeffs = np.polyfit(x, y, 1)  # slope + intercept
        predicted = coeffs[1] + coeffs[0] * (regression_window - 1)
        residual = close.iloc[i - 1] - predicted
        residuals.append(residual)
    result = pd.Series(np.nan, index=close.index)
    result.iloc[regression_window-1:] = residuals
    return result