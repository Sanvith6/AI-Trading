"""
Statistical Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- beta: beta relative to market
- rolling_correlation: correlation with market
- residual_return: residual from market model
- correlation_regime: correlation regime
- volatility_ratio: ratio of short-term to long-term volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def beta(stock_returns: pd.Series, market_returns: pd.Series, window: int = 30) -> pd.Series:
    """Beta: sensitivity of stock returns to market returns.
    
    Beta = Cov(stock, market) / Var(market), estimated over window.
    Time-safe: uses only past data.
    """
    cov = stock_returns.rolling(window=window).cov(market_returns)
    var = market_returns.rolling(window=window).var()
    return cov / var.replace(0, np.nan)


def rolling_correlation(stock_returns: pd.Series, market_returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling correlation between stock and market."""
    return stock_returns.rolling(window=window).corr(market_returns)


def residual_return(stock_returns: pd.Series, market_returns: pd.Series, beta_est: pd.Series = None, window: int = 30) -> pd.Series:
    """Residual from market model: stock - beta * market.
    
    The idiosyncratic component after removing market exposure.
    Time-safe: beta estimated from past data only.
    """
    if beta_est is None:
        beta_est = beta(stock_returns, market_returns, window)
    # Excess return = stock_return - beta * market_return
    # Need to align indices properly
    b = beta_est.reindex(stock_returns.index)
    mr = market_returns.reindex(stock_returns.index)
    return stock_returns - b * mr


def correlation_regime(correlation: pd.Series, high_threshold: float = 0.7, low_threshold: float = 0.3) -> str:
    """Identify correlation regime.
    
    Returns: 'HIGH' if correlation > high_threshold,
              'LOW' if correlation < low_threshold,
              'MODERATE' otherwise.
    """
    last_corr = float(correlation.iloc[-1]) if len(correlation) > 0 else 0
    if last_corr >= high_threshold:
        return "HIGH"
    elif last_corr <= low_threshold:
        return "LOW"
    else:
        return "MODERATE"


def volatility_ratio(short_window: int = 10, long_window: int = 30) -> callable:
    """Return a function that calculates short/long volatility ratio.
    
    Use: vol_ratio = volatility_ratio(short_window, long_window)(close)
    """
    def compute(close: pd.Series) -> pd.Series:
        short_vol = close.pct_change().rolling(short_window).std()
        long_vol = close.pct_change().rolling(long_window).std()
        return short_vol / long_vol.replace(0, np.nan)
    return compute