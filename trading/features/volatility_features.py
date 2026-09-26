"""
Market Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- index_return: return of market index
- index_volatility: volatility of market index
- sector_momentum: sector-level momentum
- market_regime: identified market regime
- breadth: market breadth indicator
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def market_return(index_prices: pd.Series) -> pd.Series:
    """Return of market index."""
    return index_prices.pct_change()


def market_volatility(index_prices: pd.Series, window: int = 20) -> pd.Series:
    """Volatility of market index."""
    return np.log(index_prices / index_prices.shift(1)).rolling(window=window).std() * np.sqrt(252)


def sector_momentum(sector_returns: pd.Series, window: int = 20) -> pd.Series:
    """Sector momentum: rolling return of sector index."""
    return sector_returns.rolling(window=window).mean()


def market_breadth(advance: pd.Series, decline: pd.Series) -> pd.Series:
    """Market breadth: advancing vs declining issues.
    
    Advance / (Advance + Decline) gives a ratio in [0, 1].
    """
    total = advance + decline
    return advance / total.replace(0, np.nan)


def market_regime_adx(high: pd.Series, low: pd.Series, close: pd.Series, 
                      adx_period: int = 14, threshold: int = 20) -> str:
    """Identify market regime based on ADX.
    
    Returns: 'TRENDING' if ADX > threshold, 'CHOPPY' if ADX < 15, else 'WEAK_TREND'
    """
    from trading.features.trend_features import adx as _adx
    adx_val = _adx(high, low, close, adx_period)
    last_adx = float(adx_val.iloc[-1])
    
    if last_adx >= threshold:
        return "TRENDING"
    elif last_adx < 15:
        return "CHOPPY"
    else:
        return "WEAK_TREND"