"""
Cross-Sectional Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- rank: percentile rank within universe
- percentile: alternative percentile measure
- relative_strength: relative strength vs benchmark
- sector_adjusted_momentum: momentum adjusted for sector
- market_adjusted_momentum: momentum adjusted for market
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def rank(series: pd.Series, window: int = 20) -> pd.Series:
    """Percentile rank of series within rolling window.
    
    Returns value in [0, 1] where 1 = highest in window.
    Time-safe: uses only past data within window.
    """
    return series.rolling(window=window).rank(pct=True)


def relative_strength(current: pd.Series, benchmark: pd.Series, window: int = 20) -> pd.Series:
    """Relative strength: current performance vs benchmark over window.
    
    (current - benchmark) / benchmark gives the out/underperformance.
    """
    return (current / benchmark - 1).rolling(window=window).mean()


def sector_adjusted_momentum(momentum: pd.Series, sector_returns: pd.Series, window: int = 20) -> pd.Series:
    """Momentum adjusted for sector effects.
    
    Subtract sector-average momentum to get residual alpha.
    """
    sector_avg = sector_returns.rolling(window=window).mean()
    return momentum - sector_avg


def market_adjusted_momentum(momentum: pd.Series, market_return: pd.Series, window: int = 20) -> pd.Series:
    """Momentum adjusted for market effects.
    
    Subtract market-average return to get residual alpha.
    """
    market_avg = market_return.rolling(window=window).mean()
    return momentum - market_avg