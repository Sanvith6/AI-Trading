"""
Momentum Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- roc: Rate of Change
- rsi: Relative Strength Index
- multi_horizon_momentum: momentum at multiple horizons
- relative_cross_sectional_rank: rank within universe
- momentum_cross_sectional: cross-sectional momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from trading.features.price_features import returns, log_returns


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change: (close_t / close_{t-period}) - 1
    
    Time-safe: uses only current and past close.
    """
    return (close / close.shift(period)) - 1


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.
    
    Time-safe: uses only current and prior price changes.
    Classic Wilder smoothing via rolling mean of gains/losses.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def multi_horizon_momentum(close: pd.Series, horizons: list = [5, 10, 20]) -> Dict[str, pd.Series]:
    """Momentum at multiple horizons.
    
    Returns dict with ROC for each horizon.
    """
    return {f"roc_{h}": roc(close, h) for h in horizons}


def relative_cross_sectional_rank(close: pd.Series, universe: pd.DataFrame = None, window: int = 20) -> pd.Series:
    """Relative strength: rank of current close within rolling window.
    
    Time-safe: rank uses only data within window, no future data.
    Returns 0-100 percentile.
    """
    # rank returns percentage rank (0-100)
    return close.rolling(window=window).rank(pct=True).iloc[:, 0]


def momentum_cross_sectional(close: pd.Series, window: int = 20, universes: list = None) -> pd.Series:
    """Cross-sectional momentum: how does this rank vs universe.
    
    In a full implementation, this would compare against a universe of symbols.
    Here we compute a simplified version using rolling rank.
    """
    return close.rolling(window=window).rank(pct=True).iloc[:, 0]