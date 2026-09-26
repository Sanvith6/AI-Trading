"""
Volume Features — Time-Safe

All features in this module are time-safe: they use only information available
at the decision bar close. No future data leakage.

Features:
- relative_volume: volume relative to average
- volume_acceleration: change in volume trend
- volume_surprise: deviation from expected volume
- price_volume_interaction: price change * volume
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from trading.features.price_features import returns


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to average volume over window.
    
    Time-safe: average uses only past data.
    Values > 1.5 indicate above-average volume.
    """
    vol_avg = volume.rolling(window=window).mean()
    return volume / vol_avg.replace(0, np.nan)


def volume_acceleration(volume: pd.Series, window: int = 20) -> pd.Series:
    """Acceleration of volume trend — change in relative volume.
    
    Time-safe: uses only past data.
    Positive = volume increasing, Negative = volume decreasing.
    """
    rel_vol = relative_volume(volume, window)
    return rel_vol - rel_vol.shift(1)


def volume_surprise(volume: pd.Series, price_change: pd.Series, window: int = 20) -> pd.Series:
    """Volume surprise: deviation of actual volume from expected.
    
    Expected volume could be modeled from historical patterns.
    Here: compare current relative volume to recent average of relative volume.
    """
    rel_vol = relative_volume(volume, window)
    expected_rel_vol = rel_vol.rolling(window=window//2).mean()
    return rel_vol - expected_rel_vol


def price_volume_interaction(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Price-change-weighted volume interaction.
    
    Measures whether volume confirms price move.
    Positive = high volume with price up (confirmation)
    Negative = high volume with price down (selling pressure)
    """
    ret = returns(close)
    return ret * volume