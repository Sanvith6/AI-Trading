"""
Universe Manager

Manages the dynamic market universe of eligible instruments.

Features:
- Configurable universe size (50, 100, 250, 500, 1000+)
- Dynamic ranking rather than hard-coded watchlists
- Multiple filter categories
- Universe version tracking for reproducibility
- Support for equities, ETFs, indices, sectors
- Future asset-class expansion support
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
import os

from trading.data_provider import get_ohlcv, get_quote
from trading.features import *
from trading.features.price_features import *
from trading.features.volume_features import *


class UniverseFilter:
    """Filter class for universe selection.
    
    Each filter is a callable that takes instrument data and returns
    a boolean indicating whether the instrument passes.
    """
    
    def __init__(self, 
                 min_price: float = 1.0,
                 max_price: float = 10000.0,
                 min_volume: int = 100000,
                 min_average_traded_value: float = 1000000,
                 max_penny_stocks: bool = True,
                 exclusion_list: Optional[List[str]] = None,
                 max_missing_bars: float = 0.1,  # 10% max missing
                 min_data_completeness: float = 0.95):  # 95% min completeness
        
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_average_traded_value = min_average_traded_value
        self.max_penny_stocks = max_penny_stocks
        self.exclusion_list = set(exclusion_list) if exclusion_list else set()
        self.max_missing_bars = max_missing_bars
        self.min_data_completeness = min_data_completeness
    
    def validate_instrument(self, symbol: str, ohlcv: pd.DataFrame, quote: Dict) -> bool:
        """Validate a single instrument against all filter criteria."""
        if ohlcv.empty or len(ohlcv) < 30:
            return False
        
        # Price filters
        last_close = ohlcv["close"].iloc[-1]
        if last_close < self.min_price or last_close > self.max_price:
            return False
        
        if self.max_penny_stocks and last_close < 5.0:
            return False
        
        # Volume filter
        avg_volume = ohlcv["volume"].rolling(20).mean().iloc[-1]
        if avg_volume < self.min_volume:
            return False
        
        # Average traded value filter
        avg_tv = (last_close * avg_volume)
        if avg_tv < self.min_average_traded_value:
            return False
        
        # Data completeness check
        missing_bars = len(ohlcv[ohlcv["close"].isna()])
        total_bars = len(ohlcv)
        if total_bars > 0 and missing_bars / total_bars > self.max_missing_bars:
            return False
        
        # Stale price detection (last bar too old)
        # Skip for now - would need market open/close times
        
        # Exclusion list check
        if symbol.upper() in self.exclusion_list:
            return False
        
        return True


class UniverseManager:
    """Manager for dynamic market universe selection.
    
    Supports configurable universe sizes and comprehensive filtering.
    Maintains universe_version for reproducibility.
    """
    
    def __init__(self, 
                 universe_size: int = 100,
                 filter_config: Optional[Dict[str, Any]] = None,
                 exclusion_list: Optional[List[str]] = None,
                 universe_version: Optional[str] = None):
        
        self.universe_size = universe_size
        self.filter_config = filter_config or {}
        self.exclusion_list = exclusion_list or []
        self.universe_version = universe_version or self._generate_version()
        
        # Initialize filter with config
        self.filter = UniverseFilter(
            min_price=self.filter_config.get("min_price", 1.0),
            max_price=self.filter_config.get("max_price", 10000.0),
            min_volume=self.filter_config.get("min_volume", 100000),
            min_average_traded_value=self.filter_config.get("min_average_traded_value", 1000000),
            max_penny_stocks=self.filter_config.get("max_penny_stocks", True),
            exclusion_list=self.exclusion_list,
            max_missing_bars=self.filter_config.get("max_missing_bars", 0.1),
            min_data_completeness=self.filter_config.get("min_data_completeness", 0.95),
        )
    
    def _generate_version(self) -> str:
        """Generate a version string based on config for reproducibility."""
        import hashlib
        config_str = f"{self.universe_size}_{self.filter_config}_{len(self.exclusion_list)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_eligible_instruments(self, 
                                 watchlist: Optional[List[str]] = None,
                                 extra_universe: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get eligible instruments from watchlist + optional extra universe.
        
        Returns list of dicts with instrument info and filter scores.
        """
        # Start with watchlist or default
        wl = watchlist or self._get_default_watchlist()
        
        # Add extra universe if provided
        all_symbols = wl
        if extra_universe:
            all_symbols.extend(extra_universe)
        
        # Deduplicate
        all_symbols = list(set([s.upper() for s in all_symbols]))
        
        # Fetch data and filter
        eligible = []
        
        for symbol in all_symbols:
            try:
                ohlcv = get_ohlcv(symbol, interval="ONE_DAY", days=365)
                quote = get_quote(symbol)
                
                if self.filter.validate_instrument(symbol, ohlcv, quote):
                    # Calculate ranking scores
                    scores = self._calculate_ranking_scores(ohlcv, quote)
                    eligible.append({
                        "symbol": symbol,
                        "universe_version": self.universe_version,
                        **scores
                    })
            except Exception as e:
                # Skip symbols with data errors
                continue
        
        # Rank by composite score and return top universe_size
        eligible.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return eligible[:self.universe_size]
    
    def _calculate_ranking_scores(self, ohlcv: pd.DataFrame, quote: Dict) -> Dict[str, Any]:
        """Calculate ranking scores for an instrument."""
        last_close = ohlcv["close"].iloc[-1]
        avg_volume = ohlcv["volume"].rolling(20).mean().iloc[-1]
        
        # Volume score (normalized)
        volume_score = min(avg_volume / 10000000, 1.0)  # Cap at 10M volume
        
        # Price momentum score
        if len(ohlcv) > 20:
            mom_10 = (ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[-11]) - 1
            momentum_score = abs(mom_10)  # Higher absolute momentum
        else:
            momentum_score = 0.0
        
        # Volatility score (inverse - lower volatility preferred for stability)
        returns = ohlcv["close"].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std()
            volatility_score = max(0, 1.0 - vol * 100)  # Lower vol = higher score
        else:
            volatility_score = 0.5
        
        # Composite score (weighted combination)
        composite_score = (
            0.4 * volume_score +
            0.3 * momentum_score +
            0.3 * volatility_score
        )
        
        return {
            "last_price": last_close,
            "average_volume": int(avg_volume) if not pd.isna(avg_volume) else 0,
            "momentum_10d": round(momentum_score, 4),
            "volatility_score": round(volatility_score, 4),
            "composite_score": round(composite_score, 4),
        }
    
    def _get_default_watchlist(self) -> List[str]:
        """Get default watchlist of major NSE symbols."""
        return [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "ONGC", "COALINDIA",
            "BajajFinance", "KOTAKMAHIND", "ASIANPAINT", "MARUTI",
            "HCLTECH", "ITC", "TITAN", "NTPC", "POWERGRID",
            "UPL", "DIVISLAB", "CIPLA", "DRREDDY", "WIPRO",
        ]
    
    def update_universe_size(self, new_size: int) -> None:
        """Update the universe size."""
        self.universe_size = new_size
    
    def add_exclusion(self, symbol: str) -> None:
        """Add a symbol to the exclusion list."""
        self.exclusion_list.append(symbol.upper())
    
    def set_filter_config(self, config: Dict[str, Any]) -> None:
        """Update filter configuration."""
        self.filter_config = config
        self.filter = UniverseFilter(
            min_price=config.get("min_price", 1.0),
            max_price=config.get("max_price", 10000.0),
            min_volume=config.get("min_volume", 100000),
            min_average_traded_value=config.get("min_average_traded_value", 1000000),
            max_penny_stocks=config.get("max_penny_stocks", True),
            exclusion_list=self.exclusion_list,
            max_missing_bars=config.get("max_missing_bars", 0.1),
            min_data_completeness=config.get("min_data_completeness", 0.95),
        )
    
    def get_universe_info(self) -> Dict[str, Any]:
        """Return universe configuration info for reproducibility."""
        return {
            "universe_size": self.universe_size,
            "universe_version": self.universe_version,
            "exclusion_list": list(self.exclusion_list),
            "filter_config": self.filter_config,
        }