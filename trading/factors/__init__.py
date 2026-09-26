"""
Factor Exposure & Attribution Module for Phase 1.
Computes multi-factor risks (Market, Momentum, Size, Value, Quality, Volatility, Sector)
and decomposes portfolio returns into true stock-selection Alpha vs Factor Beta.
"""

from .factor_model import FactorModel, PortfolioFactorProfile

__all__ = [
    "FactorModel",
    "PortfolioFactorProfile",
]
