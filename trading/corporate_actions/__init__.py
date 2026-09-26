"""
Corporate Actions Module for Phase 1.
Handles stock splits, dividends, symbol transitions, and three-tier price separation.
"""

from .corporate_action_engine import CorporateActionEngine, CorporateAction, PriceRepresentation

__all__ = [
    "CorporateActionEngine",
    "CorporateAction",
    "PriceRepresentation",
]
