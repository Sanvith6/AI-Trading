"""
Fundamental Data Layer for Phase 1.
Provides point-in-time financial statements, SEC EDGAR metrics, and valuation ratios
with strict filing_date / available_at causality enforcement.
"""

from .edgar_provider import EdgarFundamentalProvider, FundamentalRecord

__all__ = [
    "EdgarFundamentalProvider",
    "FundamentalRecord",
]
