"""
Event Data Foundation for Phase 1.
Tracks SEC filings, earnings dates, and corporate announcements with strictly causal availability timestamps.
"""

from .event_foundation import EventFoundation, CorporateEvent

__all__ = [
    "EventFoundation",
    "CorporateEvent",
]
