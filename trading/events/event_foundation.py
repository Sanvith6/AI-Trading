"""
Event Data Foundation & Causal Event Registry.

Ensures that macro and corporate events (Earnings announcements, 10-K/10-Q SEC filings,
dividend declarations, guidance updates) are timestamped with strict causality:
- event_time: When the event occurred in the real world
- available_at: Exact UTC timestamp when the event became publicly accessible to market participants

Prevents lookahead leakage where a model trades ahead of an after-hours earnings release
or assumes 10-Q balance sheet availability prior to SEC EDGAR dissemination.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_FILE = Path(__file__).parent.parent.parent / "data_lake" / "events.json"


@dataclass
class CorporateEvent:
    event_id: str
    symbol: str
    event_type: str  # "EARNINGS_RELEASE", "SEC_FILING", "DIVIDEND_ANNOUNCEMENT", "GUIDANCE_REVISION"
    event_time: str  # "YYYY-MM-DDTHH:MM:SSZ"
    source: str  # "SEC_EDGAR", "COMPANY_IR", "EXCHANGE"
    available_at: str  # "YYYY-MM-DDTHH:MM:SSZ" (Strict Causality Boundary)
    importance: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    raw_reference: str  # Accession number, URL, or press release ID
    data_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventFoundation:
    """
    Causal Event Registry and Query Engine.
    Filters out any event whose available_at timestamp is later than the query as_of timestamp.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_EVENTS_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[CorporateEvent] = self._load()

    def _load(self) -> List[CorporateEvent]:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                return [CorporateEvent(**item) for item in data]
            except Exception as e:
                logger.error(f"Failed to read events storage: {e}")
        return []

    def _save(self) -> None:
        try:
            data = [e.to_dict() for e in self.events]
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write events storage: {e}")

    def record_event(self, event: CorporateEvent) -> None:
        """Register a new corporate or macro event."""
        # Check duplicate
        if any(e.event_id == event.event_id for e in self.events):
            return
        self.events.append(event)
        self._save()

    def get_events_for_symbol(
        self,
        symbol: str,
        as_of_time: str,
        event_types: Optional[List[str]] = None,
        lookback_days: int = 90,
    ) -> List[CorporateEvent]:
        """
        Query events for a symbol strictly available on or before as_of_time.
        Guarantees zero future information leakage.
        """
        clean_sym = symbol.strip().upper()
        as_of_dt = datetime.fromisoformat(as_of_time.replace("Z", "+00:00"))

        valid_events = []
        for e in self.events:
            if e.symbol.upper() != clean_sym:
                continue

            if event_types and e.event_type not in event_types:
                continue

            avail_dt = datetime.fromisoformat(e.available_at.replace("Z", "+00:00"))
            # CAUSALITY CHECK: Event must be publicly available AT or BEFORE as_of_time
            if avail_dt <= as_of_dt:
                # Lookback check
                diff_days = (as_of_dt - avail_dt).total_seconds() / 86400.0
                if diff_days <= lookback_days:
                    valid_events.append(e)

        return sorted(valid_events, key=lambda x: x.available_at, reverse=True)

    def is_earnings_blackout(self, symbol: str, as_of_time: str, blackout_days_pre: int = 1, blackout_days_post: int = 1) -> bool:
        """
        Returns True if as_of_time is within the blackout window around a known earnings date.
        """
        clean_sym = symbol.strip().upper()
        as_of_dt = datetime.fromisoformat(as_of_time.replace("Z", "+00:00"))

        for e in self.events:
            if e.symbol.upper() == clean_sym and e.event_type == "EARNINGS_RELEASE":
                ev_dt = datetime.fromisoformat(e.event_time.replace("Z", "+00:00"))
                diff_days = (as_of_dt - ev_dt).total_seconds() / 86400.0
                if -blackout_days_pre <= diff_days <= blackout_days_post:
                    return True
        return False
