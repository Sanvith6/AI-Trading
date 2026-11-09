"""
Centralized US Market Session Calendar & Service.

Enforces:
1. Strict timezone-aware timestamps (America/New_York).
2. Distinction between REGULAR (09:30 - 16:00 EST), PRE_MARKET (04:00 - 09:30),
   POST_MARKET (16:00 - 20:00), CLOSED, and HOLIDAY.
3. Accurate US equity holidays (NYSE / NASDAQ calendar).
4. Fails safe: Normal strategy execution is blocked outside REGULAR hours
   unless extended hours are explicitly enabled.
"""

from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo
from enum import Enum
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass


class SessionStatus(str, Enum):
    REGULAR = "REGULAR"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"
    HOLIDAY = "HOLIDAY"
    WEEKEND = "WEEKEND"


@dataclass(frozen=True)
class MarketSessionInfo:
    session_date: str  # YYYY-MM-DD
    current_time: str  # ISO 8601 with timezone
    session_open: str  # ISO 8601
    session_close: str  # ISO 8601
    timezone: str  # "America/New_York"
    status: SessionStatus
    is_regular_trading_open: bool
    can_execute_orders: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_date": self.session_date,
            "current_time": self.current_time,
            "session_open": self.session_open,
            "session_close": self.session_close,
            "timezone": self.timezone,
            "status": self.status.value,
            "is_regular_trading_open": self.is_regular_trading_open,
            "can_execute_orders": self.can_execute_orders,
            "reason": self.reason,
        }


class USMarketCalendar:
    """Accurate exchange-level US Market Session service (NYSE/NASDAQ)."""

    NY_TZ = ZoneInfo("America/New_York")

    # Fixed and known US market holidays for 2025-2027
    US_HOLIDAYS: Set[date] = {
        # 2025
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Jr. Day
        date(2025, 2, 17),  # Washington's Birthday (Presidents' Day)
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
        # 2026
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # MLK Jr. Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (Observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving Day
        date(2026, 12, 25), # Christmas Day
        # 2027
        date(2027, 1, 1),   # New Year's Day
    }

    # Fixed and known US market half-days (early close at 13:00 America/New_York)
    US_HALF_DAYS: Set[date] = {
        date(2025, 7, 3),    # Day before Independence Day
        date(2025, 11, 28),  # Day after Thanksgiving (Black Friday)
        date(2025, 12, 24),  # Christmas Eve
        date(2026, 11, 27),  # Day after Thanksgiving (Black Friday)
        date(2026, 12, 24),  # Christmas Eve
        date(2027, 7, 2),    # Day before Independence Day (Observed)
        date(2027, 11, 26),  # Day after Thanksgiving (Black Friday)
        date(2027, 12, 24),  # Christmas Eve
    }

    REGULAR_OPEN = dtime(9, 30)
    REGULAR_CLOSE = dtime(16, 0)
    EARLY_CLOSE = dtime(13, 0)
    PRE_MARKET_OPEN = dtime(4, 0)
    POST_MARKET_CLOSE = dtime(20, 0)

    @classmethod
    def get_session_info(
        cls,
        dt: Optional[datetime] = None,
        allow_extended_hours: bool = False,
    ) -> MarketSessionInfo:
        """
        Determines current US market session status in America/New_York timezone.
        Seamlessly handles EST, EDT, and half-day 13:00 early closes.
        """
        if dt is None:
            now_ny = datetime.now(cls.NY_TZ)
        else:
            if dt.tzinfo is None:
                # Assume UTC if naive, then convert to NY
                now_ny = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(cls.NY_TZ)
            else:
                now_ny = dt.astimezone(cls.NY_TZ)

        cur_date = now_ny.date()
        cur_time = now_ny.time()

        is_half_day = cur_date in cls.US_HALF_DAYS
        active_close_time = cls.EARLY_CLOSE if is_half_day else cls.REGULAR_CLOSE

        open_dt = datetime.combine(cur_date, cls.REGULAR_OPEN, tzinfo=cls.NY_TZ)
        close_dt = datetime.combine(cur_date, active_close_time, tzinfo=cls.NY_TZ)

        # Check Weekend
        if cur_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return MarketSessionInfo(
                session_date=cur_date.strftime("%Y-%m-%d"),
                current_time=now_ny.isoformat(),
                session_open=open_dt.isoformat(),
                session_close=close_dt.isoformat(),
                timezone="America/New_York",
                status=SessionStatus.WEEKEND,
                is_regular_trading_open=False,
                can_execute_orders=False,
                reason="Market closed: Weekend.",
            )

        # Check Holiday
        if cur_date in cls.US_HOLIDAYS:
            return MarketSessionInfo(
                session_date=cur_date.strftime("%Y-%m-%d"),
                current_time=now_ny.isoformat(),
                session_open=open_dt.isoformat(),
                session_close=close_dt.isoformat(),
                timezone="America/New_York",
                status=SessionStatus.HOLIDAY,
                is_regular_trading_open=False,
                can_execute_orders=False,
                reason="Market closed: US Federal/Exchange Holiday.",
            )

        # Check Regular Market Hours (accounts for 13:00 early close on half-days)
        if cls.REGULAR_OPEN <= cur_time < active_close_time:
            reason = "Market open: Regular trading session active (Early Close 13:00 ET)." if is_half_day else "Market open: Regular trading session active."
            return MarketSessionInfo(
                session_date=cur_date.strftime("%Y-%m-%d"),
                current_time=now_ny.isoformat(),
                session_open=open_dt.isoformat(),
                session_close=close_dt.isoformat(),
                timezone="America/New_York",
                status=SessionStatus.REGULAR,
                is_regular_trading_open=True,
                can_execute_orders=True,
                reason=reason,
            )

        # Check Pre-Market Hours
        if cls.PRE_MARKET_OPEN <= cur_time < cls.REGULAR_OPEN:
            return MarketSessionInfo(
                session_date=cur_date.strftime("%Y-%m-%d"),
                current_time=now_ny.isoformat(),
                session_open=open_dt.isoformat(),
                session_close=close_dt.isoformat(),
                timezone="America/New_York",
                status=SessionStatus.PRE_MARKET,
                is_regular_trading_open=False,
                can_execute_orders=allow_extended_hours,
                reason="Pre-market session active. Standard execution blocked." if not allow_extended_hours else "Pre-market session open.",
            )

        # Check Post-Market Hours
        if cls.REGULAR_CLOSE <= cur_time < cls.POST_MARKET_CLOSE:
            return MarketSessionInfo(
                session_date=cur_date.strftime("%Y-%m-%d"),
                current_time=now_ny.isoformat(),
                session_open=open_dt.isoformat(),
                session_close=close_dt.isoformat(),
                timezone="America/New_York",
                status=SessionStatus.POST_MARKET,
                is_regular_trading_open=False,
                can_execute_orders=allow_extended_hours,
                reason="Post-market session active. Standard execution blocked." if not allow_extended_hours else "Post-market session open.",
            )

        # Overnight / Closed
        return MarketSessionInfo(
            session_date=cur_date.strftime("%Y-%m-%d"),
            current_time=now_ny.isoformat(),
            session_open=open_dt.isoformat(),
            session_close=close_dt.isoformat(),
            timezone="America/New_York",
            status=SessionStatus.CLOSED,
            is_regular_trading_open=False,
            can_execute_orders=False,
            reason="Market closed: Overnight period.",
        )
