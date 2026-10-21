"""
Trading calendar helpers.

Adapted from FinRL-X `src/data/trading_calendar.py` (Apache 2.0, AI4Finance Foundation). The
pandas_market_calendars schedule is cached, and missing trading days are consolidated into fetch
ranges so the live provider is only called for gaps.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Iterable, List, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

_EXCHANGE_TZ = {"NYSE": "America/New_York", "NASDAQ": "America/New_York"}


def _iso(d: date | str | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


@lru_cache(maxsize=8)
def _calendar(exchange: str):
    return mcal.get_calendar(exchange)


@lru_cache(maxsize=512)
def _trading_days(exchange: str, start: str, end: str) -> Tuple[str, ...]:
    schedule = _calendar(exchange).schedule(start_date=start, end_date=end)
    return tuple(schedule.index.strftime("%Y-%m-%d").tolist())


def trading_days(start: date | str, end: date | str, exchange: str = "NYSE") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(_trading_days(exchange, _iso(start), _iso(end)))))


def trading_days_set(start: date | str, end: date | str, exchange: str = "NYSE") -> Set[str]:
    return set(_trading_days(exchange, _iso(start), _iso(end)))


def is_trading_day(d: date | str, exchange: str = "NYSE") -> bool:
    s = _iso(d)
    return s in trading_days_set(s, s, exchange)


def last_completed_trading_day(exchange: str = "NYSE", now: datetime | None = None) -> date:
    """Most recent trading day whose session has closed in the exchange time zone."""
    tz = ZoneInfo(_EXCHANGE_TZ.get(exchange.upper(), "America/New_York"))
    now = now.astimezone(tz) if now else datetime.now(tz)
    start = (now - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    schedule = _calendar(exchange).schedule(start_date=start, end_date=end)
    closed = schedule[schedule["market_close"] <= pd.Timestamp(now)]
    return closed.index[-1].date()


def missing_trading_days(
    existing: Iterable[date | str], start: date | str, end: date | str, exchange: str = "NYSE"
) -> List[str]:
    """Trading days in [start, end] with no entry in `existing`. Ignores future days."""
    end_dt = min(pd.Timestamp(end).date(), last_completed_trading_day(exchange))
    if pd.Timestamp(start).date() > end_dt:
        return []
    have = {_iso(d) for d in existing}
    return sorted(trading_days_set(start, end_dt, exchange) - have)


def consolidate_ranges(days: Iterable[str], max_gap_days: int = 4) -> List[Tuple[str, str]]:
    """Group sorted ISO dates into (start, end) ranges, merging gaps of at most `max_gap_days`
    calendar days (weekends and single holidays do not split a range)."""
    days = sorted(set(days))
    if not days:
        return []
    ranges: List[Tuple[str, str]] = []
    start = prev = pd.Timestamp(days[0])
    for d in days[1:]:
        cur = pd.Timestamp(d)
        if (cur - prev).days > max_gap_days:
            ranges.append((_iso(start), _iso(prev)))
            start = cur
        prev = cur
    ranges.append((_iso(start), _iso(prev)))
    return ranges
