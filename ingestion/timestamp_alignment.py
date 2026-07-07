"""
Timestamp alignment — maps a message/post timestamp to the trading day whose
forward return it may legitimately predict.

This enforces the no-look-ahead constraint documented in CLAUDE.md: a
message only counts toward a trading day's sentiment if it was posted
strictly before that day's market close (4pm ET). A message posted after
close, or on a weekend/holiday, rolls forward to the next real trading day.

Trading days are derived from the `prices` table's QQQ history rather than
a separate market-calendar dependency — QQQ has traded since 1999 with gaps
only on actual NYSE closures, so its date column is already a real trading
calendar "for free".
"""

import bisect
from datetime import date, datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

from ingestion import config, db

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = dtime(16, 0)  # 4:00 PM ET


def load_trading_days(conn, ticker: str = config.BENCHMARK_TICKER) -> list[date]:
    """Sorted list of dates `ticker` has a price bar for — used as a proxy
    for the real NYSE trading calendar."""
    with conn.cursor() as cur:
        cur.execute("SELECT date FROM prices WHERE ticker = %s ORDER BY date", (ticker,))
        return [row[0] for row in cur.fetchall()]


def align_to_trading_day(timestamp_utc: datetime, trading_days: list[date]) -> date | None:
    """
    Return the earliest trading day (from `trading_days`, sorted ascending)
    whose market close strictly follows `timestamp_utc`.

    A message posted before a trading day's 4pm ET close aligns to that day;
    a message posted at/after close, or on a non-trading day, rolls forward
    to the next trading day. Returns None if there is no later trading day
    in `trading_days` to align to.
    """
    if timestamp_utc.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware")

    local = timestamp_utc.astimezone(MARKET_TZ)
    local_date = local.date()

    idx = bisect.bisect_left(trading_days, local_date)

    if idx < len(trading_days) and trading_days[idx] == local_date:
        if local.time() < MARKET_CLOSE:
            return local_date
        idx += 1  # posted at/after close: roll to the next trading day

    return trading_days[idx] if idx < len(trading_days) else None
