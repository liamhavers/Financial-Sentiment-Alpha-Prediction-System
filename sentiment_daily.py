"""
Phase 2, final step: aggregate per-message sentiment_scores into a
daily/entity-level signal — one row per (ticker, trading_day, method) in the
new `daily_sentiment` table (see db.py).

Each of the three Phase 2 methods (lm_dict, tfidf_logreg, finbert) scored the
same deduplicated message set independently, so aggregation runs per method
rather than combining them into one blended score — keeping them separate
lets Phase 3 evaluate each method's IC independently rather than baking a
premature "best method" choice into the signal.

Messages are bucketed by their aligned trading day (timestamp_alignment.py),
not calendar day, so the no-look-ahead guarantee established at ingestion
carries through into the daily signal. Below config.MIN_DAILY_MESSAGES
(the Phase 0 volume floor), a ticker/day's mean_score is stored as NULL —
treated as missing signal rather than a noisy one — while message_count and
above_volume_floor are still recorded so Phase 3 can see why it's missing.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

import config
import db
import timestamp_alignment as ta

METHODS = ["lm_dict", "tfidf_logreg", "finbert"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_scores_with_timestamps(conn, method: str, tickers: list[str]) -> list[tuple]:
    """(ticker, created_at_utc, score, label) for every sentiment_scores row
    under `method`, joined back to messages for the timestamp needed to align
    to a trading day. sentiment_scores was itself built from
    deduplicated_messages_for_ticker, so this join carries no duplicates."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT s.ticker, m.created_at_utc, s.score, s.label
               FROM sentiment_scores s
               JOIN messages m ON s.source = m.source AND s.message_id = m.message_id
               WHERE s.method = %s AND s.ticker = ANY(%s)""",
            (method, tickers),
        )
        return cur.fetchall()


def bucket_by_trading_day(rows: list[tuple], trading_days: list) -> dict:
    """Group (score, label) pairs by (ticker, aligned trading_day)."""
    buckets = defaultdict(list)
    for ticker, created_at_utc, score, label in rows:
        trading_day = ta.align_to_trading_day(created_at_utc, trading_days)
        if trading_day is None:
            continue
        buckets[(ticker, trading_day)].append((score, label))
    return buckets


def to_daily_rows(buckets: dict, method: str) -> list[tuple]:
    now = datetime.now(timezone.utc)
    rows = []
    for (ticker, trading_day), items in buckets.items():
        n = len(items)
        above_floor = n >= config.MIN_DAILY_MESSAGES

        scores = [float(s) for s, _label in items if s is not None]
        mean_score = (sum(scores) / len(scores)) if (above_floor and scores) else None

        labels = [label for _score, label in items]
        pct_bullish = labels.count("bullish") / n
        pct_bearish = labels.count("bearish") / n
        pct_neutral = labels.count("neutral") / n

        rows.append((
            ticker, trading_day, method, mean_score, n,
            pct_bullish, pct_bearish, pct_neutral, above_floor, now,
        ))
    return rows


def run(tickers: list[str] = None) -> None:
    tickers = tickers if tickers is not None else config.TICKERS
    conn = db.get_connection()
    db.init_db(conn)
    trading_days = ta.load_trading_days(conn)

    total_stored = 0
    for method in METHODS:
        rows = load_scores_with_timestamps(conn, method, tickers)
        if not rows:
            log.info("method=%s: no sentiment_scores rows found, skipping", method)
            continue

        buckets = bucket_by_trading_day(rows, trading_days)
        daily_rows = to_daily_rows(buckets, method)
        stored = db.store_daily_sentiment(conn, daily_rows)
        total_stored += stored

        below_floor = sum(1 for r in daily_rows if not r[8])
        log.info(
            "method=%s: %d ticker-days aggregated (%d below volume floor of %d, mean_score=NULL)",
            method, len(daily_rows), below_floor, config.MIN_DAILY_MESSAGES,
        )

    conn.close()
    log.info("Daily sentiment aggregation done (%d rows stored).", total_stored)


if __name__ == "__main__":
    run()
