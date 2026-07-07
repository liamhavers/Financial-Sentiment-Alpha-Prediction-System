"""
Deduplication — collapses repeated/syndicated posts so a single account
reposting near-identical text doesn't inflate a trading day's message
volume or sentiment score.

Investigated directly in the ingested StockTwits data (see CLAUDE.md): some
accounts post the exact same body text more than once on the same day
(e.g. sharing the same article link twice), and some repost the exact same
body across different days (e.g. a recurring scheduled post). Distinct
authors independently posting a short identical body (e.g. just "$PLTR")
are common and NOT duplicates — only same-ticker + same-author + same-body
+ same aligned trading day collapses, keeping the earliest post.

This does not mutate stored data — raw ingested messages are kept as-is
(same philosophy as look-ahead handling: store raw, filter downstream).
Phase 2 sentiment scoring should fetch a ticker's message set through
`deduplicated_messages_for_ticker` rather than querying `messages`
directly, so repost inflation doesn't bias the score.
"""

from typing import Iterable

from ingestion import timestamp_alignment as ta


def deduplicate_messages(rows: Iterable[tuple], trading_days: list) -> list[tuple]:
    """
    rows: tuples matching the `messages` table column order — (source,
    message_id, ticker, created_at_utc, body, author, sentiment_label,
    score, extra, ingested_at_utc).

    Returns rows with duplicates collapsed: among rows sharing the same
    (ticker, author, body) that align to the same trading day (see
    timestamp_alignment.py), only the earliest-posted row is kept.
    """
    rows = sorted(rows, key=lambda r: r[3])  # created_at_utc ascending

    seen = set()
    deduped = []
    for row in rows:
        _, _, ticker, created_at_utc, body, author, *_ = row
        trading_day = ta.align_to_trading_day(created_at_utc, trading_days)
        key = (ticker, author, body, trading_day)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def deduplicated_messages_for_ticker(conn, ticker: str, trading_days: list = None) -> list[tuple]:
    """Fetch all stored StockTwits messages for `ticker` with syndicated reposts collapsed."""
    if trading_days is None:
        trading_days = ta.load_trading_days(conn)

    with conn.cursor() as cur:
        cur.execute(
            """SELECT source, message_id, ticker, created_at_utc, body, author,
                      sentiment_label, score, extra, ingested_at_utc
               FROM messages
               WHERE source = 'stocktwits' AND ticker = %s""",
            (ticker,),
        )
        rows = cur.fetchall()

    return deduplicate_messages(rows, trading_days)
