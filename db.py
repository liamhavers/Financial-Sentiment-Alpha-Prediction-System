"""
Shared Postgres storage layer for the ingestion pipeline.

Uses a single `messages` table, distinguished by a `source` column, so
additional data sources can be added later without a schema change. Reddit
ingestion was removed (see CLAUDE.md) due to API access limitations, but
existing Reddit-sourced rows are left untouched under source='reddit'.
Fields that only apply to one source (e.g. StockTwits' user-tagged
sentiment label) either get a dedicated nullable column or go in the
`extra` JSONB column if source-specific and unlikely to be queried
directly at this stage.

Primary key is (source, message_id) rather than a surrogate key, since the
natural key from each platform is already unique and this makes idempotent
re-runs straightforward via ON CONFLICT DO NOTHING.
"""

import logging
from typing import Iterable

import psycopg2
import psycopg2.extras

import config

log = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DATABASE,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
    )


def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                source          TEXT NOT NULL,
                message_id      TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                created_at_utc  TIMESTAMPTZ NOT NULL,
                body            TEXT NOT NULL,
                author          TEXT,
                sentiment_label TEXT,           -- StockTwits user-tagged label only
                score           INTEGER,         -- likes (StockTwits)
                extra           JSONB,           -- source-specific extras
                ingested_at_utc TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (source, message_id)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_created ON messages (ticker, created_at_utc)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON messages (source)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker          TEXT NOT NULL,
                date            DATE NOT NULL,
                open            NUMERIC,
                high            NUMERIC,
                low             NUMERIC,
                close           NUMERIC,
                adj_close       NUMERIC,
                volume          BIGINT,
                ingested_at_utc TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Point-in-time snapshot, not a time series — one row per ticker,
        # overwritten on each ingestion run (see upsert_fundamentals).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker          TEXT PRIMARY KEY,
                sector          TEXT,
                industry        TEXT,
                market_cap      BIGINT,
                updated_at_utc  TIMESTAMPTZ NOT NULL
            )
        """)

        # Per-message sentiment scores. `method` discriminates which Phase 2
        # technique produced the row (e.g. 'lm_dict', 'tfidf_logreg',
        # 'finbert'), so all three can coexist without a schema change —
        # same discriminator pattern as `messages.source`. `score` is
        # normalized to roughly [-1, 1] across methods for comparability;
        # method-specific detail (e.g. LM's raw pos/neg counts and
        # subjectivity, or FinBERT's class probabilities) goes in `extra`.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                method          TEXT NOT NULL,
                source          TEXT NOT NULL,
                message_id      TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                score           NUMERIC,
                label           TEXT,
                extra           JSONB,
                scored_at_utc   TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (method, source, message_id)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_scores (ticker, method)"
        )

        # Daily/entity-level aggregate of sentiment_scores, one row per
        # (ticker, trading_day, method). mean_score is NULL when
        # message_count is below config.MIN_DAILY_MESSAGES (the Phase 0
        # volume floor) — treated as missing signal, not a noisy one, per
        # CLAUDE.md/README's Phase 0 Decisions.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_sentiment (
                ticker              TEXT NOT NULL,
                trading_day         DATE NOT NULL,
                method              TEXT NOT NULL,
                mean_score          NUMERIC,
                message_count       INTEGER NOT NULL,
                pct_bullish         NUMERIC,
                pct_bearish         NUMERIC,
                pct_neutral         NUMERIC,
                above_volume_floor  BOOLEAN NOT NULL,
                computed_at_utc     TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (ticker, trading_day, method)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_method ON daily_sentiment (method, trading_day)"
        )
    conn.commit()
    log.info("Schema ready.")


def get_last_seen_native_id(conn, source: str, ticker: str) -> str | None:
    """Most recent message_id already stored for this source/ticker, used as
    a pagination cursor where the source's API supports one (StockTwits)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT message_id FROM messages
               WHERE source = %s AND ticker = %s
               ORDER BY created_at_utc DESC LIMIT 1""",
            (source, ticker),
        )
        row = cur.fetchone()
        return row[0] if row else None


def store_messages(conn, rows: Iterable[tuple]) -> int:
    """
    rows: iterable of tuples matching column order:
    (source, message_id, ticker, created_at_utc, body, author,
     sentiment_label, score, extra, ingested_at_utc)

    `extra` should already be wrapped in psycopg2.extras.Json(...) by the
    caller if it's a dict, or None.
    """
    rows = list(rows)
    if not rows:
        return 0

    with conn.cursor() as cur:
        # fetch=True + RETURNING: execute_values pages large inserts (default
        # page_size=100), and cur.rowcount only reflects the last page, not
        # the cumulative total. fetch=True concatenates each page's RETURNING
        # rows instead, giving an accurate count.
        inserted_rows = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO messages
               (source, message_id, ticker, created_at_utc, body, author,
                sentiment_label, score, extra, ingested_at_utc)
               VALUES %s
               ON CONFLICT (source, message_id) DO NOTHING
               RETURNING 1""",
            rows,
            fetch=True,
        )
        inserted = len(inserted_rows)
    conn.commit()
    return inserted


def store_prices(conn, rows: Iterable[tuple]) -> int:
    """
    rows: iterable of tuples matching column order:
    (ticker, date, open, high, low, close, adj_close, volume, ingested_at_utc)
    """
    rows = list(rows)
    if not rows:
        return 0

    with conn.cursor() as cur:
        inserted_rows = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO prices
               (ticker, date, open, high, low, close, adj_close, volume, ingested_at_utc)
               VALUES %s
               ON CONFLICT (ticker, date) DO NOTHING
               RETURNING 1""",
            rows,
            fetch=True,
        )
        inserted = len(inserted_rows)
    conn.commit()
    return inserted


def store_sentiment_scores(conn, rows: Iterable[tuple]) -> int:
    """
    rows: iterable of tuples matching column order:
    (method, source, message_id, ticker, score, label, extra, scored_at_utc)

    ON CONFLICT DO UPDATE rather than DO NOTHING: re-scoring the same
    message with the same method (e.g. after a dictionary/model tweak)
    should overwrite the prior score, not silently no-op.
    """
    rows = list(rows)
    if not rows:
        return 0

    with conn.cursor() as cur:
        affected_rows = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO sentiment_scores
               (method, source, message_id, ticker, score, label, extra, scored_at_utc)
               VALUES %s
               ON CONFLICT (method, source, message_id) DO UPDATE SET
                   score = EXCLUDED.score,
                   label = EXCLUDED.label,
                   extra = EXCLUDED.extra,
                   scored_at_utc = EXCLUDED.scored_at_utc
               RETURNING 1""",
            rows,
            fetch=True,
        )
        affected = len(affected_rows)
    conn.commit()
    return affected


def store_daily_sentiment(conn, rows: Iterable[tuple]) -> int:
    """
    rows: iterable of tuples matching column order:
    (ticker, trading_day, method, mean_score, message_count, pct_bullish,
     pct_bearish, pct_neutral, above_volume_floor, computed_at_utc)

    ON CONFLICT DO UPDATE rather than DO NOTHING: re-aggregating (e.g. after
    new messages are ingested for a day, or a method is re-scored) should
    overwrite the prior daily row, not silently no-op.
    """
    rows = list(rows)
    if not rows:
        return 0

    with conn.cursor() as cur:
        affected_rows = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO daily_sentiment
               (ticker, trading_day, method, mean_score, message_count,
                pct_bullish, pct_bearish, pct_neutral, above_volume_floor,
                computed_at_utc)
               VALUES %s
               ON CONFLICT (ticker, trading_day, method) DO UPDATE SET
                   mean_score = EXCLUDED.mean_score,
                   message_count = EXCLUDED.message_count,
                   pct_bullish = EXCLUDED.pct_bullish,
                   pct_bearish = EXCLUDED.pct_bearish,
                   pct_neutral = EXCLUDED.pct_neutral,
                   above_volume_floor = EXCLUDED.above_volume_floor,
                   computed_at_utc = EXCLUDED.computed_at_utc
               RETURNING 1""",
            rows,
            fetch=True,
        )
        affected = len(affected_rows)
    conn.commit()
    return affected


def upsert_fundamentals(conn, row: tuple) -> None:
    """
    row: (ticker, sector, industry, market_cap, updated_at_utc)

    Fundamentals are a current snapshot per ticker rather than a time
    series, so this overwrites the prior snapshot instead of appending.
    """
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO fundamentals
               (ticker, sector, industry, market_cap, updated_at_utc)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (ticker) DO UPDATE SET
                   sector = EXCLUDED.sector,
                   industry = EXCLUDED.industry,
                   market_cap = EXCLUDED.market_cap,
                   updated_at_utc = EXCLUDED.updated_at_utc""",
            row,
        )
    conn.commit()
