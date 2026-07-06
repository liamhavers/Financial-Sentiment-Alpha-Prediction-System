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
