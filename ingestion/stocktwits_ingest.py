"""
StockTwits ingestion — refactored to use the shared Postgres storage layer
(db.py). See run_all.py to run ingestion against the database.

NOTE ON API ACCESS:
Uses StockTwits' public streams/symbol endpoint. A browser-like User-Agent
header is required to avoid 403s (bot-protection, not a formal auth wall, as
confirmed in testing). StockTwits' official developer program is currently
not accepting new registrations; if that changes, add the key/header here.

Look-ahead safety: created_at_utc is stored as-is from StockTwits (UTC).
Downstream feature code is responsible for only using messages timestamped
strictly before a given trading day's market close.
"""

import logging
import time
from datetime import datetime, timezone

import requests
import psycopg2.extras

import config
import db

BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
SECONDS_BETWEEN_REQUESTS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 30
MAX_PAGES_PER_TICKER = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get(url: str, params: dict) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        except requests.RequestException as e:
            log.warning("Request error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            log.warning("Rate limited. Backing off for %ds.", RETRY_BACKOFF_SECONDS)
            time.sleep(RETRY_BACKOFF_SECONDS)
            continue

        log.error("Unexpected status %d for %s: %s", resp.status_code, url, resp.text[:300])
        return None

    log.error("Giving up on %s after %d attempts.", url, MAX_RETRIES)
    return None


def fetch_ticker_messages(ticker: str, since_id: str | None) -> list[dict]:
    all_messages: list[dict] = []
    max_id = None

    for _ in range(MAX_PAGES_PER_TICKER):
        params = {}
        if max_id is not None:
            params["max"] = max_id
        if since_id is not None:
            params["since"] = since_id

        data = _get(BASE_URL.format(symbol=ticker), params)
        time.sleep(SECONDS_BETWEEN_REQUESTS)

        if not data or "messages" not in data or not data["messages"]:
            break

        messages = data["messages"]
        all_messages.extend(messages)
        max_id = min(m["id"] for m in messages) - 1

        if since_id is not None and min(m["id"] for m in messages) <= int(since_id):
            break

    return all_messages


def to_rows(ticker: str, messages: list[dict]) -> list[tuple]:
    now = datetime.now(timezone.utc)
    rows = []
    for m in messages:
        entities = m.get("entities") or {}
        sentiment = entities.get("sentiment")
        sentiment_label = sentiment.get("basic") if sentiment else None
        rows.append((
            "stocktwits",
            str(m["id"]),
            ticker,
            m["created_at"],
            m.get("body", ""),
            m.get("user", {}).get("username"),
            sentiment_label,
            m.get("likes", {}).get("total", 0) if m.get("likes") else 0,
            psycopg2.extras.Json({"user_id": m.get("user", {}).get("id")}),
            now,
        ))
    return rows


def run(tickers: list[str] = config.TICKERS) -> None:
    conn = db.get_connection()
    db.init_db(conn)

    for ticker in tickers:
        log.info("Fetching StockTwits: %s ...", ticker)
        since_id = db.get_last_seen_native_id(conn, "stocktwits", ticker)
        messages = fetch_ticker_messages(ticker, since_id)

        if not messages:
            log.info("  no new messages for %s", ticker)
            continue

        rows = to_rows(ticker, messages)
        inserted = db.store_messages(conn, rows)
        log.info("  fetched %d, inserted %d new for %s", len(messages), inserted, ticker)

    conn.close()
    log.info("StockTwits ingestion done.")


if __name__ == "__main__":
    run()
