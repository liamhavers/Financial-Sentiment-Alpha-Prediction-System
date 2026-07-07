"""
Price/fundamental data ingestion — pulls daily OHLCV bars for the fixed
ticker universe plus the QQQ benchmark (needed for the excess-return label;
see CLAUDE.md's Fixed Decisions) via yfinance, storing them in the shared
Postgres `prices` table (see db.py). Also pulls a lightweight fundamentals
snapshot (sector, industry, market cap) per ticker into `fundamentals`.

RDDT note: RDDT IPO'd March 2024, so yfinance simply returns no bars before
that date — no special-casing needed here. Any backtest/feature code must
still respect the window constraint documented in CLAUDE.md.

Look-ahead safety: each row is a full trading day's OHLCV bar, dated by
that day. Downstream feature code must still only use a given day's bar
once that day's session has closed — same constraint as the message data.
"""

import logging
import time
from datetime import datetime, timezone

import yfinance as yf

import config
import db

SECONDS_BETWEEN_TICKERS = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def price_universe() -> list[str]:
    return config.TICKERS + [config.BENCHMARK_TICKER]


def fetch_price_history(ticker: str):
    return yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)


def price_rows(ticker: str, history) -> list[tuple]:
    now = datetime.now(timezone.utc)
    rows = []
    for date, bar in history.iterrows():
        rows.append((
            ticker,
            date.date(),
            float(bar["Open"]),
            float(bar["High"]),
            float(bar["Low"]),
            float(bar["Close"]),
            float(bar["Adj Close"]),
            int(bar["Volume"]),
            now,
        ))
    return rows


def fetch_fundamentals(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        # yfinance's .info scrape is best-effort and occasionally throttled
        # or missing for a given ticker; don't let that break price ingestion.
        log.warning("Could not fetch fundamentals for %s: %s", ticker, e)
        return {}


def fundamentals_row(ticker: str, info: dict) -> tuple:
    now = datetime.now(timezone.utc)
    return (
        ticker,
        info.get("sector"),
        info.get("industry"),
        info.get("marketCap"),
        now,
    )


def run(tickers: list[str] = None) -> None:
    tickers = tickers if tickers is not None else price_universe()
    conn = db.get_connection()
    db.init_db(conn)

    for ticker in tickers:
        log.info("Fetching price history: %s ...", ticker)
        history = fetch_price_history(ticker)

        if history.empty:
            log.info("  no price data for %s", ticker)
        else:
            rows = price_rows(ticker, history)
            inserted = db.store_prices(conn, rows)
            log.info("  fetched %d bars, inserted %d new for %s", len(rows), inserted, ticker)

        info = fetch_fundamentals(ticker)
        if info:
            db.upsert_fundamentals(conn, fundamentals_row(ticker, info))
            log.info("  fundamentals updated for %s", ticker)

        time.sleep(SECONDS_BETWEEN_TICKERS)

    conn.close()
    log.info("Price/fundamentals ingestion done.")


if __name__ == "__main__":
    run()
