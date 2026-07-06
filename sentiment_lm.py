"""
Phase 2 baseline: Loughran-McDonald finance-specific dictionary sentiment.

Uses `pysentiment2`'s bundled LM word lists (the standard finance sentiment
dictionary, built from 10-K/10-Q filings — see
https://www3.nd.edu/~mcdonald/Word_Lists.html). Scores each deduplicated
message independently; no cross-message aggregation happens here (that's a
later Phase 2 step: "aggregate document-level scores into daily/entity-level
sentiment signal").

Known limitation (expected, not a bug): LM is tuned for formal filing
language. Informal social-media phrasing (e.g. "beat estimates") often
scores neutral since those terms aren't in the dictionary, while formal
words like "risk" or "litigation" score negative regardless of context.
This is exactly why classical ML (TF-IDF/logreg) and FinBERT are the next
two methods in the roadmap, to compare against this baseline.

Scores are stored in the shared `sentiment_scores` table (see db.py) under
method='lm_dict', so they can be compared against the other two Phase 2
methods once those exist, and against StockTwits' own user-tagged label
(already present as `messages.sentiment_label`).
"""

import logging
from datetime import datetime, timezone

import psycopg2.extras
import pysentiment2 as ps

import config
import db
import deduplicate
import timestamp_alignment as ta

METHOD = "lm_dict"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_lm = None


def get_lm() -> ps.LM:
    """Lazily build the LM dictionary (loads word lists + NLTK stemmer once)."""
    global _lm
    if _lm is None:
        _lm = ps.LM()
    return _lm


def score_body(body: str, lm: ps.LM = None) -> dict:
    """
    Score a single message body against the LM dictionary.

    Returns Polarity (-1..1, sign of net tone) and Subjectivity (0..1,
    fraction of tokens that carried sentiment) as defined in pysentiment2,
    plus the raw positive/negative hit counts.
    """
    lm = lm or get_lm()
    tokens = lm.tokenize(body or "")
    result = lm.get_score(tokens)
    return {
        "positive": int(result["Positive"]),
        "negative": int(result["Negative"]),
        "polarity": float(result["Polarity"]),
        "subjectivity": float(result["Subjectivity"]),
    }


def score_to_label(polarity: float, subjectivity: float) -> str:
    """Bucket a polarity/subjectivity pair into bullish/bearish/neutral so
    this method's output is directly comparable to StockTwits' own
    bullish/bearish user label."""
    if subjectivity == 0:
        return "neutral"
    if polarity > 0:
        return "bullish"
    if polarity < 0:
        return "bearish"
    return "neutral"


def to_rows(message_rows: list[tuple], lm: ps.LM = None) -> list[tuple]:
    """
    message_rows: tuples in `messages` column order — (source, message_id,
    ticker, created_at_utc, body, author, sentiment_label, score, extra,
    ingested_at_utc) — as returned by deduplicate.deduplicated_messages_for_ticker.

    Returns rows matching `sentiment_scores` column order.
    """
    lm = lm or get_lm()
    now = datetime.now(timezone.utc)
    rows = []
    for source, message_id, ticker, _created_at, body, *_rest in message_rows:
        scored = score_body(body, lm)
        label = score_to_label(scored["polarity"], scored["subjectivity"])
        rows.append((
            METHOD,
            source,
            message_id,
            ticker,
            scored["polarity"],
            label,
            psycopg2.extras.Json({
                "positive": scored["positive"],
                "negative": scored["negative"],
                "subjectivity": scored["subjectivity"],
            }),
            now,
        ))
    return rows


def run(tickers: list[str] = None) -> None:
    tickers = tickers if tickers is not None else config.TICKERS
    conn = db.get_connection()
    db.init_db(conn)
    lm = get_lm()
    trading_days = ta.load_trading_days(conn)

    for ticker in tickers:
        log.info("Scoring (LM dict): %s ...", ticker)
        message_rows = deduplicate.deduplicated_messages_for_ticker(conn, ticker, trading_days)

        if not message_rows:
            log.info("  no messages for %s", ticker)
            continue

        rows = to_rows(message_rows, lm)
        stored = db.store_sentiment_scores(conn, rows)
        log.info("  scored %d messages for %s", stored, ticker)

    conn.close()
    log.info("LM dictionary sentiment scoring done.")


if __name__ == "__main__":
    run()
