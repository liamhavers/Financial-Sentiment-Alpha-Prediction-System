"""
Phase 2, fourth method: RoBERTa fine-tuned directly on StockTwits data.

Uses `zhayunduo/roberta-base-stocktwits-finetuned` — a RoBERTa-base model
fine-tuned on StockTwits posts (informal social-media finance text), unlike
`sentiment_finbert.py`'s FinBERT which is tuned on formal financial
news/analyst reports. Added alongside FinBERT (not replacing it) so Phase 3
can compare all methods; FinBERT's zero-shot-on-out-of-domain-text result
(14.1% accuracy) is a real, honest finding worth keeping, not something to
erase in favor of a better-suited model.

Like FinBERT, this model is used zero-shot with respect to *our* labeled
data: it was pretrained/fine-tuned by its author on other StockTwits data,
not fit on this project's messages, so there's no train/test split to
construct here either — evaluation against our StockTwits labels uses all
616 labeled messages directly.

The model's own label set is binary (Negative/Positive — confirmed via
AutoConfig.id2label), matching `sentiment_tfidf.py`'s binary behavior rather
than LM/FinBERT's 3-class one: every message is forced into bullish/bearish,
never neutral. Labels are normalized to bullish/bearish so this method's
output is directly comparable to the others in `sentiment_scores`.
"""

import logging
from datetime import datetime, timezone

import psycopg2.extras
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import db
import deduplicate
import timestamp_alignment as ta

METHOD = "stocktwits_roberta"
MODEL_NAME = "zhayunduo/roberta-base-stocktwits-finetuned"
BATCH_SIZE = 16
MAX_LENGTH = 128

LABEL_MAP = {"positive": "bullish", "negative": "bearish"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_tokenizer = None
_model = None


def get_model():
    """Lazily load the model (downloads weights from HuggingFace Hub on first use)."""
    global _tokenizer, _model
    if _model is None:
        log.info("Loading %s ...", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_batch(bodies: list[str], tokenizer, model) -> list[dict]:
    """Run one batch through the model, returning per-message class
    probabilities keyed by the model's own (lowercased) label names."""
    inputs = tokenizer(
        bodies, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label
    results = []
    for row in probs:
        results.append({id2label[i].lower(): float(p) for i, p in enumerate(row)})
    return results


def to_rows(message_rows: list[tuple], tokenizer, model) -> list[tuple]:
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(0, len(message_rows), BATCH_SIZE):
        batch = message_rows[i:i + BATCH_SIZE]
        bodies = [row[4] or "" for row in batch]
        scored = score_batch(bodies, tokenizer, model)

        for row, probs in zip(batch, scored):
            source, message_id, ticker = row[0], row[1], row[2]
            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            label = "bullish" if p_pos >= p_neg else "bearish"
            score = p_pos - p_neg
            rows.append((
                METHOD,
                source,
                message_id,
                ticker,
                score,
                label,
                psycopg2.extras.Json({
                    "proba_positive": p_pos,
                    "proba_negative": p_neg,
                }),
                now,
            ))
        log.info("  scored %d/%d messages", min(i + BATCH_SIZE, len(message_rows)), len(message_rows))
    return rows


def evaluate_against_stocktwits(message_rows: list[tuple], tokenizer, model) -> None:
    """Out-of-the-box comparison against StockTwits' own bullish/bearish tag,
    on every labeled message (no train/test split — this model wasn't fit on
    any of our data, so there's no held-out set to construct)."""
    labeled = [row for row in message_rows if row[6] in ("Bullish", "Bearish")]
    if not labeled:
        return

    bodies = [row[4] or "" for row in labeled]
    truth = ["bullish" if row[6] == "Bullish" else "bearish" for row in labeled]

    preds = []
    for i in range(0, len(bodies), BATCH_SIZE):
        batch = bodies[i:i + BATCH_SIZE]
        scored = score_batch(batch, tokenizer, model)
        for probs in scored:
            p_pos = probs.get("positive", 0.0)
            p_neg = probs.get("negative", 0.0)
            preds.append("bullish" if p_pos >= p_neg else "bearish")

    correct = sum(1 for p, t in zip(preds, truth) if p == t)
    log.info(
        "StockTwits-RoBERTa vs StockTwits label (n=%d, zero-shot on our data, no train/test split): "
        "accuracy=%.3f",
        len(truth), correct / len(truth),
    )


def run(tickers: list[str] = None) -> None:
    tickers = tickers if tickers is not None else config.TICKERS
    conn = db.get_connection()
    db.init_db(conn)
    trading_days = ta.load_trading_days(conn)

    message_rows = []
    for ticker in tickers:
        message_rows.extend(deduplicate.deduplicated_messages_for_ticker(conn, ticker, trading_days))

    if not message_rows:
        log.info("No messages to score.")
        conn.close()
        return

    tokenizer, model = get_model()

    evaluate_against_stocktwits(message_rows, tokenizer, model)

    rows = to_rows(message_rows, tokenizer, model)
    stored = db.store_sentiment_scores(conn, rows)
    log.info("Scored and stored %d messages (method=%s).", stored, METHOD)

    conn.close()
    log.info("StockTwits-RoBERTa sentiment scoring done.")


if __name__ == "__main__":
    run()
