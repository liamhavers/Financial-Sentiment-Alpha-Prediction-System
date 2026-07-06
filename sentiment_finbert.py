"""
Phase 2, third method: FinBERT (transformer-based) sentiment.

Uses `ProsusAI/finbert` — a BERT model fine-tuned on financial text
(analyst reports/news) for 3-class sentiment (positive/negative/neutral).
Unlike `sentiment_tfidf.py`, this model is used zero-shot: it is NOT
fine-tuned on our StockTwits labels, just applied as a pretrained
classifier. Comparison against StockTwits' own label is therefore an
out-of-the-box evaluation, not a held-out split of a model we trained —
there is no train/test split here because there's nothing of ours it was
fit on.

Labels are normalized to bullish/bearish/neutral (matching sentiment_lm's
convention) so all three Phase 2 methods are directly comparable in
`sentiment_scores`. Score is polarity in [-1, 1]: proba(positive) -
proba(negative).
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

METHOD = "finbert"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
MAX_LENGTH = 128

LABEL_MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_tokenizer = None
_model = None


def get_model():
    """Lazily load FinBERT (downloads weights from HuggingFace Hub on first use)."""
    global _tokenizer, _model
    if _model is None:
        log.info("Loading %s ...", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_batch(bodies: list[str], tokenizer, model) -> list[dict]:
    """Run one batch through FinBERT, returning per-message class
    probabilities keyed by the model's own label names."""
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
            p_neu = probs.get("neutral", 0.0)
            top_label = max(probs, key=probs.get)
            label = LABEL_MAP.get(top_label, "neutral")
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
                    "proba_neutral": p_neu,
                }),
                now,
            ))
        log.info("  scored %d/%d messages", min(i + BATCH_SIZE, len(message_rows)), len(message_rows))
    return rows


def evaluate_against_stocktwits(message_rows: list[tuple], tokenizer, model) -> None:
    """Out-of-the-box comparison against StockTwits' own bullish/bearish tag,
    on every labeled message (no train/test split — FinBERT wasn't fit on
    any of this data, so there's no held-out set to construct)."""
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
            top_label = max(probs, key=probs.get)
            preds.append(LABEL_MAP.get(top_label, "neutral"))

    correct = sum(1 for p, t in zip(preds, truth) if p == t)
    neutral_count = sum(1 for p in preds if p == "neutral")
    log.info(
        "FinBERT vs StockTwits label (n=%d, zero-shot, no train/test split): "
        "accuracy=%.3f (neutral predictions=%d/%d, scored as misses)",
        len(truth), correct / len(truth), neutral_count, len(truth),
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
    log.info("FinBERT sentiment scoring done.")


if __name__ == "__main__":
    run()
