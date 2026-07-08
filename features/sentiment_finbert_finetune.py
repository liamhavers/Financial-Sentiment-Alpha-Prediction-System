"""
Phase 2 extension: fine-tune FinBERT on our own labeled StockTwits data,
instead of running it zero-shot as sentiment_finbert.py does. Reuses
sentiment_tfidf.py's exact data loading and train/test split (same
TEST_SIZE/RANDOM_STATE/stratify) so held-out accuracy is directly comparable
to TF-IDF/LogReg and to FinBERT's own zero-shot result on identical data.

FinBERT's pretrained head stays 3-class (positive/negative/neutral) rather
than being resized to 2 classes: our labeled data is only ever
bullish/bearish (StockTwits never tags "Neutral"), so training targets map
onto FinBERT's existing positive/negative indices and the neutral index is
never a direct training target, but softmax normalization still shifts its
logit during training. This keeps the output label space identical to
sentiment_finbert.py's zero-shot method, so mean_score/label stay comparable
across both in sentiment_scores, rather than forcing a head resize purely
for architectural purity.

Like sentiment_tfidf.py, the held-out split exists only to report an honest
accuracy number — the final model used to score every message is a fresh
fine-tune on the FULL labeled set, not the train-only model.
"""

import logging
from datetime import datetime, timezone

import psycopg2.extras
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingestion import config, db, timestamp_alignment as ta
from features import finetune_common as ft
from features.sentiment_tfidf import load_all_deduplicated, labeled_examples, TEST_SIZE, RANDOM_STATE
from features.sentiment_finbert import score_batch, MODEL_NAME, BATCH_SIZE

METHOD = "finbert_finetuned"
EPOCHS = 3
LR = 2e-5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def label_ids_for(labels: list[str], label2id: dict) -> list[int]:
    return [label2id["positive"] if label == "bullish" else label2id["negative"] for label in labels]


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
            label = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}.get(top_label, "neutral")
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


def run(tickers: list[str] = None) -> None:
    tickers = tickers if tickers is not None else config.TICKERS
    conn = db.get_connection()
    db.init_db(conn)
    trading_days = ta.load_trading_days(conn)

    message_rows = load_all_deduplicated(conn, tickers, trading_days)
    bodies, labels = labeled_examples(message_rows)
    log.info("Loaded %d deduplicated messages, %d labeled (bullish=%d, bearish=%d)",
              len(message_rows), len(bodies), labels.count("bullish"), labels.count("bearish"))

    if len(bodies) < 20:
        log.warning("Too few labeled examples to fine-tune/evaluate — skipping.")
        conn.close()
        return

    log.info("Loading %s for fine-tuning ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_classes = len(id2label)

    X_train, X_test, y_train, y_test = train_test_split(
        bodies, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    log.info("Fine-tuning %s on %d train examples (%d held out) ...", MODEL_NAME, len(X_train), len(X_test))
    ft.fine_tune(model, tokenizer, X_train, label_ids_for(y_train, label2id), num_classes, epochs=EPOCHS, lr=LR)

    scored = score_batch(X_test, tokenizer, model)
    preds = ["bullish" if p.get("positive", 0.0) >= p.get("negative", 0.0) else "bearish" for p in scored]
    ft.evaluate_binary(preds, y_test, "FinBERT (fine-tuned)")

    # Refit fresh from the pretrained checkpoint on the FULL labeled set —
    # same pattern as sentiment_tfidf.py's final_pipeline refit — for the
    # model actually used to score every message.
    log.info("Refitting on full labeled set (n=%d) for final scoring model ...", len(bodies))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    ft.fine_tune(model, tokenizer, bodies, label_ids_for(labels, label2id), num_classes, epochs=EPOCHS, lr=LR)

    rows = to_rows(message_rows, tokenizer, model)
    stored = db.store_sentiment_scores(conn, rows)
    log.info("Scored and stored %d messages (method=%s).", stored, METHOD)

    conn.close()
    log.info("FinBERT fine-tuning done.")


if __name__ == "__main__":
    run()
