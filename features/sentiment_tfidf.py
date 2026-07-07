"""
Phase 2, second method: classical ML sentiment via TF-IDF + logistic regression.

Ground truth for training comes from StockTwits' own user-tagged bullish/bearish
label (`messages.sentiment_label`) — the same free label used to sanity-check
the Loughran-McDonald baseline in `sentiment_lm.py`. Only ~636 of 1,535
deduplicated messages carry a label (the rest are untagged posts), and the
labeled set is imbalanced (~4:1 bullish:bearish), so this is a small, noisy
training set — treated as such in reporting, not glossed over.

Because StockTwits only ever tags Bullish/Bearish (never "Neutral"), this
model is a binary classifier: every message gets forced into bullish or
bearish. That's a real difference from the LM dictionary baseline, which can
return neutral. Worth remembering when comparing the two methods' output
distributions later.

Note on validation: this trains a text classifier (message -> sentiment
label), not a forward-return predictor. CLAUDE.md's "walk-forward CV only"
constraint applies to the return-prediction model in Phase 4; a random
stratified train/test split is methodologically fine here since there's no
temporal leakage risk in classifying already-written text.
"""

import logging
from datetime import datetime, timezone

import psycopg2.extras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ingestion import config, db, deduplicate, timestamp_alignment as ta
from features import sentiment_lm

METHOD = "tfidf_logreg"
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_all_deduplicated(conn, tickers: list[str], trading_days: list) -> list[tuple]:
    """Pool deduplicated messages across all tickers — training data is
    pooled rather than per-ticker since the labeled set (636 messages) is
    already small, and sentiment vocabulary is largely shared across this
    tech-sector universe."""
    all_rows = []
    for ticker in tickers:
        all_rows.extend(deduplicate.deduplicated_messages_for_ticker(conn, ticker, trading_days))
    return all_rows


def labeled_examples(message_rows: list[tuple]) -> tuple[list[str], list[str]]:
    """Extract (body, label) pairs where StockTwits' own tag exists,
    normalized to lowercase to match sentiment_lm's label convention."""
    bodies, labels = [], []
    for row in message_rows:
        body, label = row[4], row[6]
        if label in ("Bullish", "Bearish"):
            bodies.append(body)
            labels.append(label.lower())
    return bodies, labels


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words="english",
        )),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def evaluate(bodies: list[str], labels: list[str]) -> None:
    """Held-out evaluation, logged for honest reporting (n=%d test examples
    is small — treat metrics as indicative, not precise)."""
    X_train, X_test, y_train, y_test = train_test_split(
        bodies, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    log.info("=== TF-IDF/LogReg held-out evaluation (train=%d, test=%d) ===", len(X_train), len(X_test))
    log.info("\n%s", classification_report(y_test, y_pred, zero_division=0))
    log.info("Confusion matrix (rows=true, cols=pred, order=%s):\n%s",
              sorted(set(labels)), confusion_matrix(y_test, y_pred, labels=sorted(set(labels))))

    # Fair comparison against the LM dictionary baseline on the same test
    # split — LM can output 'neutral', which is scored as a miss here since
    # ground truth is always bullish/bearish.
    lm = sentiment_lm.get_lm()
    lm_pred = []
    for body in X_test:
        scored = sentiment_lm.score_body(body, lm)
        lm_pred.append(sentiment_lm.score_to_label(scored["polarity"], scored["subjectivity"]))
    lm_correct = sum(1 for p, t in zip(lm_pred, y_test) if p == t)
    lm_neutral = sum(1 for p in lm_pred if p == "neutral")
    log.info(
        "LM baseline on same test set: accuracy=%.3f (neutral predictions=%d/%d, scored as misses)",
        lm_correct / len(y_test), lm_neutral, len(y_test),
    )
    tfidf_accuracy = sum(1 for p, t in zip(y_pred, y_test) if p == t) / len(y_test)
    log.info("TF-IDF/LogReg accuracy=%.3f", tfidf_accuracy)


def to_rows(message_rows: list[tuple], pipeline: Pipeline, n_train: int) -> list[tuple]:
    now = datetime.now(timezone.utc)
    bodies = [row[4] for row in message_rows]
    proba = pipeline.predict_proba(bodies)
    classes = list(pipeline.classes_)  # e.g. ['bearish', 'bullish']
    bullish_idx = classes.index("bullish")

    rows = []
    for row, p in zip(message_rows, proba):
        source, message_id, ticker = row[0], row[1], row[2]
        p_bullish = float(p[bullish_idx])
        label = "bullish" if p_bullish >= 0.5 else "bearish"
        score = 2 * p_bullish - 1  # map to [-1, 1], matching sentiment_lm's polarity range
        rows.append((
            METHOD,
            source,
            message_id,
            ticker,
            score,
            label,
            psycopg2.extras.Json({"proba_bullish": p_bullish, "trained_on_labeled_n": n_train}),
            now,
        ))
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
        log.warning("Too few labeled examples to train/evaluate a classifier — skipping.")
        conn.close()
        return

    evaluate(bodies, labels)

    # Refit on the full labeled set for the model used to score every message.
    final_pipeline = build_pipeline()
    final_pipeline.fit(bodies, labels)

    rows = to_rows(message_rows, final_pipeline, n_train=len(bodies))
    stored = db.store_sentiment_scores(conn, rows)
    log.info("Scored and stored %d messages (method=%s).", stored, METHOD)

    conn.close()
    log.info("TF-IDF/LogReg sentiment scoring done.")


if __name__ == "__main__":
    run()
