# CLAUDE.md

This file gives Claude Code the context needed to work on this repo. Read this before
making changes — it captures decisions already made, why they were made, and known
gotchas so they aren't rediscovered or accidentally reversed.

## Project Overview

A Financial Sentiment & Alpha Prediction System — a portfolio project for a transition
into quant research/trading roles. Combines NLP, alternative data ingestion, statistical
validation, and ML to test whether social-media sentiment about tech stocks predicts
short-horizon forward returns.

**Thesis**: Retail sentiment expressed on StockTwits and Reddit about a fixed universe of
tech stocks contains information not yet reflected in price, predictive over a 1-5 day
horizon, decaying over that window. Full proposal: `phase0_proposal.md`.

This is explicitly a *research* project, not a production trading system. Correctness of
methodology (no look-ahead bias, proper validation splits, honest treatment of
limitations) matters more than model performance or code sophistication.

## Current Status

Phase 0 (scoping), Phase 1 (data ingestion), and Phase 2 (NLP sentiment extraction)
are done: all four zero-shot sentiment methods (LM dictionary, TF-IDF/logreg, FinBERT,
StockTwits-finetuned RoBERTa) plus daily/entity-level aggregation (see Architecture
below) are implemented and run against the full ingested dataset. A side experiment
fine-tuned FinBERT and the StockTwits RoBERTa model on our own labeled data to test
fine-tuning vs. zero-shot in-domain pretraining — see "Fine-tuned FinBERT and
StockTwits-RoBERTa" below — but that tangent is deliberately excluded from
`daily_sentiment`, so Phase 3 (signal validation, next) builds only on the four
zero-shot methods. See README.md's phase checklist for the authoritative up-to-date
status.

## Fixed Decisions (do not change without discussion)

- **Ticker universe** (locked, do not add/remove after seeing results):
  `AAPL, NVDA, GOOG, AMZN, MU, PLTR, RDDT, AI, TSLA, META, MSFT`
- **Sector**: Technology
- **Prediction horizons**: t+1 and t+5 trading days
- **Label**: return relative to QQQ (excess return), not raw return
- **RDDT constraint**: IPO'd March 2024 — any backtest window must start no earlier than
  that, or exclude RDDT from earlier periods
- **Data sources**: StockTwits only. Reddit ingestion was implemented and then removed
  (2026-07-06) due to recent API access limitations — see Roadmap note below. No Reddit
  rows exist in the database, so this was a code/config removal, not a data migration.

## Architecture

```
.
├── ingestion/                # Phase 1: data pipeline (a Python package — has __init__.py)
│   ├── config.py             # tickers, benchmark ticker, env-driven DB settings
│   ├── db.py                 # Postgres connection + shared schema
│   ├── stocktwits_ingest.py  # pulls StockTwits streams/symbol endpoint
│   ├── price_ingest.py       # pulls daily OHLCV + fundamentals via yfinance
│   ├── timestamp_alignment.py # maps a message timestamp to its aligned trading day
│   ├── deduplicate.py        # collapses same-account/body/day reposts before scoring
│   └── run_all.py            # runs ingestion sources against the DB
├── features/                 # Phase 2: NLP sentiment extraction (a Python package)
│   ├── sentiment_lm.py       # baseline: Loughran-McDonald dictionary scoring
│   ├── sentiment_tfidf.py    # TF-IDF + logistic regression, trained on StockTwits' own label
│   ├── sentiment_finbert.py  # FinBERT (ProsusAI/finbert), zero-shot
│   ├── sentiment_stocktwits_roberta.py # RoBERTa fine-tuned on StockTwits text, zero-shot on our data
│   ├── finetune_common.py    # side-experiment: shared PyTorch fine-tuning loop (see CLAUDE.md)
│   ├── sentiment_finbert_finetune.py   # side-experiment: fine-tunes FinBERT, excluded from daily_sentiment
│   ├── sentiment_stocktwits_roberta_finetune.py # side-experiment: fine-tunes RoBERTa, excluded from daily_sentiment
│   └── sentiment_daily.py    # aggregates sentiment_scores into daily/entity signal
├── models/                   # Phase 4: training scripts + saved model artifacts (not started)
├── backtest/                 # Phase 3/5: signal validation, portfolio simulation (not started)
├── notebooks/                # exploratory analysis only — no pipeline logic (not started)
├── requirements.txt
├── .env.example               # copy to .env and fill in real values (never commit .env)
├── docker-compose.yml         # local Postgres for development
├── README.md                  # project plan, phase checklist, tech stack
├── CLAUDE.md                  # this file
└── phase0_proposal.md         # full thesis/scoping proposal (Phase 0 deliverable)
```

Cross-package imports use absolute package paths (`from ingestion import config, db, ...` /
`from features import sentiment_lm`), not relative imports or `sys.path` hacks — this
means every script must be run as a module from the repo root (e.g. `python -m
ingestion.run_all`, `python -m features.sentiment_daily`), not as a bare script path.
`.env`/`.env.example` and `requirements.txt` live at the repo root rather than inside
`ingestion/`, since `docker-compose.yml` (also at root) auto-loads `.env` from its own
directory — nesting it inside `ingestion/` would silently break that.

**Storage**: Postgres, five tables (see `db.py::init_db`):
- `messages` — single table for text/sentiment data sources, distinguished by a
  `source` column (schema kept source-generic so another source could be added later
  without a migration).
  - Primary key is `(source, message_id)` — natural key from each platform, not a
    surrogate key. This makes re-running ingestion idempotent (`ON CONFLICT DO NOTHING`).
  - `source` is `'stocktwits'` (only active source; `'reddit'` rows would exist only if
    the removed Reddit ingestion had ever run against this DB — see Fixed Decisions
    above, it never did).
  - `sentiment_label` is populated from StockTwits' user-tagged bullish/bearish label —
    intended to be a free baseline to sanity-check Phase 2's own NLP-derived sentiment
    scores against.
  - `extra` is JSONB for source-specific fields (StockTwits: user_id) that don't need
    dedicated columns yet.
- `prices` — daily OHLCV bars, `PRIMARY KEY (ticker, date)`, one row per ticker/day,
  idempotent via `ON CONFLICT DO NOTHING`. Covers the fixed ticker universe plus
  `config.BENCHMARK_TICKER` (QQQ), needed for the excess-return label. RDDT rows
  naturally start at its March 2024 IPO — yfinance just returns nothing earlier.
- `fundamentals` — one row per ticker (`PRIMARY KEY ticker`), a point-in-time
  snapshot (sector/industry/market cap), not a time series — each ingestion run
  overwrites the previous snapshot via `ON CONFLICT ... DO UPDATE` (`db.py::upsert_fundamentals`).
  Expect `NULL` sector/industry/market_cap for ETFs like QQQ — yfinance's `.info`
  doesn't populate those fields for non-equities, this is not a bug.
- `sentiment_scores` — one row per (method, message), `PRIMARY KEY (method, source,
  message_id)`. `method` discriminates which Phase 2 technique produced the row —
  the four forward-going methods (`'lm_dict'`, `'tfidf_logreg'`, `'finbert'`,
  `'stocktwits_roberta'`) plus two side-experiment methods (`'finbert_finetuned'`,
  `'stocktwits_roberta_finetuned'`, see "Fine-tuned FinBERT and StockTwits-RoBERTa"
  below) kept here as a research record but deliberately excluded from
  `daily_sentiment` — same discriminator pattern as `messages.source`, so all six
  coexist without a schema change.
  `score` is normalized to roughly [-1, 1] for cross-method comparability; `label` is
  bucketed to `bullish`/`bearish`/`neutral` so it's directly comparable to
  `messages.sentiment_label`; method-specific detail (LM's raw pos/neg counts and
  subjectivity, FinBERT's class probabilities, etc.) goes in `extra` JSONB rather than
  dedicated columns. Upserts via `ON CONFLICT ... DO UPDATE` (not `DO NOTHING`) since
  re-scoring the same message with the same method should overwrite, not no-op.
- `daily_sentiment` — one row per (ticker, trading_day, method), `PRIMARY KEY (ticker,
  trading_day, method)`, built by `sentiment_daily.py` from `sentiment_scores` grouped
  by aligned trading day (not calendar day — same look-ahead-safe grouping as
  `deduplicate.py`). Stores `mean_score`, `message_count`, and the bullish/bearish/
  neutral mix; `mean_score` is `NULL` when `message_count < config.MIN_DAILY_MESSAGES`
  (the Phase 0 volume floor) — treated as missing signal, not a noisy one —
  `above_volume_floor` records why. Only the four forward-going methods are
  aggregated here (not blended into one score, so Phase 3 can evaluate each
  method's IC separately) — the two fine-tuned side-experiment methods are
  intentionally excluded, see "Fine-tuned FinBERT and StockTwits-RoBERTa" below.
  Upserts via `ON CONFLICT ... DO UPDATE` for the same re-scoring-overwrites reason as
  `sentiment_scores`.

**Environment**: managed via `.env` (loaded by `python-dotenv` in `config.py`). Requires
Postgres running locally (or via Docker).

**Timestamp alignment** (`timestamp_alignment.py`): the no-look-ahead enforcement
mechanism referenced throughout this file. `load_trading_days(conn)` reads the sorted
list of dates the benchmark ticker (QQQ) has a `prices` row for — this is a real NYSE
trading calendar "for free" (weekends and holidays like July 4th are simply absent from
QQQ's price history), so there's no separate market-calendar dependency.
`align_to_trading_day(timestamp_utc, trading_days)` then maps any tz-aware UTC
timestamp to the earliest trading day whose 4pm ET close strictly follows it — a
message posted before that day's close aligns to that day; at/after close, or on a
non-trading day, it rolls forward to the next trading day. Returns `None` if the
timestamp is beyond the last currently-ingested trading day (nothing to align to yet).
Any Phase 2/3 code building a daily sentiment score MUST route timestamps through this
function rather than reimplementing the close-time logic — it's the single source of
truth for the look-ahead constraint.

**Deduplication** (`deduplicate.py`): investigated directly in the ingested StockTwits
data — some accounts post the exact same body text more than once on the same aligned
trading day (e.g. sharing the same article link twice, or a recurring scheduled repost
that lands on the same trading day across a holiday weekend). Distinct authors
independently posting a short identical body (e.g. just `"$PLTR"`) are common and are
NOT duplicates. `deduplicate_messages` collapses rows sharing the same `(ticker,
author, body, aligned_trading_day)` — note the grouping key is the *aligned* trading
day from `timestamp_alignment.py`, not the calendar day, so a repost spanning a
holiday/weekend that resolves to the same trading day still collapses. Verified against
real data: 1,562 → 1,535 messages, all 27 removed rows confirmed genuine same-account
reposts. Does not mutate stored data — raw `messages` rows are untouched; Phase 2
sentiment scoring should fetch through `deduplicated_messages_for_ticker` rather than
querying `messages` directly.

**LM dictionary sentiment scoring** (`sentiment_lm.py`): Phase 2's baseline method,
using `pysentiment2`'s bundled Loughran-McDonald word lists (built from formal 10-K/10-Q
filing language — see https://www3.nd.edu/~mcdonald/Word_Lists.html). Scores each
deduplicated message independently (no cross-message aggregation yet — that's a later
Phase 2 step) and writes to `sentiment_scores` under `method='lm_dict'`. Verified against
real data: mostly returns `neutral` on short informal StockTwits posts (1,048/1,535
messages) and shows weak agreement with StockTwits' own bullish/bearish label — an
**expected baseline weakness**, not a bug: LM is tuned for formal filing language, so
informal phrasing (e.g. "beat estimates") often doesn't register while formal words like
"risk"/"litigation" score negative regardless of context. This is the documented
motivation for the next two Phase 2 methods (TF-IDF/logreg, FinBERT), not something to
tune away in the dictionary approach itself.

**TF-IDF/logistic regression sentiment scoring** (`sentiment_tfidf.py`): Phase 2's
second method, a supervised classifier trained on StockTwits' own bullish/bearish tag
as ground truth (`messages.sentiment_label`) — the only labeled data available at this
stage. Only 616 of 1,535 deduplicated messages carry a label, and it's imbalanced
(~4:1 bullish:bearish); `LogisticRegression(class_weight="balanced")` compensates for
that. Because StockTwits never tags "Neutral", this model is a **binary** classifier —
every message is forced into bullish or bearish, unlike LM which can return neutral.
That's a real behavioral difference to keep in mind when comparing `sentiment_scores`
across methods, not just an accuracy difference.
Held-out evaluation (stratified 80/20 split, train=492/test=124 — small, so treat
metrics as indicative not precise): 78% accuracy, 0.85 F1 bullish / 0.57 F1 bearish
(minority class is harder, as expected). The LM baseline gets only 16% accuracy on the
identical test split, mostly because it defaults to neutral (94/124 messages) rather
than committing to bullish/bearish — a fair like-for-like comparison since ground
truth is always one of the two classes.
Note on validation: this is text classification (message → label), not forward-return
prediction, so CLAUDE.md's "walk-forward CV only" constraint (which applies to Phase 4)
does not apply here — a random stratified split carries no temporal look-ahead risk.

**FinBERT sentiment scoring** (`sentiment_finbert.py`): Phase 2's third method, using
`ProsusAI/finbert` (BERT fine-tuned on financial news/analyst text for 3-class
positive/negative/neutral sentiment) via HuggingFace `transformers`. Run **zero-shot** —
not fine-tuned on our StockTwits data — so there's no train/test split to construct;
evaluation against StockTwits' own label uses all 616 labeled messages directly, since
nothing here was fit on them. Result: 14.1% accuracy, again mostly defaulting to neutral
(493/616) — FinBERT's formal financial-news training doesn't transfer well to short
informal social posts either, a different root cause than LM's but a similar symptom.
Kept in the codebase and in `sentiment_scores` even after `sentiment_stocktwits_roberta.py`
(below) beat it decisively — the weak result is itself the useful finding (general
finance-text pretraining doesn't transfer to social media), not something to delete
once a better method exists. Uses CPU-only PyTorch (no GPU available/needed for this
dataset size); `torch`/`transformers` added to `requirements.txt`.

**StockTwits-finetuned RoBERTa sentiment scoring** (`sentiment_stocktwits_roberta.py`):
Phase 2's fourth method, added specifically to fix FinBERT's domain mismatch — uses
`zhayunduo/roberta-base-stocktwits-finetuned`, a RoBERTa-base model fine-tuned by its
author directly on StockTwits posts (informal social-media finance text) rather than
formal news/analyst text. Like FinBERT, run **zero-shot with respect to our data** (not
fine-tuned on this project's messages), so evaluation again uses all 616 StockTwits-labeled
messages directly with no train/test split. The model's own label set (confirmed via
`AutoConfig.id2label`) is binary — `Negative`/`Positive`, no neutral class — so like
`sentiment_tfidf.py` it forces every message into bullish/bearish. Result: **87.5%
accuracy**, clearly the best of all four methods — confirms that in-domain (StockTwits)
pretraining matters far more than general finance-text pretraining for this short,
informal data, more so even than `sentiment_tfidf.py`'s in-domain-but-small-labeled-set
approach (78%).
**Cross-method takeaway (four zero-shot methods)**: StockTwits-RoBERTa (87.5%) >
TF-IDF/logreg (78%, trained on our own labels) >> LM dictionary (16%) > FinBERT (14%)
on StockTwits' own label. The consistent pattern: StockTwits-domain text — whether via
someone else's fine-tuning or our own small labeled set — beats general-purpose
finance-text tools applied out-of-the-box. Added alongside FinBERT rather than
replacing it (explicit user decision) so the honest negative result stays on record.

**Fine-tuned FinBERT and StockTwits-RoBERTa** (`sentiment_finbert_finetune.py`,
`sentiment_stocktwits_roberta_finetune.py`, shared training loop in
`finetune_common.py`): a **side experiment, not part of the forward-going Phase 2
pipeline** — see the note at the end of this section for why it's excluded from
`daily_sentiment`/Phase 3. Both `sentiment_finbert.py` and
`sentiment_stocktwits_roberta.py` above are run zero-shot; these two scripts instead
fine-tune the same base checkpoints on our own labeled StockTwits data via a plain
PyTorch training loop (no HuggingFace `Trainer`/`accelerate` — unnecessary for a
few hundred examples on CPU). Both reuse `sentiment_tfidf.py`'s exact
`load_all_deduplicated`/`labeled_examples` loading and stratified 80/20
`train_test_split` (same `TEST_SIZE`/`RANDOM_STATE`) so held-out accuracy is
comparable to TF-IDF/LogReg and to each model's own zero-shot result on identical
data, not just an identical metric. A class-weighted `CrossEntropyLoss` (inverse
label frequency, same "balanced" idea as `sentiment_tfidf.py`'s
`class_weight="balanced"`) compensates for the ~4:1 bullish:bearish imbalance.
Following `sentiment_tfidf.py`'s pattern, each model is fine-tuned once for the
held-out evaluation, then **refit from the pretrained checkpoint** (not
continued-from-the-eval-run) on the full labeled set for the model that actually
scores every message — the held-out split exists only to produce an honest metric,
not to withhold data from the production scorer.
FinBERT's pretrained head is kept 3-class (positive/negative/neutral) rather than
resized to 2 classes: our labels are only ever bullish/bearish, so training targets
map onto FinBERT's existing positive/negative indices and the neutral index is
never a direct target (though softmax normalization still shifts it during
training) — this keeps its output label space identical to the zero-shot method's,
so `mean_score`/`label` stay comparable in `sentiment_scores`. The RoBERTa model
needs no such trick — its pretrained head is already the binary Negative/Positive
we need.
Results (train=612, test=153 as of this run — the labeled set has grown past the
616 messages used in the zero-shot evaluations above, since ingestion has kept
running since then): **FinBERT fine-tuned 56.2% accuracy** — up sharply from its
14% zero-shot result, so fine-tuning clearly helps close the domain gap, but formal
financial-news pretraining still leaves it well behind. **StockTwits-RoBERTa
fine-tuned 86.3% accuracy** — statistically indistinguishable from its own 87.5%
zero-shot result; it was already fine-tuned on StockTwits-domain text by its
original author, so further fine-tuning on our much smaller labeled set moves the
needle little. Reinforces the phase's core finding: **in-domain pretraining**, not
fine-tuning per se, is what separates these methods — fine-tuning narrows an
out-of-domain model's gap but doesn't let it catch up to one that was in-domain
from the start.
**Scope note**: this was a one-off test of fine-tuning as a technique, not an
extension to the production method lineup — `finbert_finetuned` and
`stocktwits_roberta_finetuned` are stored in `sentiment_scores` as a research
record (same "keep the honest result" precedent as FinBERT's zero-shot score
above) but are deliberately left out of `sentiment_daily.py`'s `METHODS` list, so
they never reach `daily_sentiment` and Phase 3+ builds only on the four zero-shot
methods.

**Daily/entity-level sentiment aggregation** (`sentiment_daily.py`): Phase 2's last
step, turning the per-message `sentiment_scores` rows (one per method) into a daily
signal per `(ticker, trading_day, method)` in `daily_sentiment`. Groups by the
*aligned* trading day from `timestamp_alignment.py`, so the look-ahead guarantee
established at ingestion/dedup time carries through unchanged. Applies the Phase 0
volume floor (`config.MIN_DAILY_MESSAGES = 3`): below that count, `mean_score` is
stored `NULL` rather than an average over too few noisy posts, while `message_count`/
`above_volume_floor` are still recorded so it's visible *why* a day is missing rather
than silently absent. Verified against real data: the four forward-going methods
cover 26 distinct (ticker, trading_day) combinations, with message counts per day
ranging 6–147, so none currently fall below the floor; it will start binding once
ingestion has run long enough to accumulate sparser days. This completes
Phase 2 — Phase 3 (signal validation) is next.

## Setup Commands

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # then fill in real values
python -m ingestion.run_all
```

## Known Issues / Gotchas

- **StockTwits 403s**: the public `streams/symbol` endpoint requires a browser-like
  `User-Agent` header or it returns 403 (bot-protection, not a real auth wall — already
  handled in `stocktwits_ingest.py`'s `HEADERS`). StockTwits' official developer program
  is not currently accepting new registrations, so this public-endpoint approach is the
  practical path for now, not a real API key.
- **Look-ahead bias**: `stocktwits_ingest.py` stores timestamps as-is; it does NOT filter
  by market close. Any code that computes a daily sentiment score MUST run timestamps
  through `timestamp_alignment.py::align_to_trading_day` rather than reimplementing the
  close-time logic. Getting this wrong invalidates the entire signal validation in
  Phase 3 — treat it as a hard constraint, not a nice-to-have.
- **`AI`/`MU` ticker ambiguity**: still relevant for StockTwits (and for Phase 2 NLP
  generally) since `AI` is also a common English word — this was previously handled in
  `reddit_ingest.py`'s filtering (now removed); StockTwits data is pre-tagged by ticker
  so it's less exposed to this, but Phase 2 text processing should stay aware of it.
- **Rate limits**: StockTwits ingestion sleeps between requests (20s) to stay under
  historical unauthenticated rate limits. Don't remove this without a reason — the
  script currently gets 403/429s if it moves too fast.
- **yfinance `.info` reliability**: `price_ingest.py`'s fundamentals fetch is
  best-effort — Yahoo's `.info` scrape is occasionally throttled or incomplete for a
  given ticker. `fetch_fundamentals` catches exceptions and logs a warning rather than
  failing the whole run; a `None` sector/industry/market_cap for a real equity may just
  mean the scrape came back empty that run, not a code bug.

## Coding Conventions

- Config (tickers, credentials) lives in `config.py` — don't hardcode tickers or
  credentials in individual scripts.
- New *text/sentiment* data sources should write into the shared `messages` table via
  `db.py`'s `store_messages`, using the same `(source, message_id, ticker,
  created_at_utc, body, author, sentiment_label, score, extra, ingested_at_utc)` row
  shape, rather than creating source-specific tables. Structurally different data
  (e.g. numeric time series like OHLCV) gets its own table instead — see `prices` /
  `fundamentals` — forcing it into `messages`' text-message shape would be worse than
  a second table.
- Prefer explicit, named functions per ingestion step (fetch → filter → to_rows →
  store) over monolithic scripts, matching the existing structure in
  `stocktwits_ingest.py`.
- No secrets in code — everything credential-related goes through `.env`.

## Roadmap (see README.md for full detail)

- **Phase 1 (complete)**: StockTwits done, Reddit removed (see Fixed Decisions),
  price/fundamentals ingestion for the ticker universe + QQQ done via `price_ingest.py`,
  timestamp alignment done via `timestamp_alignment.py`, deduplication done via
  `deduplicate.py`, storage in Postgres done from the start.
- **Phase 2 (complete)**: NLP sentiment extraction — Loughran-McDonald dictionary
  baseline done (`sentiment_lm.py`, 16% accuracy vs. StockTwits' label) → TF-IDF/logistic
  regression done (`sentiment_tfidf.py`, 78% held-out accuracy) → FinBERT done
  (`sentiment_finbert.py`, 14% accuracy, zero-shot) → StockTwits-finetuned RoBERTa done
  (`sentiment_stocktwits_roberta.py`, 87.5% accuracy, zero-shot on our data — clear
  winner, added alongside FinBERT rather than replacing it) → daily/entity-level
  aggregation done (`sentiment_daily.py`, `daily_sentiment` table, volume floor applied).
  Side experiment (not part of the forward pipeline): fine-tuned versions of both
  transformers on our own labeled data (`sentiment_finbert_finetune.py`, 56.2%
  accuracy; `sentiment_stocktwits_roberta_finetune.py`, 86.3% accuracy — confirms
  in-domain pretraining matters more than fine-tuning per se), deliberately excluded
  from `daily_sentiment` so Phase 3+ builds only on the four zero-shot methods.
- **Phase 3**: signal validation — IC/rank-IC across t+1 and t+5, factor neutralization,
  explicit treatment of multiple-testing risk
- **Phase 4**: ML model combining sentiment + traditional factors, walk-forward CV only
- **Phase 5**: backtesting with realistic costs/slippage
- **Phase 6**: research write-up (`research_note.md`)

## When Making Changes

- If a change would affect a "Fixed Decision" above (universe, horizons, label
  definition), flag it explicitly rather than changing it silently — these were locked
  deliberately to avoid results-driven scope changes.
- If you add a new dependency, update `requirements.txt`.
- If you change the DB schema, update the `init_db` function in `db.py` and note the
  change here.