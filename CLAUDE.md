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

Phase 0 (scoping) and the early part of Phase 1 (data ingestion) are done. Currently
building out the ingestion layer. See README.md's phase checklist for the authoritative
up-to-date status — update it as phases complete.

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
ingestion/
├── config.py            # tickers, benchmark ticker, env-driven DB settings
├── db.py                 # Postgres connection + shared schema
├── stocktwits_ingest.py  # pulls StockTwits streams/symbol endpoint
├── price_ingest.py       # pulls daily OHLCV + fundamentals via yfinance
├── run_all.py            # runs ingestion sources against the DB
├── requirements.txt
├── .env.example          # copy to .env and fill in real values (never commit .env)
README.md                 # project plan, phase checklist, tech stack
phase0_proposal.md        # full thesis/scoping proposal (Phase 0 deliverable)
```

**Storage**: Postgres, three tables (see `db.py::init_db`):
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

**Environment**: managed via `.env` (loaded by `python-dotenv` in `config.py`). Requires
Postgres running locally (or via Docker).

## Setup Commands

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r ingestion/requirements.txt
cp ingestion/.env.example ingestion/.env   # then fill in real values
python ingestion/run_all.py
```

## Known Issues / Gotchas

- **StockTwits 403s**: the public `streams/symbol` endpoint requires a browser-like
  `User-Agent` header or it returns 403 (bot-protection, not a real auth wall — already
  handled in `stocktwits_ingest.py`'s `HEADERS`). StockTwits' official developer program
  is not currently accepting new registrations, so this public-endpoint approach is the
  practical path for now, not a real API key.
- **Look-ahead bias**: `stocktwits_ingest.py` stores timestamps as-is; it does NOT filter
  by market close. Any code that computes a daily sentiment score MUST filter to only
  use messages timestamped strictly before that trading day's market close.
  Getting this wrong invalidates the entire signal validation in Phase 3 — treat it as
  a hard constraint, not a nice-to-have.
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

- **Phase 1 (in progress)**: StockTwits done, Reddit removed (see Fixed Decisions),
  price/fundamentals ingestion for the ticker universe + QQQ done via `price_ingest.py`
- **Phase 2**: NLP sentiment extraction — Loughran-McDonald dictionary baseline →
  TF-IDF/logistic regression → FinBERT, compared against StockTwits' own sentiment label
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