# Financial Sentiment & Alpha Prediction System

A research project combining NLP, alternative data, and statistical modeling to test whether
text-derived sentiment signals carry predictive information about forward asset returns.

Built as a portfolio project for a transition into quantitative research/trading roles.

## Project Thesis

> Retail sentiment expressed on StockTwits about a fixed universe of large-cap and
> high-retail-interest tech stocks contains information not yet reflected in price, and
> this signal — if it exists — decays within 1-5 trading days.

A clear, falsifiable thesis is the foundation of the whole project; everything downstream
exists to test it.

## Phase 0 Decisions (locked)

These decisions are fixed before any data is pulled, to avoid biasing the universe or
methodology after seeing results.

- **Sector**: Technology
- **Universe (final, locked)**: `AAPL, NVDA, GOOG, AMZN, MU, PLTR, RDDT, AI, TSLA, META, MSFT`
  — 11 tickers mixing mega-cap tech (AAPL, NVDA, GOOG, AMZN, META, MSFT, TSLA) with
  higher-retail-chatter / thinner-coverage names (MU, PLTR, RDDT, AI). This list is now
  fixed for the duration of the project and will not be revised after seeing results.
  Note: RDDT IPO'd in March 2024 — backtest period must either start no earlier than
  that, or RDDT must be excluded from any test window predating its listing.
- **Data source**: StockTwits only. Reddit ingestion was attempted and then removed
  due to recent API access limitations; may be revisited if that changes.
- **Horizons tested**: next-day (t+1) and t+5 return, to support a signal decay analysis
  in Phase 3.
- **Label**: return relative to a tech sector ETF (QQQ), not raw return — a more
  defensible measure of "alpha" than absolute price movement.
- **Known assumptions/limitations accepted**:
  - Survivorship: confirm all tickers existed for the full backtest period.
  - Look-ahead: a post only counts toward a day's sentiment score if timestamped before
    that day's market close.
  - Volume floor: below a minimum post-count threshold per ticker/day, treat sentiment
    as missing rather than a noisy signal.
  - Social sentiment is noisier than news (sarcasm, hype, possible bot activity) —
    validated in Phase 2 against StockTwits' own user-tagged bullish/bearish label as a
    sanity check.

## Goals

- Build an end-to-end pipeline from raw alternative data to a validated trading signal
- Apply and compare NLP techniques ranging from lexicon-based to transformer-based sentiment extraction
- Rigorously test signal quality using standard quant research methods (IC, decay analysis, factor neutralization)
- Train and validate an ML model that combines sentiment with traditional factors
- Backtest the resulting signal with realistic assumptions about costs and turnover
- Document the process as a research note, not just a codebase

## Repo Structure

```
.
├── data/               # Raw and processed data (gitignored if large; document source instead)
├── ingestion/          # Scripts to pull news/filings/social data + price data
├── features/           # Sentiment extraction and feature engineering
├── models/             # Training scripts and saved model artifacts
├── backtest/           # Signal validation, IC analysis, portfolio simulation
├── notebooks/           # Exploratory analysis only — no pipeline logic lives here
├── research_note.md    # Write-up: hypothesis, methodology, results, limitations
├── requirements.txt
└── README.md
```

## Project Plan

### Phase 0 — Scope & Thesis (~1 week)
- [x] Define universe — tech sector, ~10-15 fixed tickers (see Phase 0 Decisions above)
- [x] Define prediction horizon — t+1 and t+5, vs. QQQ
- [x] Choose primary data source — StockTwits (Reddit dropped, see Phase 1 note)
- [x] Finalize exact ticker list and paste into Phase 0 Decisions
- [x] Write one-page thesis proposal doc — see [`phase0_proposal.md`](./phase0_proposal.md)

### Phase 1 — Data Pipeline (~2-3 weeks)
- [x] Ingest StockTwits data (ticker-tagged posts + built-in bullish/bearish user label) for the fixed ticker list
- [x] ~~Reddit ingestion~~ — implemented, then removed due to recent Reddit API access
      limitations; StockTwits is the sole data source going forward
- [x] Ingest price/fundamental data (yfinance) — daily OHLCV + sector/industry/market-cap
      snapshot for the fixed ticker list plus the QQQ benchmark
- [x] Timestamp alignment — `timestamp_alignment.py` maps a message timestamp to the
      trading day whose return it may predict (strictly before that day's 4pm ET
      close, else rolls to the next trading day), using the `prices` table's QQQ
      history as the real NYSE trading calendar (no look-ahead)
- [x] Deduplicate republished/syndicated stories — `deduplicate.py` collapses posts
      sharing the same (ticker, author, body, aligned trading day), so an account
      reposting identical text doesn't inflate that day's sentiment; verified against
      real ingested data (1,562 → 1,535 messages, all genuine same-account reposts)
- [x] Store in structured DB — Postgres (via Docker), used from the start rather than
      SQLite-then-migrate

### Phase 2 — NLP Sentiment Extraction (~2-3 weeks)
- [x] Baseline: Loughran-McDonald finance-specific sentiment dictionary —
      `sentiment_lm.py` scores every deduplicated message via `pysentiment2`'s
      bundled LM word lists, storing polarity/subjectivity/label in the new
      `sentiment_scores` table. Sanity-checked against StockTwits' own
      bullish/bearish label: LM mostly returns neutral on short informal
      posts (1,048/1,535) and shows weak agreement with the crowd label —
      an expected baseline weakness (LM is tuned for formal filing
      language), not a bug, and the motivation for the next two methods.
- [x] Classical ML: TF-IDF + logistic regression — `sentiment_tfidf.py` trains on
      StockTwits' own bullish/bearish tag as ground truth (616 of 1,535 deduplicated
      messages are labeled, ~4:1 bullish:bearish). Held-out evaluation (train=492,
      test=124): 78% accuracy, 0.85 F1 on bullish / 0.57 F1 on bearish (minority class
      is harder, as expected with the imbalance). Clearly beats the LM baseline on the
      same test set (16% accuracy — mostly because LM defaults to neutral). Note:
      unlike LM, this is a binary classifier (StockTwits never tags "Neutral"), so it
      forces every message into bullish/bearish — a real behavioral difference between
      the two methods, not just an accuracy gap.
- [x] Transformer: FinBERT for sentence/document-level sentiment — `sentiment_finbert.py`
      runs `ProsusAI/finbert` zero-shot (not fine-tuned on our data) over every
      deduplicated message. Against StockTwits' own label (n=616, no train/test split
      needed since nothing was fit on this data): 14.1% accuracy, again mostly
      defaulting to neutral (493/616) — FinBERT is tuned on formal analyst/news text,
      so short informal social posts trip it up in a similar way to the LM dictionary,
      just for a different underlying reason.
- [x] Transformer (in-domain): `zhayunduo/roberta-base-stocktwits-finetuned` — added
      alongside FinBERT (not replacing it) via `sentiment_stocktwits_roberta.py`, since
      this RoBERTa model is fine-tuned directly on StockTwits posts rather than formal
      financial news. Against StockTwits' own label (n=616, zero-shot on *our* data, no
      train/test split needed since nothing was fit on this project's messages): 87.5%
      accuracy — clearly the best of all four methods, confirming that in-domain
      (StockTwits-trained) pretraining matters far more than general finance-text
      pretraining for this informal, short-form data. Binary like TF-IDF/LogReg (the
      model's own label set is Negative/Positive, no neutral class).
- [x] Aggregate document-level scores into daily/entity-level sentiment signal —
      `sentiment_daily.py` groups each method's `sentiment_scores` rows by
      (ticker, aligned trading day) into the new `daily_sentiment` table:
      mean score, message count, and bullish/bearish/neutral mix, computed
      independently per method (not blended) so Phase 3 can evaluate each
      method's IC separately. Below `config.MIN_DAILY_MESSAGES` (the Phase 0
      volume floor), `mean_score` is stored as NULL rather than a noisy
      average — message_count is still recorded so it's visible *why* a
      day is missing. Current ingested data spans only 26 ticker-days (a
      narrow recent window, not a full backtest history yet), none below
      the floor of 3.
- [x] Compare model performance (F1, latency) against baseline — see the four methods'
      results above: StockTwits-RoBERTa (87.5%) > TF-IDF/logreg (78%, trained on our
      labels) >> LM dictionary (16%) > FinBERT (14%) on StockTwits' own label. The
      pattern is consistent across all four: in-domain data (StockTwits text, whether
      via our own labels or someone else's fine-tuning) beats general-purpose
      finance-text tools applied out-of-the-box — an honest, if unglamorous, finding
      worth carrying into Phase 3.
**Side experiment (tangent — not carried into Phase 3+)**: tested whether
fine-tuning FinBERT and the StockTwits RoBERTa model on our own labeled
StockTwits data (both are run zero-shot above) beats zero-shot in-domain
pretraining. `sentiment_finbert_finetune.py` / `sentiment_stocktwits_roberta_finetune.py`
(shared PyTorch training loop in `features/finetune_common.py`) fine-tune both
models, reusing `sentiment_tfidf.py`'s exact stratified 80/20 split
(train=612, test=153 on the labeled set as of this run) for direct
comparability, with a class-weighted loss for the ~4:1 bullish:bearish
imbalance. Held-out results: **FinBERT fine-tuned 56.2% accuracy** (up
sharply from its 14% zero-shot baseline — fine-tuning clearly helps, but
domain-general pretraining still leaves it far behind), **StockTwits-RoBERTa
fine-tuned 86.3% accuracy** (statistically indistinguishable from its own
87.5% zero-shot result — it was already fine-tuned on StockTwits-domain text
by its author, so further fine-tuning on our much smaller labeled set changes
little). Confirms this phase's core finding — in-domain *pretraining* is
what matters here, not fine-tuning per se — but this was a one-off test of
that idea, not an extension to the production pipeline: its two methods
(`finbert_finetuned`, `stocktwits_roberta_finetuned`) are stored in
`sentiment_scores` as a research record but are **deliberately excluded**
from `sentiment_daily.py`'s aggregation, so they never reach `daily_sentiment`
and Phase 3+ builds only on the four zero-shot methods above.

### Phase 3 — Signal Construction & Statistical Validation (~2 weeks)
- [ ] Construct forward return labels (raw and market/sector-neutral excess returns)
- [ ] Compute Information Coefficient (IC) and rank-IC across multiple lags
- [ ] Run t-stats on long-short decile spread returns
- [ ] Address multiple-testing risk explicitly
- [ ] Neutralize against known factors (beta, size, momentum)

### Phase 4 — ML Prediction Model (~2-3 weeks)
- [ ] Combine sentiment features with traditional factors (momentum, volume, volatility, fundamentals)
- [ ] Baseline: Ridge/Lasso/ElasticNet
- [ ] Extend: LightGBM/XGBoost
- [ ] Optional: sequence models (LSTM/Transformer) if using sentiment time series —
      modeling the sentiment→return relationship over time (rather than as
      cross-sectional features) is a genuine training task, unimplemented so far;
      a natural fit for PyTorch, same framework as the Phase 2 fine-tuning above
- [ ] Use walk-forward / expanding-window CV — never random K-fold

### Phase 5 — Backtesting & Portfolio Simulation (~1-2 weeks)
- [ ] Build or adapt a backtester (vectorbt / bt / zipline)
- [ ] Include transaction costs, slippage, position limits
- [ ] Report Sharpe, cumulative return, max drawdown, turnover
- [ ] Evaluate performance across regimes (e.g. 2020 crash, 2022 rate hikes)

### Phase 6 — Write-up & Packaging
- [ ] Write research note: hypothesis, data, methodology, results, limitations
- [ ] Document known biases (survivorship, look-ahead, sample size)
- [ ] Clean repo structure and commit history
- [ ] Optional: short blog post/PDF summary for resume/LinkedIn

## Tech Stack

- **Language**: Python
- **Data/ML**: pandas, numpy, scikit-learn, LightGBM
- **Deep Learning**: PyTorch (already used for zero-shot FinBERT/RoBERTa inference;
  Phase 2/4 fine-tuning and sequence-model work will build on it rather than adding
  TensorFlow as a second framework)
- **NLP**: HuggingFace `transformers` (FinBERT), NLTK/spaCy
- **Statistics**: statsmodels
- **Backtesting**: vectorbt / bt / custom
- **Storage**: SQLite / PostgreSQL

## Key Design Principles

1. **No look-ahead bias** — every feature must only use information available at decision time.
2. **Walk-forward validation only** — random cross-validation on time series data is invalid here.
3. **Neutralize known factors** — a "sentiment alpha" that's really just momentum in disguise isn't a real finding.
4. **Be honest about limitations** — this matters more to reviewers/interviewers than an inflated Sharpe ratio.

## Status

🚧 In progress — see checkboxes above for current phase.

## License

MIT (or your preference)
