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
- **Universe**: ~10-15 fixed tickers, mixing mega-cap tech (e.g. AAPL, MSFT, NVDA, GOOGL,
  AMZN, META, TSLA) with higher-retail-chatter mid-caps (e.g. PLTR, SOFI, AI). Exact list
  TBD — write it here once finalized and do not change it after seeing results.
- **Data source**: StockTwits (Phase 1). Reddit is a stretch goal for later, once the
  pipeline works end-to-end.
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
- [x] Choose primary data source — StockTwits (Reddit as stretch goal)
- [ ] Finalize exact ticker list and paste into Phase 0 Decisions
- [ ] Write one-page thesis proposal doc (expand thesis above into full proposal)

### Phase 1 — Data Pipeline (~2-3 weeks)
- [ ] Ingest StockTwits data (ticker-tagged posts + built-in bullish/bearish user label) for the fixed ticker list
- [ ] Stretch goal: Reddit ingestion (PRAW/Pushshift) with regex-based ticker extraction, once StockTwits pipeline works end-to-end
- [ ] Ingest price/fundamental data (yfinance / Alpha Vantage / Polygon.io)
- [ ] Timestamp alignment — ensure sentiment timestamps strictly precede return timestamps (no look-ahead)
- [ ] Deduplicate republished/syndicated stories
- [ ] Store in structured DB (SQLite to start, Postgres optional)

### Phase 2 — NLP Sentiment Extraction (~2-3 weeks)
- [ ] Baseline: Loughran-McDonald finance-specific sentiment dictionary
- [ ] Classical ML: TF-IDF + logistic regression / SVM
- [ ] Transformer: FinBERT for sentence/document-level sentiment
- [ ] Aggregate document-level scores into daily/entity-level sentiment signal
- [ ] Compare model performance (F1, latency) against baseline

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
- [ ] Optional: sequence models (LSTM/Transformer) if using sentiment time series
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
