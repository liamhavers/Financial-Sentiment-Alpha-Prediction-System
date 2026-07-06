# Financial Sentiment & Alpha Prediction System

A research project combining NLP, alternative data, and statistical modeling to test whether
text-derived sentiment signals carry predictive information about forward asset returns.

Built as a portfolio project for a transition into quantitative research/trading roles.

## Project Thesis

> Sentiment extracted from [news / earnings call transcripts / SEC filings / social media]
> about [equity universe] contains information not yet reflected in price, and this signal
> decays over [target horizon].

This is a placeholder — fill in your specific hypothesis here once Phase 0 is complete.
A clear, falsifiable thesis is the foundation of the whole project; everything downstream
exists to test it.

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
- [ ] Define universe (single sector vs. broad market)
- [ ] Define prediction horizon (intraday / 1-day / 1-week)
- [ ] Choose primary data source(s)
- [ ] Write one-page thesis statement (see above)

### Phase 1 — Data Pipeline (~2-3 weeks)
- [ ] Ingest alternative data: news (NewsAPI / GDELT), filings (SEC EDGAR), and/or social (Reddit / StockTwits)
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
