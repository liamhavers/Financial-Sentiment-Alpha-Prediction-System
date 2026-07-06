# Phase 0 Proposal: Financial Sentiment & Alpha Prediction System

## 1. Thesis

Retail sentiment expressed on StockTwits about a fixed universe of large-cap and
high-retail-interest technology stocks contains information not yet reflected in price.
If this signal exists, it should predict short-horizon forward returns (relative to the
broader tech sector) and should decay measurably between one day and five days after
the sentiment is expressed.

**Falsifiable prediction**: a sentiment score derived from StockTwits posts will show a
statistically significant, non-zero rank correlation (Information Coefficient) with
1-day and 5-day forward excess returns (vs. QQQ) for the tickers below. If no such
correlation exists, or it does not survive basic robustness checks (see Section 6), the
thesis is rejected for this universe and horizon.

## 2. Universe

`AAPL, NVDA, GOOG, AMZN, MU, PLTR, RDDT, AI, TSLA, META, MSFT`

11 tickers, fixed prior to any data collection or analysis, chosen to mix:
- **Mega-cap, heavily-covered tech** (AAPL, NVDA, GOOG, AMZN, META, MSFT, TSLA) — high
  liquidity and post volume, but sentiment may already be efficiently priced by
  institutional flow, making alpha harder to detect.
- **Thinner-coverage / higher-retail-attention names** (MU, PLTR, RDDT, AI) — lower
  institutional coverage means retail sentiment plausibly carries more marginal
  information, but data volume per ticker will be noisier and sparser.

This list will not be changed after results are seen. Any future addition or removal of
tickers is a new experiment, not a refinement of this one.

## 3. Data

**Sentiment data**: StockTwits — ticker-tagged posts (`$TICKER` cashtags), including the
platform's own user-supplied bullish/bearish label, which will serve as a free baseline
sentiment signal to sanity-check any custom NLP-derived score against.

**Price data**: Daily OHLCV for each ticker plus QQQ (as the sector benchmark), sourced
via yfinance or a comparable free provider.

**Why StockTwits first**: ticker-tagging is already built in (no need to extract company
mentions via regex/NER), and the platform's own sentiment label provides a natural
benchmark. Reddit is a stretch goal once this pipeline is validated end-to-end.

## 4. Horizon & Labels

Two forward horizons will be tested in parallel:
- **t+1**: next trading day's return
- **t+5**: return five trading days forward

Both are measured as **excess return relative to QQQ**, not raw return — this is a more
defensible measure of "alpha" than absolute price movement, since it controls for
general tech-sector moves that have nothing to do with stock-specific sentiment.

Testing both horizons supports a signal decay analysis in Phase 3: if the IC at t+1 is
meaningfully higher than at t+5, that's consistent with a real but short-lived signal;
if there's no difference, that's a warning sign the "signal" may just be noise or a
proxy for something else (e.g. contemporaneous price momentum).

## 5. Known Assumptions & Limitations (accepted upfront)

- **Survivorship**: all 11 tickers are currently listed and liquid; no delisted names
  are included, which is a mild survivorship bias worth naming even though it's
  standard for a project of this scope.
- **RDDT listing date**: RDDT IPO'd in March 2024. Any backtest window must either
  start no earlier than RDDT's IPO or exclude RDDT from periods before it existed.
- **Look-ahead control**: a StockTwits post only counts toward a given day's sentiment
  score if it was posted before that day's market close. Same-day contamination (a post
  timestamped after close being used to predict that same day's return) will be
  explicitly checked for in the data pipeline, not just assumed away.
- **Volume floor**: tickers/days with fewer than a minimum post count (to be set once
  actual volumes are observed, e.g. after an initial data pull) will have their
  sentiment score treated as missing rather than as a noisy but usable data point.
- **Noise in social data**: StockTwits posts contain sarcasm, hype, and possible bot or
  spam activity. The user-tagged bullish/bearish label will be used in Phase 2 as one
  check on whether a custom NLP sentiment score is capturing something sensible.
- **Multiple testing**: because two horizons and up to 11 tickers will be tested, any
  positive result must be interpreted with this in mind — a single significant IC
  out of many tests is not strong evidence on its own (addressed more rigorously in
  Phase 3).

## 6. What Would Change or Reject the Thesis

- IC/rank-IC between sentiment score and forward excess return is not statistically
  distinguishable from zero at either horizon, across the fixed universe.
- Any observed relationship disappears once neutralized against basic momentum
  (i.e., the "sentiment signal" is really just price momentum restated in text form).
- Signal quality does not differ meaningfully between t+1 and t+5 (suggesting noise
  rather than a decaying information effect).

## 7. Next Step

Proceed to Phase 1: build the StockTwits ingestion pipeline for the 11 fixed tickers,
with strict timestamp controls to avoid look-ahead contamination, and pull matching
daily price data for the same tickers plus QQQ.
