"""Shared configuration for the ingestion pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the working directory, if present


TICKERS = ["AAPL", "NVDA", "GOOG", "AMZN", "MU", "PLTR", "RDDT", "AI", "TSLA", "META", "MSFT"]

# Benchmark used for the excess-return label (see CLAUDE.md Fixed Decisions).
# Not part of the locked ticker universe — priced separately for label
# construction, not analyzed as a signal target itself.
BENCHMARK_TICKER = "QQQ"

# Volume floor (see phase0_proposal.md / README Phase 0 Decisions): below this
# many deduplicated messages on a given (ticker, aligned trading day), treat
# that day's sentiment as missing rather than a noisy signal.
MIN_DAILY_MESSAGES = 3

# --- Postgres connection (env-driven; see .env.example) ---
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "5432")
PG_DATABASE = os.environ.get("PGDATABASE", "sentiment")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "")
