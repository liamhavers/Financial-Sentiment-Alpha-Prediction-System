"""Shared configuration for the ingestion pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the working directory, if present


TICKERS = ["AAPL", "NVDA", "GOOG", "AMZN", "MU", "PLTR", "RDDT", "AI", "TSLA", "META", "MSFT"]

# --- Postgres connection (env-driven; see .env.example) ---
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "5432")
PG_DATABASE = os.environ.get("PGDATABASE", "sentiment")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "")
