"""Run ingestion sources against the shared Postgres database."""

import logging

import stocktwits_ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=== StockTwits ingestion ===")
    stocktwits_ingest.run()

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
