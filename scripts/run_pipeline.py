"""CLI entry point for running the data pipeline."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the financial data pipeline")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Specific tickers to process (default: all configured companies)",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding and indexing steps (for testing fetch/parse/chunk only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    asyncio.run(
        run_pipeline(
            tickers=args.tickers,
            skip_embedding=args.skip_embedding,
        )
    )


if __name__ == "__main__":
    main()
