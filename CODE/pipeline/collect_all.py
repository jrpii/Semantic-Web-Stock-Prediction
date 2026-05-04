import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from collect_ohlcv import collect as collect_ohlcv
from collect_news import collect as collect_news

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = ROOT / "DATA" / "datasets"


def main():
    parser = argparse.ArgumentParser(description="Collect OHLCV and news data for given tickers.")
    parser.add_argument("--tickers", default="AAPL,AMZN", help="Comma-separated ticker symbols")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out", default=str(DEFAULT_DATASETS_DIR), help="Output root directory")
    parser.add_argument("--skip-ohlcv", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    if not args.skip_ohlcv:
        print(f"\n=== OHLCV | {tickers} | {args.start} to {args.end} ===")
        collect_ohlcv(tickers, args.start, args.end, args.out)

    if not args.skip_news:
        print(f"\n=== News | {tickers} | {args.start} to {args.end} ===")
        collect_news(tickers, args.start, args.end, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
