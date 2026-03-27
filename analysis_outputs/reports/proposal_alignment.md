# Proposal Alignment Notes

Implemented now:
- UTC normalization for stock and news timestamps.
- Technical indicators for OHLCV: SMA, EMA, RSI, rolling volatility.
- Deduplicated news metadata outputs.
- Relevance filtering for AAPL and AMZN mentions.
- News-to-next-valid-bar alignment across all available stock intervals.
- Chronological 80/20 split metadata for future leakage-safe modeling.