from __future__ import annotations

import csv
import html
import json
import math
import os
import re
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
CHARTS_DIR = DATASETS_DIR / "CHARTS"
NEWS_DIR = DATASETS_DIR / "News"
OUTPUT_DIR = ROOT / "analysis_outputs"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
REPORTS_DIR = OUTPUT_DIR / "reports"

STOCK_HEADERS = [
    "ticker",
    "interval_minutes",
    "date",
    "time",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_range",
    "return_pct",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_10",
    "volatility_10",
    "rsi_14",
    "target_up_next_bar",
    "split",
]

NEWS_HEADERS = [
    "uuid",
    "fingerprint",
    "published_utc",
    "crawled_utc",
    "site",
    "site_full",
    "url",
    "title_clean",
    "author_clean",
    "language",
    "country",
    "organization_mentions",
    "person_mentions",
    "mention_aapl",
    "mention_amzn",
    "text_length",
    "word_count",
]

ALIGNED_NEWS_HEADERS = [
    "ticker",
    "interval_minutes",
    "aligned_bar_utc",
    "article_count",
    "weighted_article_count",
    "avg_word_count",
    "unique_sites",
]

CHRONO_SPLIT_HEADERS = [
    "ticker",
    "interval_minutes",
    "split_ratio",
    "train_rows",
    "test_rows",
    "train_end_utc",
    "test_start_utc",
]

MOJIBAKE_REPLACEMENTS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": "-",
    "\u2013": "-",
    "\xa0": " ",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€”": "-",
    "â€“": "-",
    "â€¦": "...",
}

TICKER_PATTERNS = {
    "AAPL": re.compile(r"\b(aapl|nasdaq:aapl|apple|apple inc)\b", re.IGNORECASE),
    "AMZN": re.compile(r"\b(amzn|nasdaq:amzn|amazon|amazon\.com)\b", re.IGNORECASE),
}


@dataclass
class StockRow:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StockFileSummary:
    ticker: str
    interval_minutes: int
    rows: int
    missing_rows: int
    duplicate_timestamps: int
    date_min: str
    date_max: str
    mean_close: float
    median_close: float
    min_close: float
    max_close: float
    avg_volume: float
    volatility_pct: float


def ensure_output_dirs() -> None:
    for path in [
        OUTPUT_DIR,
        CLEANED_DIR / "stocks",
        CLEANED_DIR / "news",
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    for old, new in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_iso_to_utc(value: object) -> str:
    if not value:
        return ""
    text = clean_text(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    except ValueError:
        return ""


def interval_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer interval from {path.name}")
    return int(digits)


def pct_change(previous: float | None, current: float) -> float | None:
    if previous in (None, 0):
        return None
    return ((current - previous) / previous) * 100.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def rolling_std(values: deque[float]) -> float | None:
    seq = list(values)
    if len(seq) < 2:
        return None
    return stdev(seq)


def compute_rsi(window: deque[float]) -> float | None:
    if len(window) < 14:
        return None
    gains = [value for value in window if value > 0]
    losses = [-value for value in window if value < 0]
    avg_gain = sum(gains) / 14 if gains else 0.0
    avg_loss = sum(losses) / 14 if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def summarize_stock_files() -> tuple[
    list[StockFileSummary],
    dict[tuple[str, int], list[datetime]],
    list[dict[str, object]],
]:
    summaries: list[StockFileSummary] = []
    market_timelines: dict[tuple[str, int], list[datetime]] = {}
    split_rows: list[dict[str, object]] = []

    for csv_path in sorted(CHARTS_DIR.rglob("*.csv")):
        ticker = csv_path.parent.name
        interval_minutes = interval_from_name(csv_path)
        out_path = CLEANED_DIR / "stocks" / f"{ticker}_{interval_minutes}m_cleaned.csv"

        rows = 0
        missing_rows = 0
        duplicate_timestamps = 0
        timestamps_seen: set[str] = set()
        closes: list[float] = []
        volumes: list[float] = []
        returns: list[float] = []
        parsed_rows: list[StockRow] = []

        with csv_path.open("r", newline="", encoding="utf-8") as src:
            reader = csv.reader(src)
            for raw_row in reader:
                if not raw_row:
                    continue
                rows += 1
                if len(raw_row) != 7:
                    missing_rows += 1
                    continue

                date_str, time_str, open_s, high_s, low_s, close_s, volume_s = raw_row
                date_clean = clean_text(date_str)
                time_clean = clean_text(time_str)

                try:
                    timestamp = datetime.strptime(
                        f"{date_clean} {time_clean}", "%Y.%m.%d %H:%M"
                    ).replace(tzinfo=UTC)
                    open_v = float(open_s)
                    high_v = float(high_s)
                    low_v = float(low_s)
                    close_v = float(close_s)
                    volume_v = float(volume_s)
                except ValueError:
                    missing_rows += 1
                    continue

                timestamp_iso = timestamp.isoformat()
                if timestamp_iso in timestamps_seen:
                    duplicate_timestamps += 1
                else:
                    timestamps_seen.add(timestamp_iso)

                parsed_rows.append(
                    StockRow(
                        timestamp=timestamp,
                        open=open_v,
                        high=high_v,
                        low=low_v,
                        close=close_v,
                        volume=volume_v,
                    )
                )
                closes.append(close_v)
                volumes.append(volume_v)

        parsed_rows.sort(key=lambda row: row.timestamp)
        market_timelines[(ticker, interval_minutes)] = [row.timestamp for row in parsed_rows]
        date_min = parsed_rows[0].timestamp.date().isoformat() if parsed_rows else ""
        date_max = parsed_rows[-1].timestamp.date().isoformat() if parsed_rows else ""

        split_index = int(len(parsed_rows) * 0.8)
        if parsed_rows:
            train_end = parsed_rows[max(split_index - 1, 0)].timestamp.isoformat()
            test_start = parsed_rows[min(split_index, len(parsed_rows) - 1)].timestamp.isoformat()
        else:
            train_end = ""
            test_start = ""

        split_rows.append(
            {
                "ticker": ticker,
                "interval_minutes": interval_minutes,
                "split_ratio": "80/20",
                "train_rows": split_index,
                "test_rows": max(len(parsed_rows) - split_index, 0),
                "train_end_utc": train_end,
                "test_start_utc": test_start,
            }
        )

        rolling_5: deque[float] = deque(maxlen=5)
        rolling_10: deque[float] = deque(maxlen=10)
        rolling_20: deque[float] = deque(maxlen=20)
        rolling_returns_10: deque[float] = deque(maxlen=10)
        price_changes_14: deque[float] = deque(maxlen=14)
        previous_close: float | None = None
        ema_10: float | None = None
        ema_alpha = 2 / (10 + 1)

        with out_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst)
            writer.writerow(STOCK_HEADERS)

            for idx, row in enumerate(parsed_rows):
                close_v = row.close
                rolling_5.append(close_v)
                rolling_10.append(close_v)
                rolling_20.append(close_v)

                row_return = pct_change(previous_close, close_v)
                price_change = None if previous_close is None else close_v - previous_close
                previous_close = close_v

                if row_return is not None:
                    returns.append(row_return)
                    rolling_returns_10.append(row_return)
                if price_change is not None:
                    price_changes_14.append(price_change)

                ema_10 = close_v if ema_10 is None else (close_v * ema_alpha) + (ema_10 * (1 - ema_alpha))
                sma_5 = mean(rolling_5) if len(rolling_5) == 5 else None
                sma_10 = mean(rolling_10) if len(rolling_10) == 10 else None
                sma_20 = mean(rolling_20) if len(rolling_20) == 20 else None
                volatility_10 = rolling_std(rolling_returns_10)
                rsi_14 = compute_rsi(price_changes_14)
                target_up_next_bar = ""
                if idx + 1 < len(parsed_rows):
                    target_up_next_bar = int(parsed_rows[idx + 1].close > close_v)

                writer.writerow(
                    [
                        ticker,
                        interval_minutes,
                        row.timestamp.date().isoformat(),
                        row.timestamp.strftime("%H:%M:%S"),
                        row.timestamp.isoformat(),
                        f"{row.open:.6f}",
                        f"{row.high:.6f}",
                        f"{row.low:.6f}",
                        f"{row.close:.6f}",
                        f"{row.volume:.0f}",
                        f"{(row.high - row.low):.6f}",
                        "" if row_return is None else f"{row_return:.6f}",
                        "" if sma_5 is None else f"{sma_5:.6f}",
                        "" if sma_10 is None else f"{sma_10:.6f}",
                        "" if sma_20 is None else f"{sma_20:.6f}",
                        "" if ema_10 is None else f"{ema_10:.6f}",
                        "" if volatility_10 is None else f"{volatility_10:.6f}",
                        "" if rsi_14 is None else f"{rsi_14:.6f}",
                        target_up_next_bar,
                        "train" if idx < split_index else "test",
                    ]
                )

        summaries.append(
            StockFileSummary(
                ticker=ticker,
                interval_minutes=interval_minutes,
                rows=rows,
                missing_rows=missing_rows,
                duplicate_timestamps=duplicate_timestamps,
                date_min=date_min,
                date_max=date_max,
                mean_close=mean(closes) if closes else 0.0,
                median_close=median(closes) if closes else 0.0,
                min_close=min(closes) if closes else 0.0,
                max_close=max(closes) if closes else 0.0,
                avg_volume=mean(volumes) if volumes else 0.0,
                volatility_pct=stdev(returns) if returns else 0.0,
            )
        )

    return summaries, market_timelines, split_rows


def detect_ticker_mentions(title: str, text: str, url: str) -> dict[str, bool]:
    haystack = " ".join(part for part in [title, text, url] if part)
    return {ticker: bool(pattern.search(haystack)) for ticker, pattern in TICKER_PATTERNS.items()}


def process_news_file(json_path: Path) -> dict[str, object] | None:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    thread = payload.get("thread") or {}
    entities = payload.get("entities") or {}
    organizations = entities.get("organizations") or []
    persons = entities.get("persons") or []

    uuid = clean_text(payload.get("uuid"))
    published_utc = safe_iso_to_utc(payload.get("published") or thread.get("published"))
    crawled_utc = safe_iso_to_utc(payload.get("crawled"))
    site = clean_text(thread.get("site"))
    site_full = clean_text(thread.get("site_full"))
    url = clean_text(payload.get("url") or thread.get("url"))
    title_clean = clean_text(payload.get("title") or thread.get("title"))
    author_clean = clean_text(payload.get("author"))
    language = clean_text(payload.get("language")).lower()
    country = clean_text(thread.get("country")).upper()
    text_clean = clean_text(payload.get("text"))
    org_names = [clean_text(item.get("name")) for item in organizations if isinstance(item, dict)]
    person_names = [clean_text(item.get("name")) for item in persons if isinstance(item, dict)]
    text_length = len(text_clean)
    word_count = len(text_clean.split()) if text_clean else 0
    mentions = detect_ticker_mentions(title_clean, text_clean, url)
    fingerprint = clean_text("|".join([url.lower(), title_clean.lower(), published_utc]))

    return {
        "uuid": uuid,
        "fingerprint": fingerprint,
        "published_utc": published_utc,
        "crawled_utc": crawled_utc,
        "site": site,
        "site_full": site_full,
        "url": url,
        "title_clean": title_clean,
        "author_clean": author_clean,
        "language": language,
        "country": country,
        "org_names": [name for name in org_names if name],
        "person_names": [name for name in person_names if name],
        "mention_aapl": mentions["AAPL"],
        "mention_amzn": mentions["AMZN"],
        "text_length": text_length,
        "word_count": word_count,
        "has_text": bool(text_clean),
    }


def find_next_bar(published_utc: str, timeline: list[datetime]) -> datetime | None:
    if not published_utc or not timeline:
        return None
    published_dt = datetime.fromisoformat(published_utc)
    idx = bisect_left(timeline, published_dt)
    if idx >= len(timeline):
        return None
    return timeline[idx]


def summarize_news_files(
    market_timelines: dict[tuple[str, int], list[datetime]]
) -> dict[str, object]:
    metadata_path = CLEANED_DIR / "news" / "news_metadata_cleaned.csv"
    aligned_path = CLEANED_DIR / "news" / "news_aligned_by_bar.csv"
    daily_path = CLEANED_DIR / "news" / "news_daily_counts.csv"

    total_articles = 0
    missing_published = 0
    missing_text = 0
    duplicate_uuids = 0
    duplicate_fingerprints = 0
    deduplicated_articles = 0
    min_published = ""
    max_published = ""
    text_lengths: list[int] = []
    word_counts: list[int] = []
    uuid_seen: set[str] = set()
    fingerprint_seen: set[str] = set()
    site_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()
    org_counter: Counter[str] = Counter()
    person_counter: Counter[str] = Counter()
    ticker_mentions: Counter[str] = Counter()
    daily_counts: defaultdict[str, int] = defaultdict(int)
    aligned_counts: defaultdict[tuple[str, int, str], dict[str, object]] = defaultdict(
        lambda: {"article_count": 0, "weighted_sum": 0.0, "word_count_sum": 0, "sites": set()}
    )

    json_paths = list(NEWS_DIR.rglob("*.json"))
    max_workers = min(16, (os.cpu_count() or 4) * 2)

    with metadata_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(NEWS_HEADERS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(process_news_file, json_paths, chunksize=100):
                total_articles += 1
                if result is None:
                    continue

                uuid = result["uuid"]
                fingerprint = result["fingerprint"]

                if uuid and uuid in uuid_seen:
                    duplicate_uuids += 1
                    continue
                if fingerprint and fingerprint in fingerprint_seen:
                    duplicate_fingerprints += 1
                    continue

                if uuid:
                    uuid_seen.add(uuid)
                if fingerprint:
                    fingerprint_seen.add(fingerprint)

                deduplicated_articles += 1
                published_utc = result["published_utc"]
                if not published_utc:
                    missing_published += 1
                else:
                    day = published_utc[:10]
                    daily_counts[day] += 1
                    min_published = published_utc if not min_published or published_utc < min_published else min_published
                    max_published = published_utc if not max_published or published_utc > max_published else max_published

                if not result["has_text"]:
                    missing_text += 1

                text_lengths.append(result["text_length"])
                word_counts.append(result["word_count"])

                site = result["site"]
                language = result["language"]
                country = result["country"]
                if site:
                    site_counter[site] += 1
                if language:
                    language_counter[language] += 1
                if country:
                    country_counter[country] += 1

                for name in result["org_names"]:
                    org_counter[name.lower()] += 1
                for name in result["person_names"]:
                    person_counter[name.lower()] += 1

                mention_aapl = bool(result["mention_aapl"])
                mention_amzn = bool(result["mention_amzn"])
                if mention_aapl:
                    ticker_mentions["AAPL"] += 1
                if mention_amzn:
                    ticker_mentions["AMZN"] += 1

                writer.writerow(
                    [
                        uuid,
                        fingerprint,
                        published_utc,
                        result["crawled_utc"],
                        site,
                        result["site_full"],
                        result["url"],
                        result["title_clean"],
                        result["author_clean"],
                        language,
                        country,
                        len(result["org_names"]),
                        len(result["person_names"]),
                        int(mention_aapl),
                        int(mention_amzn),
                        result["text_length"],
                        result["word_count"],
                    ]
                )

                for ticker, mentioned in [("AAPL", mention_aapl), ("AMZN", mention_amzn)]:
                    if not mentioned:
                        continue
                    for (timeline_ticker, interval_minutes), timeline in market_timelines.items():
                        if timeline_ticker != ticker:
                            continue
                        next_bar = find_next_bar(published_utc, timeline)
                        if next_bar is None:
                            continue
                        hours_to_bar = max(
                            (next_bar - datetime.fromisoformat(published_utc)).total_seconds() / 3600.0,
                            0.0,
                        )
                        weight = math.exp(-hours_to_bar / 24.0)
                        key = (ticker, interval_minutes, next_bar.isoformat())
                        aligned_counts[key]["article_count"] += 1
                        aligned_counts[key]["weighted_sum"] += weight
                        aligned_counts[key]["word_count_sum"] += result["word_count"]
                        if site:
                            aligned_counts[key]["sites"].add(site)

    with daily_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(["published_date_utc", "article_count"])
        for day in sorted(daily_counts):
            writer.writerow([day, daily_counts[day]])

    with aligned_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(ALIGNED_NEWS_HEADERS)
        for ticker, interval_minutes, aligned_bar in sorted(aligned_counts):
            item = aligned_counts[(ticker, interval_minutes, aligned_bar)]
            article_count = int(item["article_count"])
            avg_word_count = item["word_count_sum"] / article_count if article_count else 0.0
            writer.writerow(
                [
                    ticker,
                    interval_minutes,
                    aligned_bar,
                    article_count,
                    f"{item['weighted_sum']:.6f}",
                    f"{avg_word_count:.2f}",
                    len(item["sites"]),
                ]
            )

    return {
        "total_articles": total_articles,
        "deduplicated_articles": deduplicated_articles,
        "missing_published": missing_published,
        "missing_text": missing_text,
        "duplicate_uuids": duplicate_uuids,
        "duplicate_fingerprints": duplicate_fingerprints,
        "min_published": min_published,
        "max_published": max_published,
        "avg_text_length": mean(text_lengths) if text_lengths else 0.0,
        "median_text_length": median(text_lengths) if text_lengths else 0.0,
        "avg_word_count": mean(word_counts) if word_counts else 0.0,
        "median_word_count": median(word_counts) if word_counts else 0.0,
        "top_sites": site_counter.most_common(10),
        "top_languages": language_counter.most_common(10),
        "top_countries": country_counter.most_common(10),
        "top_organizations": org_counter.most_common(15),
        "top_persons": person_counter.most_common(15),
        "ticker_mentions": dict(ticker_mentions),
        "daily_counts": daily_counts,
        "aligned_bars": len(aligned_counts),
    }


def write_stock_report(summaries: list[StockFileSummary], split_rows: list[dict[str, object]]) -> None:
    csv_path = REPORTS_DIR / "stock_file_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(
            [
                "ticker",
                "interval_minutes",
                "rows",
                "missing_rows",
                "duplicate_timestamps",
                "date_min",
                "date_max",
                "mean_close",
                "median_close",
                "min_close",
                "max_close",
                "avg_volume",
                "volatility_pct",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.ticker,
                    item.interval_minutes,
                    item.rows,
                    item.missing_rows,
                    item.duplicate_timestamps,
                    item.date_min,
                    item.date_max,
                    f"{item.mean_close:.6f}",
                    f"{item.median_close:.6f}",
                    f"{item.min_close:.6f}",
                    f"{item.max_close:.6f}",
                    f"{item.avg_volume:.6f}",
                    f"{item.volatility_pct:.6f}",
                ]
            )

    split_path = REPORTS_DIR / "chronological_splits.csv"
    with split_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=CHRONO_SPLIT_HEADERS)
        writer.writeheader()
        writer.writerows(split_rows)


def write_summary_markdown(
    stock_summaries: list[StockFileSummary],
    news_summary: dict[str, object],
) -> None:
    stock_total_rows = sum(item.rows for item in stock_summaries)
    stock_missing_rows = sum(item.missing_rows for item in stock_summaries)
    stock_duplicates = sum(item.duplicate_timestamps for item in stock_summaries)
    tickers = sorted({item.ticker for item in stock_summaries})
    intervals = sorted({item.interval_minutes for item in stock_summaries})

    lines = [
        "# Data Cleaning and EDA Summary",
        "",
        "## Proposal Alignment",
        "",
        "- Raw OHLCV timestamps are standardized to UTC-style ISO output files.",
        "- Stock files now include technical indicators: moving averages, EMA, RSI, volatility, and next-bar direction labels.",
        "- News articles are cleaned and deduplicated before being written to derived outputs.",
        "- Relevant AAPL and AMZN articles are aligned to the next valid market bar for each stock interval.",
        "- Chronological 80/20 split metadata is exported to support leakage-safe modeling.",
        "",
        "## Stock Dataset Summary",
        "",
        f"- Tickers found: {', '.join(tickers)}",
        f"- Intervals found (minutes): {', '.join(str(i) for i in intervals)}",
        f"- Source stock files: {len(stock_summaries)}",
        f"- Total stock rows processed: {stock_total_rows}",
        f"- Invalid or skipped stock rows: {stock_missing_rows}",
        f"- Duplicate stock timestamps detected: {stock_duplicates}",
        "",
        "| Ticker | Interval | Rows | Date Min | Date Max | Mean Close | Avg Volume | Volatility % |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]

    for item in stock_summaries:
        lines.append(
            f"| {item.ticker} | {item.interval_minutes} | {item.rows} | {item.date_min} | {item.date_max} | "
            f"{item.mean_close:.3f} | {item.avg_volume:.1f} | {item.volatility_pct:.4f} |"
        )

    lines.extend(
        [
            "",
            "## News Dataset Summary",
            "",
            f"- Raw articles scanned: {news_summary['total_articles']}",
            f"- Deduplicated articles retained: {news_summary['deduplicated_articles']}",
            f"- Missing published timestamps: {news_summary['missing_published']}",
            f"- Missing text bodies: {news_summary['missing_text']}",
            f"- Duplicate UUIDs removed: {news_summary['duplicate_uuids']}",
            f"- Duplicate fingerprint matches removed: {news_summary['duplicate_fingerprints']}",
            f"- Published UTC range: {news_summary['min_published']} to {news_summary['max_published']}",
            f"- Average text length: {news_summary['avg_text_length']:.1f} characters",
            f"- Median text length: {news_summary['median_text_length']:.1f} characters",
            f"- Average word count: {news_summary['avg_word_count']:.1f}",
            f"- Median word count: {news_summary['median_word_count']:.1f}",
            f"- Relevance-filtered ticker mentions: AAPL={news_summary['ticker_mentions'].get('AAPL', 0)}, "
            f"AMZN={news_summary['ticker_mentions'].get('AMZN', 0)}",
            f"- Aligned market bars with at least one article: {news_summary['aligned_bars']}",
            "",
            "### Top News Sites",
            "",
            "| Site | Articles |",
            "|---|---:|",
        ]
    )

    for site, count in news_summary["top_sites"]:
        lines.append(f"| {site} | {count} |")

    lines.extend(
        [
            "",
            "### Top Languages",
            "",
            "| Language | Articles |",
            "|---|---:|",
        ]
    )
    for language, count in news_summary["top_languages"]:
        lines.append(f"| {language} | {count} |")

    lines.extend(
        [
            "",
            "### Top Countries",
            "",
            "| Country | Articles |",
            "|---|---:|",
        ]
    )
    for country, count in news_summary["top_countries"]:
        lines.append(f"| {country} | {count} |")

    lines.extend(
        [
            "",
            "### Top Organization Mentions",
            "",
            "| Organization | Mentions |",
            "|---|---:|",
        ]
    )
    for org, count in news_summary["top_organizations"]:
        lines.append(f"| {org} | {count} |")

    lines.extend(
        [
            "",
            "### Top Person Mentions",
            "",
            "| Person | Mentions |",
            "|---|---:|",
        ]
    )
    for person, count in news_summary["top_persons"]:
        lines.append(f"| {person} | {count} |")

    report_path = REPORTS_DIR / "eda_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_proposal_alignment_note() -> None:
    lines = [
        "# Proposal Alignment Notes",
        "",
        "This preprocessing pass is aligned to Sections 2 and 4 of the proposal.",
        "",
        "Implemented now:",
        "- UTC normalization for stock and news timestamps.",
        "- Technical indicators for OHLCV: SMA, EMA, RSI, rolling volatility.",
        "- Deduplicated news metadata outputs.",
        "- Relevance filtering for AAPL and AMZN mentions.",
        "- News-to-next-valid-bar alignment across all available stock intervals.",
        "- Chronological 80/20 split metadata for future leakage-safe modeling.",
        "",
        "Still future work from the proposal:",
        "- FinBERT or other transformer-based sentiment embeddings.",
        "- Knowledge-graph construction from extracted entities.",
        "- Classical and modern predictive models.",
        "- Backtesting, explainability, and formal unit/integration tests.",
        "",
        "This means the repository now covers the initial data cleaning, formatting, and EDA phase of the proposal, plus the first layer of feature engineering.",
    ]
    (REPORTS_DIR / "proposal_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    stock_summaries, market_timelines, split_rows = summarize_stock_files()
    news_summary = summarize_news_files(market_timelines)
    write_stock_report(stock_summaries, split_rows)
    write_summary_markdown(stock_summaries, news_summary)
    write_proposal_alignment_note()
    print(f"Finished. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
