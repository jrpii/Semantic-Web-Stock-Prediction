import requests
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

import spacy
import newspaper

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = ROOT / "DATA" / "datasets"

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

TICKER_QUERIES = {
    "AAPL": "(Apple OR AAPL) stock",
    "AMZN": "(Amazon OR AMZN) stock",
}


def build_query(ticker):
    return TICKER_QUERIES.get(ticker.upper(), f"({ticker}) stock")


def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def gdelt_fetch(query, start_dt, end_dt):
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": 250,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "format": "json",
        "sourcelang": "english",
    }
    for attempt in range(3):
        try:
            resp = requests.get(GDELT_API, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except Exception:
            time.sleep(6)
    return []


def scrape_article(url):
    try:
        art = newspaper.Article(url)
        art.download()
        art.parse()
        text = art.text or ""
        author = art.authors[0] if art.authors else ""
        return text, author
    except Exception:
        return "", ""


def extract_entities(text, nlp):
    doc = nlp(text[:10000])
    persons = [{"name": ent.text.lower(), "sentiment": "none"}
               for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [{"name": ent.text.lower(), "sentiment": "none"}
            for ent in doc.ents if ent.label_ == "ORG"]
    locations = [{"name": ent.text.lower(), "sentiment": "none"}
                 for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return persons, orgs, locations


def parse_gdelt_date(raw):
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
    except Exception:
        return raw


def to_schema(article, text, author, persons, orgs, locations):
    url = article.get("url", "")
    uid = hashlib.md5(url.encode()).hexdigest()
    published = parse_gdelt_date(article.get("seendate", ""))
    domain = article.get("domain", "")
    title = article.get("title", "")
    crawled = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000+00:00")

    return {
        "uuid": uid,
        "thread": {
            "uuid": uid,
            "url": url,
            "site_full": domain,
            "site": domain,
            "site_section": "",
            "section_title": "",
            "title": title,
            "title_full": "",
            "published": published,
            "replies_count": 0,
            "participants_count": 1,
            "site_type": "news",
            "country": article.get("sourcecountry", ""),
            "spam_score": 0.0,
            "main_image": article.get("socialimage", ""),
            "performance_score": 0,
            "social": {
                "gplus": {"shares": 0},
                "pinterest": {"shares": 0},
                "vk": {"shares": 0},
                "linkedin": {"shares": 0},
                "facebook": {"likes": 0, "shares": 0, "comments": 0},
                "stumbledupon": {"shares": 0},
            },
        },
        "author": author,
        "url": url,
        "ord_in_thread": 0,
        "title": title,
        "highlightText": "",
        "highlightTitle": "",
        "language": "english",
        "external_links": [],
        "published": published,
        "crawled": crawled,
        "text": text,
        "persons": persons,
        "organizations": orgs,
        "locations": locations,
        "entities": {
            "persons": persons,
            "locations": locations,
            "organizations": orgs,
        },
    }


def next_month(dt):
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1)
    return datetime(dt.year, dt.month + 1, 1)


def collect(tickers, start_date, end_date, out_root, max_per_month=100):
    nlp = load_nlp()

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cursor = datetime(start.year, start.month, 1)

    while cursor <= end:
        month_end = min(next_month(cursor) - timedelta(seconds=1), end)
        year_month = cursor.strftime("%Y_%m")

        folder_hash = hashlib.md5(f"{'_'.join(sorted(tickers))}{year_month}".encode()).hexdigest()
        out_dir = Path(out_root) / "News" / f"{year_month}_{folder_hash}"

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  {year_month} already exists, skipping")
            cursor = next_month(cursor)
            continue

        raw_articles = []
        seen_urls = set()

        for ticker in tickers:
            query = build_query(ticker)
            print(f"  GDELT {ticker} {year_month} ...", end=" ", flush=True)
            fetched = gdelt_fetch(query, cursor, month_end)
            added = 0
            for article in fetched:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_articles.append(article)
                    added += 1
            print(f"{added} articles")
            time.sleep(6.0)

        raw_articles = raw_articles[:max_per_month]

        if raw_articles:
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, article in enumerate(raw_articles, 1):
                url = article.get("url", "")
                print(f"    [{i}/{len(raw_articles)}] {url[:70]}", end=" ... ", flush=True)

                text, author = scrape_article(url)
                persons, orgs, locations = extract_entities(text, nlp) if text else ([], [], [])

                record = to_schema(article, text, author, persons, orgs, locations)
                (out_dir / f"news_{i:07d}.json").write_text(
                    json.dumps(record, ensure_ascii=False), encoding="utf-8"
                )

                print(f"{len(text)} chars" if text else "no text")
                time.sleep(0.5)

        cursor = next_month(cursor)


if __name__ == "__main__":
    tickers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AAPL", "AMZN"]
    start = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    root = sys.argv[4] if len(sys.argv) > 4 else str(DEFAULT_DATASETS_DIR)
    max_per_month = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    print(f"Collecting news for {tickers} | {start} to {end} | max {max_per_month}/month")
    collect(tickers, start, end, root, max_per_month)
