# Build FinBERT features per market bar from raw news JSON files.
# Output: analysis_outputs/cleaned/news/news_finbert_by_bar.csv

# Read each article text
# Run FinBERT sentiment (pos/neg/neu)
# Align each article to the next market bar (like clean_and_eda.py)
# Aggregate sentiment per (ticker, interval, bar)
import argparse
import json
import math
import os
import re
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "analysis_outputs" / "cleaned" / "news" / "news_finbert_by_bar.csv"
DEFAULT_STOCKS_DIR = ROOT / "analysis_outputs" / "cleaned" / "stocks"
_AAPL_RE = re.compile(r"\b(aapl|nasdaq:aapl|apple|apple inc)\b", re.IGNORECASE)
_AMZN_RE = re.compile(r"\b(amzn|nasdaq:amzn|amazon|amazon\.com)\b", re.IGNORECASE)

def parse_published_dt(value):
    if value is None:
        return None
    s = str(value)
    if not s:
        return None
    s = s.replace("\x00", "").strip()
    if not s:
        return None
    if len(s) > 80:
        s = s[:80]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    m = re.search(r"([+-]\d{2})(\d{2})$", s)
    if m and ":" not in m.group(0):
        s = s[:-5] + m.group(1) + ":" + m.group(2)
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_next_bar(published_dt, timeline):
    if published_dt is None or not timeline:
        return None
    idx = bisect_left(timeline, published_dt)
    if idx >= len(timeline):
        return None
    return timeline[idx]


def load_timelines(stocks_dir, tickers, intervals):
    timelines = {}
    for ticker in tickers:
        for iv in intervals:
            path = Path(stocks_dir) / f"{ticker}_{int(iv)}m_cleaned.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
            timelines[(ticker, int(iv))] = [ts.to_pydatetime() for ts in df["timestamp_utc"].tolist()]
    return timelines


def fix_ssl_env():
    # Some conda setups set SSL_CERT_FILE to a path that doesn't exist.
    # httpx/ssl will crash if that env var points to a missing file.
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile and not Path(cafile).exists():
        print(f"Warning: SSL_CERT_FILE points to missing file: {cafile}", flush=True)
        print("Unsetting SSL_CERT_FILE so HuggingFace downloads can work.", flush=True)
        os.environ.pop("SSL_CERT_FILE", None)

    cabundle = os.environ.get("REQUESTS_CA_BUNDLE")
    if cabundle and not Path(cabundle).exists():
        print(f"Warning: REQUESTS_CA_BUNDLE points to missing file: {cabundle}", flush=True)
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

    # If certifi exists, prefer it.
    try:
        import certifi  # type: ignore

        cert_path = certifi.where()
        if cert_path and Path(cert_path).exists():
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
    except Exception:
        pass

    # Optional: silence the Windows symlink warning for HF cache.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-json-dir", type=Path, default=ROOT / "datasets" / "News")
    parser.add_argument("--stocks-dir", type=Path, default=DEFAULT_STOCKS_DIR)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "AMZN"])
    parser.add_argument("--intervals", nargs="+", type=int, default=[5, 15, 30, 60, 240, 1440])
    parser.add_argument("--model", default="ProsusAI/finbert")
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    fix_ssl_env()

    news_dir = Path(args.news_json_dir)
    json_paths = list(news_dir.rglob("*.json"))
    print(f"Found {len(json_paths)} news json files under {news_dir}", flush=True)

    timelines = load_timelines(args.stocks_dir, list(args.tickers), list(args.intervals))
    if not timelines:
        raise ValueError("No stock timelines found. Check cleaned stock CSVs.")
    print(f"Loaded timelines for {len(timelines)} (ticker, interval) pairs", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    # Prefer safetensors to avoid torch.load security restrictions on older torch versions.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model),
        use_safetensors=True,
    ).to(device)
    model.eval()

    # Try to map label ids to pos/neg/neu.
    id2label = {int(k): str(v).lower() for k, v in getattr(model.config, "id2label", {}).items()}
    label_to_id = {v: k for k, v in id2label.items()}
    pos_id = label_to_id.get("positive")
    neg_id = label_to_id.get("negative")
    neu_id = label_to_id.get("neutral")
    if pos_id is None or neg_id is None or neu_id is None:
        # Common fallback for finbert: 0=negative,1=neutral,2=positive
        neg_id, neu_id, pos_id = 0, 1, 2

    # Aggregation: key=(ticker, interval, bar_iso) -> running sums.
    agg = {}

    def ensure_row(ticker, interval, bar_iso):
        key = (ticker, int(interval), bar_iso)
        if key not in agg:
            agg[key] = {
                "article_count": 0,
                "weighted_article_count": 0.0,
                "pos_sum": 0.0,
                "neg_sum": 0.0,
                "neu_sum": 0.0,
                "sent_sum": 0.0,
                "strength_sum": 0.0,
                "confidence_sum": 0.0,
                "entropy_sum": 0.0,
            }
        return key

    batch_texts = []
    batch_meta = []
    scanned = 0
    matched = 0

    for p in json_paths:
        scanned += 1
        if scanned % 2000 == 0:
            print(f"  scanned {scanned} | matched {matched} | bars {len(agg)}", flush=True)

        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        thread = payload.get("thread") or {}
        published_utc = payload.get("published") or thread.get("published")
        pub = parse_published_dt(published_utc)
        if pub is None:
            continue

        title = str(payload.get("title") or thread.get("title") or "").strip()
        body = str(payload.get("text") or "").strip()
        url = str(payload.get("url") or thread.get("url") or "").strip()
        hay = (title + " " + body[:2000] + " " + url).lower()

        mentioned = []
        if "AAPL" in args.tickers and _AAPL_RE.search(hay):
            mentioned.append("AAPL")
        if "AMZN" in args.tickers and _AMZN_RE.search(hay):
            mentioned.append("AMZN")
        if not mentioned:
            continue

        text = body if body else title
        text = text.strip()
        if not text:
            continue
        if args.max_chars and len(text) > int(args.max_chars):
            text = text[: int(args.max_chars)]

        # Precompute all alignments for this article (per mentioned ticker and interval).
        alignments = []
        for tkr in mentioned:
            for iv in args.intervals:
                tl = timelines.get((tkr, int(iv)))
                if not tl:
                    continue
                bar_dt = find_next_bar(pub, tl)
                if bar_dt is None:
                    continue
                hours_to_bar = max((bar_dt - pub).total_seconds() / 3600.0, 0.0)
                weight = math.exp(-hours_to_bar / 24.0)
                alignments.append((tkr, int(iv), bar_dt.isoformat(), float(weight)))
        if not alignments:
            continue

        matched += 1
        batch_texts.append(text)
        batch_meta.append(alignments)

        if len(batch_texts) >= int(args.batch_size):
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(args.max_tokens),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for i in range(len(batch_texts)):
                pr = probs[i]
                p_pos = float(pr[pos_id])
                p_neg = float(pr[neg_id])
                p_neu = float(pr[neu_id])
                sent = p_pos - p_neg
                strength = abs(sent)
                confidence = max(p_pos, p_neg, p_neu)
                entropy = -(
                    p_pos * math.log(p_pos + 1e-12)
                    + p_neg * math.log(p_neg + 1e-12)
                    + p_neu * math.log(p_neu + 1e-12)
                ) / math.log(3.0)
                for (tkr, iv, bar_iso, w) in batch_meta[i]:
                    key = ensure_row(tkr, iv, bar_iso)
                    agg[key]["article_count"] += 1
                    agg[key]["weighted_article_count"] += w
                    agg[key]["pos_sum"] += w * p_pos
                    agg[key]["neg_sum"] += w * p_neg
                    agg[key]["neu_sum"] += w * p_neu
                    agg[key]["sent_sum"] += w * sent
                    agg[key]["strength_sum"] += w * strength
                    agg[key]["confidence_sum"] += w * confidence
                    agg[key]["entropy_sum"] += w * entropy

            batch_texts = []
            batch_meta = []

    # Flush last batch
    if batch_texts:
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(args.max_tokens),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for i in range(len(batch_texts)):
            pr = probs[i]
            p_pos = float(pr[pos_id])
            p_neg = float(pr[neg_id])
            p_neu = float(pr[neu_id])
            sent = p_pos - p_neg
            strength = abs(sent)
            confidence = max(p_pos, p_neg, p_neu)
            entropy = -(
                p_pos * math.log(p_pos + 1e-12)
                + p_neg * math.log(p_neg + 1e-12)
                + p_neu * math.log(p_neu + 1e-12)
            ) / math.log(3.0)
            for (tkr, iv, bar_iso, w) in batch_meta[i]:
                key = ensure_row(tkr, iv, bar_iso)
                agg[key]["article_count"] += 1
                agg[key]["weighted_article_count"] += w
                agg[key]["pos_sum"] += w * p_pos
                agg[key]["neg_sum"] += w * p_neg
                agg[key]["neu_sum"] += w * p_neu
                agg[key]["sent_sum"] += w * sent
                agg[key]["strength_sum"] += w * strength
                agg[key]["confidence_sum"] += w * confidence
                agg[key]["entropy_sum"] += w * entropy

    print(f"Done. bars with finbert features: {len(agg)}", flush=True)

    rows = []
    for (tkr, iv, bar_iso), v in agg.items():
        wsum = float(v["weighted_article_count"]) if v["weighted_article_count"] else 0.0
        if wsum <= 0:
            pos = neg = neu = sent = strength = conf = ent = 0.0
        else:
            pos = float(v["pos_sum"]) / wsum
            neg = float(v["neg_sum"]) / wsum
            neu = float(v["neu_sum"]) / wsum
            sent = float(v["sent_sum"]) / wsum
            strength = float(v["strength_sum"]) / wsum
            conf = float(v["confidence_sum"]) / wsum
            ent = float(v["entropy_sum"]) / wsum
        rows.append(
            {
                "ticker": tkr,
                "interval_minutes": int(iv),
                "aligned_bar_utc": bar_iso,
                "article_count": int(v["article_count"]),
                "weighted_article_count": float(v["weighted_article_count"]),
                "finbert_pos": pos,
                "finbert_neg": neg,
                "finbert_neu": neu,
                "finbert_sentiment": sent,
                "finbert_strength": strength,
                "finbert_confidence": conf,
                "finbert_entropy": ent,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["ticker", "interval_minutes", "aligned_bar_utc"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()

