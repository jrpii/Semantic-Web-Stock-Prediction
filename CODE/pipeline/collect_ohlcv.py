import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_DIR = ROOT / "DATA" / "datasets"

INTERVAL_SUFFIXES = {5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "1h", 1440: "1d"}
MAX_CHUNK_DAYS = {"5m": 55, "15m": 55, "30m": 55, "1h": 700, "1d": 9999}
YAHOO_LIMITS = {"5m": 59, "15m": 59, "30m": 59, "1h": 729}


def download_chunked(ticker, interval, start, end):
    if interval in YAHOO_LIMITS:
        cutoff = datetime.utcnow() - timedelta(days=YAHOO_LIMITS[interval])
        start = max(start, cutoff)
        if start >= end:
            return pd.DataFrame()

    delta = timedelta(days=MAX_CHUNK_DAYS[interval])
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        try:
            df = yf.download(
                ticker,
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)
    return pd.concat(frames).drop_duplicates() if frames else pd.DataFrame()


def to_4h(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.resample("4h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def write_csv(df, path):
    lines = []
    for ts, row in df.iterrows():
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        lines.append(
            f"{ts.strftime('%Y.%m.%d')},{ts.strftime('%H:%M')},"
            f"{row['Open']:.3f},{row['High']:.3f},{row['Low']:.3f},{row['Close']:.3f},{int(row['Volume'])}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(tickers, start_date, end_date, out_root):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    for ticker in tickers:
        out_dir = Path(out_root) / "CHARTS" / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for mins, yf_iv in INTERVAL_SUFFIXES.items():
            if yf_iv in YAHOO_LIMITS:
                cutoff = datetime.utcnow() - timedelta(days=YAHOO_LIMITS[yf_iv])
                if end < cutoff:
                    print(f"  {ticker} {mins}min ... skipped (Yahoo limit: {YAHOO_LIMITS[yf_iv]}d for {yf_iv})")
                    continue

            print(f"  {ticker} {mins}min ...", end=" ", flush=True)

            df = download_chunked(ticker, yf_iv, start, end)
            if df.empty:
                print("no data")
                continue

            df = flatten_columns(df)
            df.index = pd.to_datetime(df.index)

            if mins == 240:
                df = to_4h(df)
            elif df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            out_path = out_dir / f"{ticker}{mins}.csv"
            write_csv(df, out_path)
            print(f"{len(df)} rows -> {out_path.name}")


if __name__ == "__main__":
    tickers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AAPL", "AMZN"]
    start = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"
    root = sys.argv[4] if len(sys.argv) > 4 else str(DEFAULT_DATASETS_DIR)

    print(f"Collecting OHLCV for {tickers} | {start} to {end}")
    collect(tickers, start, end, root)
