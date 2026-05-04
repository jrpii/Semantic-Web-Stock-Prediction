import argparse
import os
from pathlib import Path
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import xgboost as xgb
except Exception:
    xgboost = None
    xgb = None

import train_lstm_ohlcv as lstm_module

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS_DIR = ROOT / "EVALUATIONS" / "analysis_outputs"
CLEANED_STOCKS_DIR = EVALUATIONS_DIR / "cleaned" / "stocks"
CLEANED_NEWS_FILE = EVALUATIONS_DIR / "cleaned" / "news" / "news_aligned_by_bar.csv"
OUTPUT_DIR = EVALUATIONS_DIR / "charts"
LSTM_MODEL_DIR = EVALUATIONS_DIR / "models" / "lstm"
LSTM_NEWS_MODEL_DIR = EVALUATIONS_DIR / "models" / "lstm_news"
LSTM_FINBERT_MODEL_DIR = EVALUATIONS_DIR / "models" / "lstm_finbert"

FEATURE_COLUMNS = [
    "volume",
    "return_pct",
    "volatility_10",
    "rsi_14",
    "dist_to_sma_5",
    "dist_to_sma_10",
    "dist_to_sma_20",
    "dist_to_ema_10",
    "high_low_spread_pct",
    "article_count",
    "weighted_article_count",
    "avg_word_count",
    "unique_sites",
    "news_volatility_interaction",
    "news_momentum",
]

MODEL_CONFIGS = {
    "LogisticRegression": LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=2000, C=1.0, random_state=42
    ),
    "LinearSVM": LinearSVC(
        penalty="l1", loss="squared_hinge", dual=False, max_iter=3000, C=1.0, random_state=42
    ),
}

if xgb is not None:
    MODEL_CONFIGS["XGBoost"] = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
else:
    print("Warning: XGBoost is unavailable in this environment. The plot will use LogisticRegression and LinearSVM only.")


def build_features(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    news_df = news_df.copy()
    news_df = news_df.rename(columns={"aligned_bar_utc": "timestamp_utc"})
    news_df["timestamp_utc"] = pd.to_datetime(news_df["timestamp_utc"], utc=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.merge(
        news_df,
        left_on=["ticker", "interval_minutes", "timestamp_utc"],
        right_on=["ticker", "interval_minutes", "timestamp_utc"],
        how="left",
    )

    df["dist_to_sma_5"] = (df["close"] - df["sma_5"]) / df["sma_5"]
    df["dist_to_sma_10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
    df["dist_to_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["dist_to_ema_10"] = (df["close"] - df["ema_10"]) / df["ema_10"]
    df["high_low_spread_pct"] = (df["high"] - df["low"]) / df["close"]

    news_cols = ["article_count", "weighted_article_count", "avg_word_count", "unique_sites"]
    df[news_cols] = df[news_cols].fillna(0)

    df["news_volatility_interaction"] = df["weighted_article_count"] * df["volatility_10"]
    df["news_momentum"] = df["article_count"] * df["return_pct"]

    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up_next_bar", "split"])
    return df


def find_latest_checkpoint(interval: int, pattern: str, model_dir: Path) -> Path | None:
    files = sorted(model_dir.glob(f"{interval}m_{pattern}_epoch_*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)))


def load_lstm_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str], str, int]:
    payload = torch.load(path, map_location="cpu")
    cfg = payload["model_config"]
    feature_set = payload.get("feature_set", "")
    tickers = payload.get("tickers", [])
    seq_len = payload["seq_len"]
    model = lstm_module.DirectionalLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_layernorm=cfg["layernorm"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tickers, feature_set, seq_len


def build_lstm_test_timestamps(df: pd.DataFrame, tickers: list[str], seq_len: int) -> np.ndarray:
    timestamps: list[np.datetime64] = []
    for ticker in sorted(tickers):
        sub = df[df["ticker"].astype(str) == ticker].sort_values("timestamp_utc")
        sub = sub[sub["split"] == "test"].reset_index(drop=True)
        if len(sub) < seq_len:
            continue
        timestamps.extend(sub["timestamp_utc"].iloc[seq_len - 1 :].to_numpy())
    return np.asarray(timestamps)


def build_lstm_predictions(df: pd.DataFrame, tickers: list[str], interval: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidates = [
        ("LSTM_NEWS", find_latest_checkpoint(interval, "LSTM_NEWS", LSTM_NEWS_MODEL_DIR), "extended_news"),
        ("LSTM_FINBERT", find_latest_checkpoint(interval, "LSTM_FINBERT", LSTM_FINBERT_MODEL_DIR), "extended_finbert"),
    ]
    df = df.sort_values(["ticker", "timestamp_utc"]).reset_index(drop=True)

    for label, checkpoint, default_feature_set in candidates:
        if checkpoint is None:
            continue
        try:
            model, checkpoint_tickers, saved_feature_set, seq_len = load_lstm_checkpoint(checkpoint)
            feature_set = saved_feature_set or default_feature_set
            tickers_sorted = sorted(checkpoint_tickers) if checkpoint_tickers else sorted(tickers)
            X_train, _, X_val, _ = lstm_module.concat_ticker_fit_val_sequences(
                df, tickers_sorted, seq_len, 0.12, feature_set
            )
            X_test, _, _ = lstm_module.concat_ticker_sequences(df, tickers_sorted, seq_len, "test", feature_set)
            if X_test.size == 0 or X_train.size == 0:
                continue
            X_train_s, X_val_s, X_test_s, _ = lstm_module.scale_sequences_fit_transform(
                X_train, X_val, X_test
            )
            probs = lstm_module.predict_probs_batched(model, X_test_s, torch.device("cpu"), batch_size=128)
            timestamps = build_lstm_test_timestamps(df, tickers_sorted, seq_len)
            if len(timestamps) != len(probs):
                timestamps = timestamps[: len(probs)]
            predictions[label] = (timestamps, probs)
        except Exception as exc:
            print(f"Warning: could not load LSTM checkpoint {checkpoint}: {exc}")
    return predictions


def load_cleaned_stock(ticker: str, interval: int) -> pd.DataFrame:
    path = CLEANED_STOCKS_DIR / f"{ticker}_{interval}m_cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned stock file not found: {path}")
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def prepare_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])
    return train_df, test_df, scaler


def predict_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_up_next_bar"]
    X_test = test_df[FEATURE_COLUMNS]
    predictions = {}

    for name, model in MODEL_CONFIGS.items():
        fitted = model.fit(X_train, y_train)
        if hasattr(fitted, "predict_proba"):
            probs = fitted.predict_proba(X_test)[:, 1]
        else:
            probs = fitted.decision_function(X_test)
        predictions[name] = probs

    return predictions


def plot_predictions(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    predicted_probs: dict[str, np.ndarray],
    ticker: str,
    interval: int,
    out_path: Path,
) -> None:
    fig, (ax_price, ax_pred) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_price.plot(
        test_df["timestamp_utc"],
        test_df["close"],
        color="#1f77b4",
        label=f"{ticker} close",
        linewidth=2.2,
    )
    ax_price.set_ylabel("Close Price", color="#1f77b4")
    ax_price.tick_params(axis="y", labelcolor="#1f77b4")
    ax_price.set_title(
        f"{ticker} {interval}m Close Price and Model Predictions (Test Partition)",
        fontsize=16,
        pad=12,
    )
    ax_price.grid(True, alpha=0.25)

    x_min = None
    x_max = None
    if "LSTM_NEWS" in predicted_probs and isinstance(predicted_probs["LSTM_NEWS"], tuple):
        lstm_news_timestamps, _ = predicted_probs["LSTM_NEWS"]
        if len(lstm_news_timestamps):
            x_min = lstm_news_timestamps[0]
            x_max = lstm_news_timestamps[-1]

    if x_min is not None:
        x_start = pd.Timestamp(year=x_min.year, month=x_min.month, day=1, tz="UTC")
        ax_price.set_xlim(left=x_start)

    if "split" in df.columns:
        first_test = df.loc[df["split"] == "test", "timestamp_utc"].min()
        if pd.notna(first_test):
            ax_price.axvline(first_test, color="#444444", linestyle="--", linewidth=1)
            ax_price.text(
                first_test,
                ax_price.get_ylim()[1],
                "  Test start",
                color="#444444",
                fontsize=9,
                va="top",
            )

    color_map = {
        "LogisticRegression": "#ff7f0e",
        "LinearSVM": "#2ca02c",
        "XGBoost": "#9467bd",
        "LSTM_OHLCV": "#e377c2",
        "LSTM_NEWS": "#17becf",
        "LSTM_FINBERT": "#8c564b",
    }
    style_map = {
        "LogisticRegression": "-",
        "LinearSVM": "--",
        "XGBoost": ":",
        "LSTM_OHLCV": "-",
        "LSTM_NEWS": "--",
        "LSTM_FINBERT": ":",
    }

    for name, data in predicted_probs.items():
        if isinstance(data, tuple):
            timestamps, probs = data
        else:
            timestamps, probs = test_df["timestamp_utc"], data

        ax_pred.plot(
            timestamps,
            probs,
            color=color_map.get(name, "#7f7f7f"),
            linestyle=style_map.get(name, "-"),
            linewidth=2.0,
            label=f"{name}",
            alpha=0.95,
        )

    ax_pred.axhline(0.5, color="#999999", linestyle="-.", linewidth=1)
    ax_pred.set_ylabel("Probability (next bar up)", color="#333333")
    ax_pred.set_ylim(-0.05, 1.05)
    ax_pred.tick_params(axis="y", labelcolor="#333333")
    ax_pred.grid(True, alpha=0.25)

    ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pred.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_pred.xaxis.get_major_locator()))
    ax_pred.set_xlabel("Timestamp")

    ax_pred.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot actual stock prices and model predictions for cleaned stock data."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to plot")
    parser.add_argument("--interval", type=int, default=1440, help="Interval in minutes")
    parser.add_argument(
        "--out-file",
        type=Path,
        default=OUTPUT_DIR / "price_with_model_predictions.png",
        help="Output PNG path",
    )

    args = parser.parse_args()
    CLEANED_STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df = load_cleaned_stock(args.ticker, args.interval)
    news_df = pd.read_csv(CLEANED_NEWS_FILE)
    stock_df = build_features(stock_df, news_df)
    train_df, test_df, _ = prepare_train_test(stock_df)

    if test_df.empty:
        raise ValueError("No test data available for the selected ticker/interval.")

    classical_preds = predict_models(train_df, test_df)
    lstm_preds = build_lstm_predictions(stock_df, [args.ticker], args.interval)
    predicted_probs = {**classical_preds, **lstm_preds}

    plot_predictions(
        stock_df,
        test_df,
        predicted_probs,
        args.ticker,
        args.interval,
        args.out_file,
    )

    print(f"Saved visualization to {args.out_file}")


if __name__ == "__main__":
    main()
