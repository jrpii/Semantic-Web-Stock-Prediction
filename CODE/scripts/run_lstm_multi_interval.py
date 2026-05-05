# Run the multi-interval fusion LSTM once and append rows to a CSV (per target horizon).
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "CODE" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lstm_multi_interval as multi

CSV_COLUMNS = ["dataset", "model", "roc_auc", "f1", "precision", "recall", "accuracy"]


def main():
    # Mirrors train_lstm_multi_interval CLI.
    p = argparse.ArgumentParser()
    p.add_argument("--out-csv", type=Path, default=ROOT / "EVALUATIONS" / "analysis_outputs" / "reports" / "lstm_multi_interval_results.csv")
    p.add_argument("--stocks-dir", type=Path, default=multi.DEFAULT_STOCKS_DIR)
    p.add_argument("--intervals", nargs="+", type=int, default=list(multi.DEFAULT_INTERVALS))
    p.add_argument("--tickers", nargs="+", default=["AAPL", "AMZN"])
    p.add_argument(
        "--feature-set",
        choices=(
            "ohlcv",
            "extended",
            "extended_news",
            "extended_finbert",
            "extended_finbert_eng",
            "extended_news_finbert",
            "extended_news_finbert_eng",
        ),
        default="extended_news_finbert_eng",
    )
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--target-emb-dim", type=int, default=16)
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-balance-loss", action="store_true")
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cyclic-phases", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--out-model", type=Path, default=None)
    p.add_argument("--news-aligned", type=Path, default=multi.DEFAULT_NEWS_ALIGNED)
    p.add_argument("--finbert-by-bar", type=Path, default=multi.DEFAULT_FINBERT_BY_BAR)
    p.add_argument("--no-news-merge", action="store_true")
    args = p.parse_args()

    args.balance_loss = not args.no_balance_loss

    # Train once; report dict holds per-target test metrics plus pooled breakout stats.
    report = multi.run_training(args)

    # Flatten to same column schema as other reports.
    rows = []
    model_name = "LSTM_MULTI"
    for iv_str, m in report["per_target_test"].items():
        rows.append(
            {
                "dataset": f"{iv_str}m_ALL",
                "model": model_name,
                "roc_auc": m["roc_auc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "accuracy": m["accuracy"],
            }
        )
    bo = report.get("test_breakout_all_targets")
    if bo is not None:
        rows.append(
            {
                "dataset": "ALL_BREAKOUTS",
                "model": model_name,
                "roc_auc": bo["roc_auc"],
                "f1": bo["f1"],
                "precision": bo["precision"],
                "recall": bo["recall"],
                "accuracy": bo["accuracy"],
            }
        )

    out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not args.out_csv.exists()
    out.to_csv(args.out_csv, mode="a", header=header, index=False)
    print(f"Wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
