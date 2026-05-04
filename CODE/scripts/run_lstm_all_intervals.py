# Run LSTM training across all bar intervals and write a simple CSV report.
# Output CSV columns match train_evaluate_models.py: dataset, model, roc_auc, f1, precision, recall, accuracy
# Checkpoints are saved by train_lstm_ohlcv.py under EVALUATIONS/analysis_outputs/models/lstm/ or EVALUATIONS/analysis_outputs/models/lstm_news/.
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "CODE" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lstm_ohlcv as lstm

CSV_COLUMNS = ["dataset", "model", "roc_auc", "f1", "precision", "recall", "accuracy"]


def _row(dataset, model_name, metrics):
    return {
        "dataset": dataset,
        "model": model_name,
        "roc_auc": metrics["roc_auc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "accuracy": metrics["accuracy"],
    }


def _avg_metrics(metrics_list):
    if not metrics_list:
        return None
    keys = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    out = {}
    for k in keys:
        vals = [m.get(k, float("nan")) for m in metrics_list if isinstance(m, dict)]
        out[k] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def main():
    # Common knobs (kept intentionally small; see train_lstm_ohlcv.py for what each does).
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", nargs="+", type=int, default=[5, 15, 30, 60, 240, 1440])
    parser.add_argument("--out-csv", type=Path, default=ROOT / "EVALUATIONS" / "analysis_outputs" / "reports" / "lstm_interval_results.csv")
    parser.add_argument("--stocks-dir", type=Path, default=lstm.DEFAULT_STOCKS_DIR)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "AMZN"])
    parser.add_argument("--feature-set", choices=("ohlcv", "extended", "extended_news"), default="extended")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-layernorm", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-balance-loss", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--news-aligned", type=Path, default=lstm.DEFAULT_NEWS_ALIGNED)
    parser.add_argument("--no-news-merge", action="store_true")
    args = parser.parse_args()

    exclude = frozenset({"intervals", "out_csv", "trials"})
    rows = []

    for interval in args.intervals:
        print(f"\n--- Interval: {interval} minutes ---")
        model_label = "LSTM_NEWS" if args.feature_set == "extended_news" else "LSTM"

        all_trials = []
        bo_trials = []

        trials = max(int(args.trials), 1)
        for t in range(trials):
            if trials > 1:
                print(f"  trial {t + 1}/{trials}")

            run_cfg = {k: v for k, v in vars(args).items() if k not in exclude}
            run_cfg["interval"] = interval
            run_cfg["seed"] = int(args.seed) + t
            run_cfg["save_model"] = bool(trials <= 1)
            run_cfg["out_model"] = None
            run_cfg["balance_loss"] = not run_cfg["no_balance_loss"]
            run_args = argparse.Namespace(**run_cfg)

            report = lstm.run_training(run_args)
            all_trials.append(report["best_test_metrics_at_best_epoch"])

            bo = report.get("best_test_breakout_metrics_at_best_epoch")
            if isinstance(bo, dict):
                bo_trials.append(bo)

        all_avg = _avg_metrics(all_trials)
        if all_avg is not None:
            rows.append(_row(f"{interval}m_ALL", model_label, all_avg))

        bo_avg = _avg_metrics(bo_trials)
        if bo_avg is not None:
            rows.append(_row(f"{interval}m_BREAKOUTS", model_label, bo_avg))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
