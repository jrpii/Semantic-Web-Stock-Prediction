# Plot FinBERT-LSTM interval results CSVs.
# Makes one chart for ALL and one for BREAKOUTS, with grouped bars per interval.
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "analysis_outputs" / "reports"
VIZ_DIR = ROOT / "analysis_outputs" / "visualizations"

def _interval_from_dataset(s):
    left = str(s).split("m_", 1)[0]
    try:
        return int(left)
    except Exception:
        return None


def _split_from_dataset(s):
    parts = str(s).split("m_", 1)
    return parts[1] if len(parts) == 2 else ""


def _auto_ylim(values, *, pad=0.03, min_span=0.20):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 0.0, 1.0

    vmin = float(np.min(vals)) - float(pad)
    vmax = float(np.max(vals)) + float(pad)
    if (vmax - vmin) < float(min_span):
        center = 0.5 * (vmin + vmax)
        vmin = center - 0.5 * float(min_span)
        vmax = center + 0.5 * float(min_span)

    vmin = max(0.0, vmin)
    vmax = min(1.0, vmax)
    vmin = float(np.floor(vmin * 100.0) / 100.0)
    vmax = float(np.ceil(vmax * 100.0) / 100.0)
    if vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def _label_from_path(p):
    stem = Path(p).stem
    # Examples:
    # - lstm_finbert_interval_results_En -> En
    # - lstm_finbert_interval_results_ExNeEn -> ExNeEn
    prefix = "lstm_finbert_interval_results_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    if stem.startswith("lstm_finbert_interval_results"):
        return stem.replace("lstm_finbert_interval_results", "").lstrip("_-") or stem
    return stem


def _pretty_label(raw):
    # Optional nice labels for your current naming.
    mapping = {
        "En": "FinBERT Eng",
        "ExNe": "NewsVol + FinBERT",
        "ExNeEn": "NewsVol + FinBERT Eng",
    }
    return mapping.get(raw, raw)


def _load_one(csv_path):
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError(f"Missing dataset column in {csv_path}")
    out = df.copy()
    out["interval_minutes"] = out["dataset"].map(_interval_from_dataset)
    out["split_tag"] = out["dataset"].map(_split_from_dataset)
    return out


def _plot(df_all, split_tag, metric, labels, out_path):
    df = df_all[df_all["split_tag"] == split_tag].copy()
    df = df[df["interval_minutes"].notna()].copy()
    df["interval_minutes"] = df["interval_minutes"].astype(int)

    intervals = sorted(df["interval_minutes"].unique().tolist())
    if not intervals:
        print(f"Nothing to plot for {split_tag}.")
        return

    n = max(len(labels), 1)
    width = min(0.80 / n, 0.22)
    x = np.arange(len(intervals), dtype=float)

    plt.figure(figsize=(11.5, 5.0))
    all_y = []

    for i, label in enumerate(labels):
        y = []
        for iv in intervals:
            sub = df[(df["interval_minutes"] == iv) & (df["run_label"] == label)]
            if len(sub) == 0:
                y.append(np.nan)
            else:
                y.append(float(sub[metric].iloc[-1]))
        all_y.extend(y)

        offset = (i - (n - 1) / 2.0) * width
        plt.bar(x + offset, y, width=width, label=_pretty_label(label))

    plt.xticks(x, [f"{iv}m" for iv in intervals])
    ymin, ymax = _auto_ylim(all_y)
    plt.ylim(ymin, ymax)
    span = ymax - ymin
    step = 0.02 if span <= 0.30 else 0.05
    ticks = np.arange(ymin, ymax + 1e-9, step)
    plt.yticks(ticks)

    plt.ylabel(metric)
    plt.title(f"LSTM FinBERT comparison ({split_tag})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--out-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--csv", nargs="*", type=Path, default=None)
    args = parser.parse_args()

    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
    else:
        csv_paths = sorted(REPORTS_DIR.glob("lstm_finbert_interval_results*.csv"))

    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        raise FileNotFoundError("No finbert results CSVs found in analysis_outputs/reports/")

    frames = []
    labels = []
    for p in csv_paths:
        label = _label_from_path(p)
        labels.append(label)
        df = _load_one(p)
        df["run_label"] = label
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    if args.metric not in df_all.columns:
        raise ValueError(f"Metric not found: {args.metric}. Available: {list(df_all.columns)}")

    out_all = Path(args.out_dir) / f"lstm_finbert_{args.metric}_ALL.png"
    out_bo = Path(args.out_dir) / f"lstm_finbert_{args.metric}_BREAKOUTS.png"

    _plot(df_all, "ALL", args.metric, labels, out_all)
    _plot(df_all, "BREAKOUTS", args.metric, labels, out_bo)


if __name__ == "__main__":
    main()

