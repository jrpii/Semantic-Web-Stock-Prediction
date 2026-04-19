# Plot LSTM interval results (technical vs tech+news).
# Produces one chart for ALL rows and one for BREAKOUTS rows.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "analysis_outputs" / "reports"
VIZ_DIR = ROOT / "analysis_outputs" / "visualizations"


def _interval_from_dataset(s):
    # "5m_ALL" -> 5, "240m_BREAKOUTS" -> 240
    left = str(s).split("m_", 1)[0]
    try:
        return int(left)
    except Exception:
        return None


def _split_from_dataset(s):
    # "5m_ALL" -> "ALL"
    parts = str(s).split("m_", 1)
    return parts[1] if len(parts) == 2 else ""


def _load_csv(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "dataset" not in df.columns or "model" not in df.columns:
        return None
    return df


def _norm(df, group_name):
    # Keep only the columns we need and add interval/split.
    out = df.copy()
    out["interval_minutes"] = out["dataset"].map(_interval_from_dataset)
    out["split_tag"] = out["dataset"].map(_split_from_dataset)
    out["group"] = group_name
    return out


def _auto_ylim(values, *, pad=0.03, min_span=0.20):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 0.0, 1.0

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    # Add padding and enforce a minimum span so tiny differences are visible.
    vmin -= pad
    vmax += pad
    if (vmax - vmin) < min_span:
        center = 0.5 * (vmin + vmax)
        vmin = center - 0.5 * min_span
        vmax = center + 0.5 * min_span

    # Metrics here are 0..1.
    vmin = max(0.0, vmin)
    vmax = min(1.0, vmax)

    # Round to nice 0.01 boundaries.
    vmin = float(np.floor(vmin * 100.0) / 100.0)
    vmax = float(np.ceil(vmax * 100.0) / 100.0)
    if vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def _plot_side_by_side(df_all, split_tag, metric, out_path):
    df = df_all[df_all["split_tag"] == split_tag].copy()
    df = df[df["interval_minutes"].notna()].copy()
    df["interval_minutes"] = df["interval_minutes"].astype(int)

    intervals = sorted(df["interval_minutes"].unique().tolist())
    if not intervals:
        print(f"Nothing to plot for {split_tag}.")
        return

    groups = ["Technical", "Tech+News"]
    width = 0.38
    x = np.arange(len(intervals), dtype=float)

    plt.figure(figsize=(11, 4.8))
    all_y = []
    for i, g in enumerate(groups):
        y = []
        for iv in intervals:
            sub = df[(df["interval_minutes"] == iv) & (df["group"] == g)]
            if len(sub) == 0:
                y.append(np.nan)
                continue
            # If there are multiple rows, take the last one (should usually be one row anyway).
            y.append(float(sub[metric].iloc[-1]))
        all_y.extend(y)
        offset = (-width / 2) if i == 0 else (width / 2)
        plt.bar(x + offset, y, width=width, label=g)

    plt.xticks(x, [f"{iv}m" for iv in intervals])
    ymin, ymax = _auto_ylim(all_y)
    plt.ylim(ymin, ymax)
    plt.ylabel(metric)
    plt.title(f"LSTM results ({split_tag})")
    plt.grid(axis="y", alpha=0.25)
    span = ymax - ymin
    step = 0.02 if span <= 0.30 else 0.05
    ticks = np.arange(ymin, ymax + 1e-9, step)
    plt.yticks(ticks)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--tech-csv", type=Path, default=REPORTS_DIR / "lstm_interval_results.csv")
    parser.add_argument("--news-csv", type=Path, default=REPORTS_DIR / "lstm_news_interval_results.csv")
    parser.add_argument("--out-dir", type=Path, default=VIZ_DIR)
    args = parser.parse_args()

    tech_df = _load_csv(args.tech_csv)
    news_df = _load_csv(args.news_csv)
    if tech_df is None and news_df is None:
        raise FileNotFoundError("Could not find lstm_interval_results.csv or lstm_news_interval_results.csv")

    frames = []
    if tech_df is not None:
        frames.append(_norm(tech_df, "Technical"))
    if news_df is not None:
        frames.append(_norm(news_df, "Tech+News"))

    df_all = pd.concat(frames, ignore_index=True)
    if args.metric not in df_all.columns:
        raise ValueError(f"Metric not found: {args.metric}. Available: {list(df_all.columns)}")

    out_all = Path(args.out_dir) / f"lstm_{args.metric}_ALL.png"
    out_bo = Path(args.out_dir) / f"lstm_{args.metric}_BREAKOUTS.png"

    _plot_side_by_side(df_all, "ALL", args.metric, out_all)
    _plot_side_by_side(df_all, "BREAKOUTS", args.metric, out_bo)


if __name__ == "__main__":
    main()

