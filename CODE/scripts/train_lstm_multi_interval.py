# Multi-interval LSTM: one shared model reads a sequence from each bar size (5m..daily),
# aligned to a target interval's bar time, and predicts next-bar direction for that target.
# Trains multi-task over all target intervals (optional per-epoch cyclic phases per target).
import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "CODE" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lstm_ohlcv as lstm

EVALUATIONS_DIR = ROOT / "EVALUATIONS" / "analysis_outputs"
DEFAULT_STOCKS_DIR = EVALUATIONS_DIR / "cleaned" / "stocks"
DEFAULT_NEWS_ALIGNED = EVALUATIONS_DIR / "cleaned" / "news" / "news_aligned_by_bar.csv"
DEFAULT_FINBERT_BY_BAR = EVALUATIONS_DIR / "cleaned" / "news" / "news_finbert_by_bar.csv"
MODELS_DIR = EVALUATIONS_DIR / "models"
LSTM_MULTI_MODEL_DIR = MODELS_DIR / "lstm_multi"

DEFAULT_INTERVALS = (5, 15, 30, 60, 240, 1440)


def bar_end_to_epoch_ns(ts) -> int:
    # Avoid np.datetime64(...) on tz-aware pandas timestamps (numpy warns).
    return int(pd.Timestamp(ts).value)


@dataclass
class StreamSample:
    streams: np.ndarray  # (n_streams, seq_len, n_feat)
    y: float
    target_idx: int
    ticker: str
    ts_ns: int  # bar end time, UTC epoch nanoseconds (for sorting / val split)
    breakout: int


# Model: one LSTM encoder per resolution stream, concat last hidden states + target embedding.
class StreamEncoder(nn.Module):
    # Per-resolution LSTM; returns last hidden state (not logits).
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.input_ln = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(x)
        x = self.input_dropout(x)
        out, _ = self.lstm(x)
        last = self.hidden_ln(out[:, -1, :])
        return last


class MultiStreamDirectionModel(nn.Module):
    # n_streams separate encoders; fuse with target-id embedding; single logit.
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_streams: int,
        n_targets: int,
        dropout: float,
        use_layernorm: bool,
        target_emb_dim: int,
    ) -> None:
        super().__init__()
        self.n_streams = n_streams
        self.encoders = nn.ModuleList(
            [
                StreamEncoder(input_dim, hidden_dim, num_layers, dropout, use_layernorm)
                for _ in range(n_streams)
            ]
        )
        self.target_emb = nn.Embedding(n_targets, target_emb_dim)
        fused = n_streams * hidden_dim + target_emb_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, 1))

    def forward(self, x: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_streams, seq, feat)
        h_parts = []
        for s in range(self.n_streams):
            h_parts.append(self.encoders[s](x[:, s, :, :]))
        h = torch.cat(h_parts, dim=-1)
        te = self.target_emb(target_ids)
        return self.head(torch.cat([h, te], dim=-1)).squeeze(-1)


def load_merged_frame(stocks_dir: Path, interval: int, tickers: list[str], args: argparse.Namespace) -> pd.DataFrame:
    df = lstm.load_frames(stocks_dir, interval, tickers)
    df = df[~pd.isna(df["target_up_next_bar"])].reset_index(drop=True)
    fs = args.feature_set
    if fs in (
        "extended_finbert",
        "extended_finbert_eng",
        "extended_news_finbert",
        "extended_news_finbert_eng",
    ):
        df = lstm.merge_finbert_into_df(df, Path(args.finbert_by_bar), interval)
        if fs in ("extended_news_finbert", "extended_news_finbert_eng"):
            df = lstm.merge_news_volume_cols(df, Path(args.news_aligned), interval)
    elif fs == "extended_news":
        df = lstm.merge_news_into_df(
            df, Path(args.news_aligned), interval, include_signal_columns=True
        )
    elif not args.no_news_merge:
        df = lstm.merge_news_into_df(
            df, Path(args.news_aligned), interval, include_signal_columns=False
        )
    return df


def prepare_stream_arrays(
    frames: dict[int, pd.DataFrame],
    tickers: list[str],
    intervals: tuple[int, ...],
    feature_set: str,
) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], list[str], int]:
    # Per ticker, per interval: (feature matrix, timestamps as int64 ns).
    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {t: {} for t in tickers}
    feat_names: list[str] | None = None
    n_feat = 0
    for iv in intervals:
        df = frames[iv]
        for ticker in tickers:
            chunk = df[df["ticker"].astype(str) == ticker].sort_values("timestamp_utc").reset_index(drop=True)
            if len(chunk) == 0:
                continue
            feats, names = lstm.build_feature_matrix(chunk, tickers, feature_set)
            if feat_names is None:
                feat_names = names
                n_feat = feats.shape[1]
            times_ns = pd.Index(chunk["timestamp_utc"]).asi8.astype(np.int64, copy=False)
            out[ticker][iv] = (feats.astype(np.float32), times_ns)
    if feat_names is None:
        raise ValueError("No data for multi-interval model (check tickers and files).")
    return out, feat_names, n_feat


def slice_stream_seq(
    arr: np.ndarray,
    times_ns: np.ndarray,
    ts_end_ns: int,
    seq_len: int,
) -> np.ndarray | None:
    idx = int(np.searchsorted(times_ns, ts_end_ns, side="right")) - 1
    if idx < 0:
        return None
    start = idx - seq_len + 1
    if start >= 0:
        return arr[start : idx + 1].copy()
    pad_n = -start
    body = arr[: idx + 1]
    pad = np.repeat(arr[0:1], pad_n, axis=0)
    return np.vstack([pad, body]).astype(np.float32)


def collect_samples_for_split(
    frames: dict[int, pd.DataFrame],
    stream_store: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    intervals: tuple[int, ...],
    tickers: list[str],
    seq_len: int,
    split_name: str,
    *,
    progress_label: str = "",
) -> list[StreamSample]:
    samples: list[StreamSample] = []
    iv_to_idx = {iv: i for i, iv in enumerate(intervals)}
    for target_iv in intervals:
        if progress_label:
            print(
                f"  [{progress_label}] collecting samples for target horizon {target_iv}m ...",
                flush=True,
            )
        df_t = frames[target_iv]
        for ticker in tickers:
            sub = df_t[df_t["ticker"].astype(str) == ticker].sort_values("timestamp_utc")
            sub = sub[~pd.isna(sub["target_up_next_bar"])].reset_index(drop=True)
            sp_ok = sub["split"].astype(str) == split_name
            sub = sub.loc[sp_ok].reset_index(drop=True)
            if len(sub) == 0:
                continue
            has_br = "news_breakout" in sub.columns
            for i in range(len(sub)):
                ts = sub["timestamp_utc"].iloc[i]
                ts_ns = bar_end_to_epoch_ns(ts)
                y = float(sub["target_up_next_bar"].iloc[i])
                if np.isnan(y):
                    continue
                br = int(sub["news_breakout"].iloc[i]) if has_br else 0
                streams_list: list[np.ndarray] = []
                ok = True
                for iv in intervals:
                    if iv not in stream_store.get(ticker, {}):
                        ok = False
                        break
                    arr, times_ns = stream_store[ticker][iv]
                    seq = slice_stream_seq(arr, times_ns, ts_ns, seq_len)
                    if seq is None:
                        ok = False
                        break
                    if seq.shape[0] != seq_len:
                        ok = False
                        break
                    streams_list.append(seq)
                if not ok or len(streams_list) != len(intervals):
                    continue
                stacked = np.stack(streams_list, axis=0)
                samples.append(
                    StreamSample(
                        streams=stacked,
                        y=y,
                        target_idx=iv_to_idx[target_iv],
                        ticker=str(ticker),
                        ts_ns=ts_ns,
                        breakout=br,
                    )
                )
    return samples


def train_val_split_samples(
    samples: list[StreamSample],
    val_fraction: float,
    intervals: tuple[int, ...],
) -> tuple[list[StreamSample], list[StreamSample]]:
    if val_fraction <= 0 or not samples:
        return samples, []
    by_key: dict[tuple[str, int], list[StreamSample]] = {}
    for s in samples:
        key = (s.ticker, intervals[s.target_idx])
        by_key.setdefault(key, []).append(s)
    fit: list[StreamSample] = []
    val: list[StreamSample] = []
    for key, group in by_key.items():
        group.sort(key=lambda x: x.ts_ns)
        n = len(group)
        nv = max(1, int(n * val_fraction))
        nv = min(nv, n - 1)
        if nv <= 0 or n - nv < 1:
            fit.extend(group)
            continue
        fit.extend(group[:-nv])
        val.extend(group[-nv:])
    return fit, val


def samples_to_arrays(samples: list[StreamSample], n_streams: int, seq_len: int, n_feat: int):
    if not samples:
        return (
            np.empty((0, n_streams, seq_len, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    X = np.stack([s.streams for s in samples], axis=0).astype(np.float32)
    y = np.asarray([s.y for s in samples], dtype=np.float32)
    tid = np.asarray([s.target_idx for s in samples], dtype=np.int64)
    br = np.asarray([s.breakout for s in samples], dtype=np.int64)
    return X, y, tid, br


def scale_multi_fit_transform(
    X_fit: np.ndarray,
    X_val: np.ndarray | None,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, StandardScaler]:
    # Fit scaler on flattened (all streams and time steps) train tensor.
    ns, n_str, t, f = X_fit.shape
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, f))
    Xf = scaler.transform(X_fit.reshape(-1, f)).reshape(ns, n_str, t, f).astype(np.float32)
    if X_val is not None and X_val.size and X_val.shape[0] > 0:
        nv = X_val.shape[0]
        Xv = scaler.transform(X_val.reshape(-1, f)).reshape(nv, n_str, t, f).astype(np.float32)
    else:
        Xv = None
    if X_test.size == 0:
        Xt = X_test
    else:
        nt = X_test.shape[0]
        Xt = scaler.transform(X_test.reshape(-1, f)).reshape(nt, n_str, t, f).astype(np.float32)
    return Xf, Xv, Xt, scaler


def make_bce_loss(y_fit: np.ndarray, device: torch.device, balance: bool) -> nn.Module:
    if not balance:
        return nn.BCEWithLogitsLoss()
    pos = float(np.sum(y_fit))
    neg = float(len(y_fit) - pos)
    pw = neg / max(pos, 1.0)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))


def train_one_epoch_mixed(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    tid: torch.Tensor,
    batch_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    rng: np.random.Generator,
) -> float:
    model.train()
    n = X.shape[0]
    order = rng.permutation(n)
    total_loss = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        xb = X[idx].to(device)
        yb = y[idx].to(device)
        tb = tid[idx].to(device)
        optimizer.zero_grad()
        logits = model(xb, tb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(idx)
        seen += len(idx)
    return total_loss / max(seen, 1)


def train_one_epoch_cyclic(
    model: nn.Module,
    tensors_by_target: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    rng: np.random.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for target_idx, (Xt, yt) in enumerate(tensors_by_target):
        if Xt.shape[0] == 0:
            continue
        n = Xt.shape[0]
        order = rng.permutation(n)
        t_ids = torch.full((n,), target_idx, dtype=torch.long, device=device)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            tb = t_ids[idx]
            optimizer.zero_grad()
            logits = model(xb, tb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)
    return total_loss / max(seen, 1)


def predict_probs_batched_multi(
    model: nn.Module,
    X: np.ndarray,
    tid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs_out: list[np.ndarray] = []
    with torch.no_grad():
        n = X.shape[0]
        for start in range(0, n, batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            tb = torch.from_numpy(tid[start : start + batch_size]).long().to(device)
            logits = model(xb, tb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_out.append(probs)
    return np.concatenate(probs_out, axis=0)


def metrics_subset(y_true: np.ndarray, probs: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    if mask is not None:
        y_true = y_true[mask]
        probs = probs[mask]
    if len(y_true) == 0:
        return {
            "roc_auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "accuracy": float("nan"),
        }
    preds = (probs >= 0.5).astype(np.int64)
    yt = y_true.astype(np.int64)
    out = {
        "roc_auc": float("nan"),
        "f1": float(f1_score(yt, preds, zero_division=0)),
        "precision": float(precision_score(yt, preds, zero_division=0)),
        "recall": float(recall_score(yt, preds, zero_division=0)),
        "accuracy": float(accuracy_score(yt, preds)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(yt, probs))
    except ValueError:
        pass
    return out


def mean_auc(metrics_list: list[dict[str, float]]) -> float:
    aucs = [m["roc_auc"] for m in metrics_list if not np.isnan(m.get("roc_auc", np.nan))]
    return float(np.mean(aucs)) if aucs else float("nan")


def format_per_target_epoch_line(
    intervals: tuple[int, ...],
    per_metrics: list[dict[str, float]],
    counts: np.ndarray,
) -> str:
    parts = []
    for iv, m, n in zip(intervals, per_metrics, counts):
        auc = m["roc_auc"]
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "nan"
        parts.append(f"{iv}m auc={auc_s} acc={m['accuracy']:.3f} n={int(n)}")
    return "    " + " | ".join(parts)


def metric_is_better(candidate: float, best: float | None) -> bool:
    if best is None:
        return True
    if np.isnan(candidate) and np.isnan(best):
        return False
    if np.isnan(candidate):
        return False
    if np.isnan(best):
        return True
    return candidate > best


def run_training(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    intervals = tuple(sorted(set(int(x) for x in args.intervals)))
    tickers = list(args.tickers)

    # Load and index each interval.
    print("Loading merged OHLCV (+ optional news / FinBERT) per interval...", flush=True)
    frames = {}
    for iv in intervals:
        frames[iv] = load_merged_frame(Path(args.stocks_dir), iv, tickers, args)

    print("Building feature matrices per ticker × interval...", flush=True)
    stream_store, feat_names, n_feat = prepare_stream_arrays(frames, tickers, intervals, args.feature_set)

    print(
        "Building aligned sequences (slow on 5m: every bar × each target horizon)...",
        flush=True,
    )
    train_samples = collect_samples_for_split(
        frames,
        stream_store,
        intervals,
        tickers,
        args.seq_len,
        "train",
        progress_label="train",
    )
    print(f"  train raw samples: {len(train_samples)}", flush=True)
    test_samples = collect_samples_for_split(
        frames,
        stream_store,
        intervals,
        tickers,
        args.seq_len,
        "test",
        progress_label="test",
    )
    print(f"  test raw samples: {len(test_samples)}", flush=True)
    train_fit, train_val = train_val_split_samples(train_samples, args.val_fraction, intervals)

    n_streams = len(intervals)
    X_fit, y_fit, tid_fit, _ = samples_to_arrays(train_fit, n_streams, args.seq_len, n_feat)
    X_val, y_val, tid_val, _ = samples_to_arrays(train_val, n_streams, args.seq_len, n_feat)
    X_test, y_test, tid_test, br_test = samples_to_arrays(test_samples, n_streams, args.seq_len, n_feat)

    if len(X_fit) == 0 or len(X_test) == 0:
        raise ValueError("Train or test empty; check intervals, seq_len, and data alignment.")

    X_val_in = X_val if len(train_val) else None
    X_fit_s, X_val_s, X_test_s, _ = scale_multi_fit_transform(X_fit, X_val_in, X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_layernorm = not args.no_layernorm
    model = MultiStreamDirectionModel(
        input_dim=n_feat,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_streams=n_streams,
        n_targets=n_streams,
        dropout=args.dropout,
        use_layernorm=use_layernorm,
        target_emb_dim=args.target_emb_dim,
    ).to(device)

    criterion = make_bce_loss(y_fit, device, args.balance_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X_fit_t = torch.from_numpy(X_fit_s)
    y_fit_t = torch.from_numpy(y_fit)
    tid_fit_t = torch.from_numpy(tid_fit)
    select_val = X_val_s is not None and len(train_val) > 0

    tensors_by_target: list[tuple[torch.Tensor, torch.Tensor]] = []
    for tix in range(n_streams):
        m = tid_fit_t == tix
        tensors_by_target.append((X_fit_t[m], y_fit_t[m]))

    # Epoch loop, targeting AUC.
    print("")
    print("=" * 72)
    print("Multi-interval LSTM")
    print("=" * 72)
    print(f"  device:           {device}")
    print(f"  intervals:      {intervals}")
    print(f"  feature_set:    {args.feature_set}")
    print(f"  seq_len:          {args.seq_len}")
    print(f"  features:       {n_feat}  {feat_names[:6]}{'...' if len(feat_names) > 6 else ''}")
    print(f"  train samples:    {len(X_fit)}  val: {len(train_val)}  test: {len(X_test)}")
    print(f"  hidden / layers:  {args.hidden_dim} / {args.num_layers}")
    print(f"  target_emb_dim:   {args.target_emb_dim}")
    print(f"  cyclic_phases:    {args.cyclic_phases}")
    print(f"  val_fraction:     {args.val_fraction}")
    print(f"  select_best_by:   {'val mean AUC' if select_val else 'test mean AUC (per-target then averaged)'}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters:       {n_params:,}")
    print("  --- how to read metrics ---", flush=True)
    print(
        "    One label per sample: target_up_next_bar for the TARGET horizon only.",
        flush=True,
    )
    print(
        "    Inputs: seq_len steps from each interval, aligned to that target bar's end (UTC).",
        flush=True,
    )
    print(
        "    target_idx + target embedding tell the head which horizon to score; loss is BCE on that label.",
        flush=True,
    )
    print(
        "    Per-interval line: ROC-AUC and accuracy on TEST rows for that target only; n = count of those rows.",
        flush=True,
    )
    print(
        "    test_mean_auc: unweighted mean of the six per-interval ROC-AUCs.",
        flush=True,
    )
    print("=" * 72)
    print("")

    test_counts = np.bincount(tid_test.astype(np.int64), minlength=n_streams)
    quiet_epochs = getattr(args, "quiet_epochs", False)

    best_mean_sel = None
    best_state = None
    best_epoch = -1
    last_epoch = 0
    no_improve = 0
    patience = args.early_stopping_patience

    for epoch in range(1, args.epochs + 1):
        if args.cyclic_phases:
            loss = train_one_epoch_cyclic(
                model,
                tensors_by_target,
                args.batch_size,
                criterion,
                optimizer,
                device,
                args.grad_clip,
                rng,
            )
        else:
            loss = train_one_epoch_mixed(
                model,
                X_fit_t,
                y_fit_t,
                tid_fit_t,
                args.batch_size,
                criterion,
                optimizer,
                device,
                args.grad_clip,
                rng,
            )

        probs_test = predict_probs_batched_multi(model, X_test_s, tid_test, device, args.batch_size)
        per_test = []
        for tix, iv in enumerate(intervals):
            mask = tid_test == tix
            per_test.append(metrics_subset(y_test, probs_test, mask))
        mean_te = mean_auc(per_test)

        if select_val:
            tid_val_np = tid_val
            probs_val = predict_probs_batched_multi(model, X_val_s, tid_val_np, device, args.batch_size)
            per_val = []
            for tix in range(n_streams):
                mask = tid_val_np == tix
                per_val.append(metrics_subset(y_val, probs_val, mask))
            mean_sel = mean_auc(per_val)
            if quiet_epochs:
                print(
                    f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                    f"val_mean_auc={mean_sel:.4f} | test_mean_auc={mean_te:.4f}"
                )
            else:
                print(
                    f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                    f"val_mean_auc={mean_sel:.4f} | test_mean_auc={mean_te:.4f}"
                )
                print(format_per_target_epoch_line(intervals, per_test, test_counts))
            key = mean_sel
        else:
            if quiet_epochs:
                print(
                    f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                    f"test_mean_auc={mean_te:.4f}"
                )
            else:
                print(
                    f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                    f"test_mean_auc={mean_te:.4f}"
                )
                print(format_per_target_epoch_line(intervals, per_test, test_counts))
            key = mean_te

        if metric_is_better(key, best_mean_sel):
            best_mean_sel = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        last_epoch = epoch

        if patience > 0 and no_improve >= patience:
            print(f"\n[early stopping] no improvement for {patience} epochs (at {epoch})\n")
            break

    # Evaluation on best weights.
    assert best_state is not None
    model.load_state_dict(best_state)

    probs_test = predict_probs_batched_multi(model, X_test_s, tid_test, device, args.batch_size)
    print("-" * 72)
    print(f"Best epoch: {best_epoch} / {last_epoch} (mean selection AUC={best_mean_sel:.4f})")
    per_report = []
    for tix, iv in enumerate(intervals):
        mask = tid_test == tix
        m = metrics_subset(y_test, probs_test, mask)
        per_report.append((iv, m))
        print(f"  target {iv}m test: {lstm.format_metrics(m)}  (n={int(mask.sum())})")
    bo_report = None
    if br_test is not None and np.any(br_test):
        m_bo = metrics_subset(y_test, probs_test, br_test.astype(bool))
        print(f"  test BREAKOUTS (all targets): {lstm.format_metrics(m_bo)}  (n={int(br_test.sum())})")
        bo_report = m_bo
    print("-" * 72)

    LSTM_MULTI_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        out_path = Path(args.out_model) if args.out_model else LSTM_MULTI_MODEL_DIR / f"multi_LSTM_epoch_{best_epoch}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "intervals": list(intervals),
            "tickers": tickers,
            "feature_set": args.feature_set,
            "seq_len": args.seq_len,
            "feature_names": feat_names,
            "n_feat": n_feat,
            "best_epoch": best_epoch,
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "target_emb_dim": args.target_emb_dim,
                "layernorm": use_layernorm,
            },
        }
        torch.save(payload, out_path)
        print(f"Saved checkpoint to {out_path}")

    return {
        "intervals": list(intervals),
        "feature_names": feat_names,
        "best_epoch": best_epoch,
        "selection_mean_auc": best_mean_sel,
        "per_target_test": {str(iv): m for iv, m in per_report},
        "test_breakout_all_targets": bo_report,
        "train_samples": len(X_fit),
        "test_samples": len(X_test),
    }


def main():
    # Defaults favor most prommising LSTM run configs.
    p = argparse.ArgumentParser()
    p.add_argument("--stocks-dir", type=Path, default=DEFAULT_STOCKS_DIR)
    p.add_argument("--intervals", nargs="+", type=int, default=list(DEFAULT_INTERVALS))
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
    p.add_argument("--quiet-epochs", action="store_true")
    p.add_argument("--cyclic-phases", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--out-model", type=Path, default=None)
    p.add_argument("--news-aligned", type=Path, default=DEFAULT_NEWS_ALIGNED)
    p.add_argument("--finbert-by-bar", type=Path, default=DEFAULT_FINBERT_BY_BAR)
    p.add_argument("--no-news-merge", action="store_true")
    args = p.parse_args()
    args.balance_loss = not args.no_balance_loss
    run_training(args)


if __name__ == "__main__":
    main()
