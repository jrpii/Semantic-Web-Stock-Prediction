# Train an LSTM to predict next-bar direction (up/down).
# Uses the chronological train/test split created in clean_and_eda.py (the `split` column).
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STOCKS_DIR = ROOT / "analysis_outputs" / "cleaned" / "stocks"
DEFAULT_NEWS_ALIGNED = ROOT / "analysis_outputs" / "cleaned" / "news" / "news_aligned_by_bar.csv"
MODELS_DIR = ROOT / "analysis_outputs" / "models"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
LSTM_NEWS_MODEL_DIR = MODELS_DIR / "lstm_news"


class DirectionalLSTM(nn.Module):
    # Simple LSTM classifier for binary direction.

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
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(x)
        x = self.input_dropout(x)
        out, _ = self.lstm(x)
        last = self.hidden_ln(out[:, -1, :])
        return self.head(last).squeeze(-1)


def lstm_from_args(args: argparse.Namespace, input_dim: int) -> "DirectionalLSTM":
    return DirectionalLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layernorm=not args.no_layernorm,
    )


def merge_news_into_df(
    df: pd.DataFrame,
    news_path: Path,
    interval: int,
    *,
    include_signal_columns: bool,
) -> pd.DataFrame:
    # Merge aligned news onto stock bars and compute a simple "breakout" flag:
    # article_count >= rolling_mean(20) + 2*rolling_std(20), and article_count > 0.
    # If include_signal_columns=True, also bring in the extra news columns and build the two
    # interaction features used in train_evaluate_models.py.
    def _breakout_only_frame() -> pd.DataFrame:
        o = df.copy()
        o["article_count"] = 0.0
        o["news_breakout"] = np.int64(0)
        if include_signal_columns:
            o["weighted_article_count"] = 0.0
            o["avg_word_count"] = 0.0
            o["unique_sites"] = 0.0
            o["news_volatility_interaction"] = 0.0
            o["news_momentum"] = 0.0
        return o

    if not news_path.exists():
        return _breakout_only_frame()

    news = pd.read_csv(news_path)
    news["timestamp_utc"] = pd.to_datetime(news["aligned_bar_utc"], utc=True)
    base_cols = ["ticker", "interval_minutes", "timestamp_utc", "article_count"]
    if include_signal_columns:
        base_cols.extend(["weighted_article_count", "avg_word_count", "unique_sites"])
    sub = news.loc[news["interval_minutes"] == interval, base_cols].copy()

    out = df.merge(sub, on=["ticker", "interval_minutes", "timestamp_utc"], how="left")
    out["article_count"] = pd.to_numeric(out["article_count"], errors="coerce").fillna(0.0)
    if include_signal_columns:
        for col in ("weighted_article_count", "avg_word_count", "unique_sites"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out.sort_values(["ticker", "timestamp_utc"]).reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
    for _, g in out.groupby("ticker", sort=False):
        ac = g["article_count"].astype(float)
        rm = ac.rolling(20, min_periods=1).mean()
        rs = ac.rolling(20, min_periods=1).std().fillna(0.0)
        br = ((ac >= rm + 2 * rs) & (ac > 0)).astype(np.int64)
        gc = g.copy()
        gc["news_breakout"] = br.values
        if include_signal_columns:
            vola = pd.to_numeric(gc["volatility_10"], errors="coerce").fillna(0.0)
            ret = pd.to_numeric(gc["return_pct"], errors="coerce").fillna(0.0)
            w = pd.to_numeric(gc["weighted_article_count"], errors="coerce").fillna(0.0)
            acount = pd.to_numeric(gc["article_count"], errors="coerce").fillna(0.0)
            gc["news_volatility_interaction"] = (w * vola).astype(np.float64)
            gc["news_momentum"] = (acount * ret).astype(np.float64)
        chunks.append(gc)
    return pd.concat(chunks, ignore_index=True)


def merge_news_for_breakout(df: pd.DataFrame, news_path: Path, interval: int) -> pd.DataFrame:
    # Backward-compatible wrapper: breakout mask + article_count only (no extra news signals).
    return merge_news_into_df(df, news_path, interval, include_signal_columns=False)


def load_frames(stocks_dir: Path, interval: int, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = stocks_dir / f"{ticker}_{interval}m_cleaned.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cleaned stock file: {path}")
        df = pd.read_csv(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "timestamp_utc"]).reset_index(drop=True)
    return out


def ticker_one_hot_series(df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    ticker_to_idx = {t: i for i, t in enumerate(sorted(tickers))}
    mat = np.zeros((len(df), len(tickers)), dtype=np.float32)
    for i, row in enumerate(df["ticker"].astype(str)):
        idx = ticker_to_idx.get(row)
        if idx is not None:
            mat[i, idx] = 1.0
    return mat


def build_feature_matrix(
    df: pd.DataFrame,
    tickers: list[str],
    feature_set: str = "ohlcv",
) -> tuple[np.ndarray, list[str]]:
    ohlcv = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    vol_log = np.log1p(np.maximum(ohlcv[:, 4], 0.0)).reshape(-1, 1)
    base = np.hstack([ohlcv[:, :4], vol_log]).astype(np.float32)
    tick_oh = ticker_one_hot_series(df, tickers)
    names = ["open", "high", "low", "close", "log1p_volume"] + [f"ticker_{t}" for t in sorted(tickers)]

    if feature_set == "ohlcv":
        X = np.hstack([base, tick_oh]).astype(np.float32)
        return X, names

    if feature_set not in ("extended", "extended_news"):
        raise ValueError(f"Unknown feature_set: {feature_set}")

    close = np.maximum(df["close"].to_numpy(dtype=np.float64), 1e-9)
    ret = pd.to_numeric(df["return_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    vola = pd.to_numeric(df["volatility_10"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    rsi = pd.to_numeric(df["rsi_14"], errors="coerce").fillna(50.0).to_numpy(dtype=np.float64) / 100.0
    pr = pd.to_numeric(df["price_range"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    hl_pct = (pr / close).astype(np.float64)
    extra = np.stack([ret, vola, rsi, hl_pct], axis=1).astype(np.float32)

    if feature_set == "extended_news":
        ac = pd.to_numeric(df["article_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        wac = pd.to_numeric(df["weighted_article_count"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )
        awc = pd.to_numeric(df["avg_word_count"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )
        us = pd.to_numeric(df["unique_sites"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        nvi = pd.to_numeric(df["news_volatility_interaction"], errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float64
        )
        nm = pd.to_numeric(df["news_momentum"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        news_blk = np.stack([ac, wac, awc, us, nvi, nm], axis=1).astype(np.float32)
        nm_news = names + ["return_pct", "volatility_10", "rsi_norm", "hl_pct"] + [
            "article_count",
            "weighted_article_count",
            "avg_word_count",
            "unique_sites",
            "news_volatility_interaction",
            "news_momentum",
        ]
        X = np.hstack([base, extra, tick_oh, news_blk]).astype(np.float32)
        return X, nm_news

    X = np.hstack([base, extra, tick_oh]).astype(np.float32)
    return X, names + ["return_pct", "volatility_10", "rsi_norm", "hl_pct"]


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for i in range(seq_len - 1, len(features)):
        x_list.append(features[i - seq_len + 1 : i + 1])
        y_list.append(labels[i])
    if not x_list:
        return np.empty((0, seq_len, features.shape[1])), np.empty((0,))
    return np.stack(x_list, axis=0), np.asarray(y_list, dtype=np.float32)


def scale_sequences_fit_transform(
    X_fit: np.ndarray,
    X_val: np.ndarray | None,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, StandardScaler]:
    # Fit StandardScaler on X_fit only; transform train/val/test sequences.
    n_fit, seq_len, n_feat = X_fit.shape
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, n_feat))
    Xf = scaler.transform(X_fit.reshape(-1, n_feat)).reshape(n_fit, seq_len, n_feat).astype(np.float32)
    if X_val is not None and X_val.size and X_val.shape[0] > 0:
        nv, _, _ = X_val.shape
        Xv = scaler.transform(X_val.reshape(-1, n_feat)).reshape(nv, seq_len, n_feat).astype(np.float32)
    else:
        Xv = None
    if X_test.size == 0:
        Xt = X_test
    else:
        nt, _, _ = X_test.shape
        Xt = scaler.transform(X_test.reshape(-1, n_feat)).reshape(nt, seq_len, n_feat).astype(np.float32)
    return Xf, Xv, Xt, scaler

def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    # Same metric set as train_evaluate_models.py.
    out: dict[str, float] = {
        "roc_auc": float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        pass
    return out


def predict_probs_batched(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_out.append(probs)
    return np.concatenate(probs_out, axis=0)


def metrics_for_split(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    probs = predict_probs_batched(model, X, device, batch_size)
    preds = (probs >= 0.5).astype(np.int64)
    return evaluate_probs(y.astype(np.int64), probs, preds)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


def make_bce_loss(y_fit: np.ndarray, device: torch.device, balance: bool) -> nn.Module:
    if not balance:
        return nn.BCEWithLogitsLoss()
    pos = float(np.sum(y_fit))
    neg = float(len(y_fit) - pos)
    pw = neg / max(pos, 1.0)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))


def metric_is_better(candidate: dict[str, float], current_best: dict[str, float] | None) -> bool:
    if current_best is None:
        return True
    c_auc, b_auc = candidate["roc_auc"], current_best["roc_auc"]
    if not np.isnan(c_auc) and not np.isnan(b_auc):
        return c_auc > b_auc
    if np.isnan(c_auc) and np.isnan(b_auc):
        if candidate["accuracy"] != current_best["accuracy"]:
            return candidate["accuracy"] > current_best["accuracy"]
        return candidate["f1"] > current_best["f1"]
    return not np.isnan(c_auc)


def format_metrics(m: dict[str, float]) -> str:
    auc = m["roc_auc"]
    auc_s = f"{auc:.4f}" if not np.isnan(auc) else "nan"
    return (
        f"acc={m['accuracy']:.4f} p={m['precision']:.4f} r={m['recall']:.4f} "
        f"f1={m['f1']:.4f} auc={auc_s}"
    )


def print_run_config(
    args: argparse.Namespace,
    device: torch.device,
    feat_names: list[str],
    n_train_seq: int,
    n_val: int,
    n_test: int,
) -> None:
    sample = np.zeros((1, args.seq_len, len(feat_names)), dtype=np.float32)
    model_for_count = lstm_from_args(args, sample.shape[2])
    n_params = sum(p.numel() for p in model_for_count.parameters())
    if args.val_fraction > 0:
        sel = "val ROC-AUC"
    else:
        sel = "test ROC-AUC (no val split)"
    es = (
        f"{args.early_stopping_patience} epochs"
        if getattr(args, "early_stopping_patience", 0) > 0
        else "off"
    )
    lines = [
        "",
        "=" * 69,
        "LSTM — run configuration",
        "=" * 69,
        f"  device:              {device}",
        f"  feature_set:         {args.feature_set}",
        f"  stocks_dir:          {args.stocks_dir}",
        f"  interval (minutes):  {args.interval}",
        f"  tickers:             {list(args.tickers)}",
        f"  seq_len:             {args.seq_len}",
        f"  features ({len(feat_names)}): {feat_names}",
        f"  hidden_dim:          {args.hidden_dim}",
        f"  num_layers:          {args.num_layers}",
        f"  dropout:             {args.dropout}",
        f"  layer_norm:          {not args.no_layernorm}",
        "  feature_normalize:   StandardScaler (fit on train seq. only)",
        f"  epochs:              {args.epochs}",
        f"  batch_size:          {args.batch_size}",
        f"  lr:                  {args.lr}",
        f"  weight_decay:        {args.weight_decay}",
        f"  grad_clip:           {args.grad_clip}",
        f"  balance_loss:        {args.balance_loss}",
        f"  val_fraction:        {args.val_fraction}",
        f"  early_stopping:      {es}",
        f"  select_best_by:      {sel}",
        f"  sequences train/val/test: {n_train_seq} / {n_val} / {n_test}",
        (
            "    (train = rows used for SGD after val carve-out; val = chronological tail)"
            if n_val > 0
            else "    (train = all train-split sequences; no val carve-out)"
        ),
        f"  approx. parameters:  {n_params:,}",
        "=" * 69,
        "",
    ]
    print("\n".join(lines))


def train_full_logging(
    X_fit_s: np.ndarray,
    y_fit: np.ndarray,
    X_val_s: np.ndarray | None,
    y_val: np.ndarray | None,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    test_breakout: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
    feat_names: list[str],
    n_val_raw: int,
) -> tuple[
    dict[str, float],
    dict[str, float] | None,
    dict[str, float],
    dict[str, float],
    dict[str, float] | None,
    int,
    int,
    DirectionalLSTM,
]:
    # X_fit_s / y_fit are the sequences used for SGD (train split, minus optional val tail).
    # We keep metrics per epoch and remember the best model state (by val AUC if val exists,
    # otherwise by test AUC).
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_fit_s),
        torch.from_numpy(y_fit),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = lstm_from_args(args, X_fit_s.shape[2]).to(device)
    criterion = make_bce_loss(y_fit, device, args.balance_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print_run_config(
        args,
        device,
        feat_names,
        len(X_fit_s),
        n_val_raw,
        len(X_test_s),
    )

    select_val = (
        X_val_s is not None
        and y_val is not None
        and len(X_val_s) > 0
    )

    best_select_metrics: dict[str, float] | None = None
    best_train_at_best: dict[str, float] | None = None
    best_val_at_best: dict[str, float] | None = None
    best_test_at_best: dict[str, float] | None = None
    best_epoch = -1
    best_state: dict | None = None
    last_epoch_ran = 0
    no_improve = 0
    patience = getattr(args, "early_stopping_patience", 0)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        mt = metrics_for_split(model, X_fit_s, y_fit, device, args.batch_size)
        mte = metrics_for_split(model, X_test_s, y_test, device, args.batch_size)

        mv: dict[str, float] | None = None
        if select_val:
            mv = metrics_for_split(model, X_val_s, y_val, device, args.batch_size)  # type: ignore[arg-type]
            print(
                f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                f"train {format_metrics(mt)} | "
                f"val  {format_metrics(mv)} | "
                f"test {format_metrics(mte)}"
            )
            key = mv
        else:
            print(
                f"[epoch {epoch:4d}/{args.epochs}] loss={loss:.6f} | "
                f"train {format_metrics(mt)} | "
                f"test {format_metrics(mte)}  (checkpoint by test ROC-AUC; no val)"
            )
            key = mte

        improved = metric_is_better(key, best_select_metrics)
        if improved:
            best_select_metrics = copy.deepcopy(key)
            best_train_at_best = copy.deepcopy(mt)
            best_val_at_best = copy.deepcopy(mv) if select_val else None
            best_test_at_best = copy.deepcopy(mte)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        last_epoch_ran = epoch

        if patience > 0:
            if not improved:
                no_improve += 1
                if no_improve >= patience:
                    crit_es = "val" if select_val else "test"
                    print(
                        f"\n[early stopping] {crit_es} did not improve vs best for {patience} epochs "
                        f"(stopped at epoch {epoch} / {args.epochs})\n"
                    )
                    break
            else:
                no_improve = 0

    assert (
        best_select_metrics is not None
        and best_train_at_best is not None
        and best_test_at_best is not None
        and best_state is not None
    )

    print("")
    print("-" * 72)
    crit = "val ROC-AUC" if select_val else "test ROC-AUC"
    print(f"Best epoch by {crit}: {best_epoch} / {args.epochs}", end="")
    if last_epoch_ran < args.epochs:
        print(f"  (stopped after epoch {last_epoch_ran})")
    else:
        print()
    print(f"  train @ best: {format_metrics(best_train_at_best)}")
    if best_val_at_best is not None:
        print(f"  val  @ best: {format_metrics(best_val_at_best)}")
    print(f"  test @ best (held-out, same epoch): {format_metrics(best_test_at_best)}")

    model.load_state_dict(best_state)

    breakout_metrics: dict[str, float] | None = None
    if (
        test_breakout is not None
        and len(test_breakout) == len(y_test)
        and np.any(test_breakout)
    ):
        mask = test_breakout.astype(bool)
        breakout_metrics = metrics_for_split(
            model, X_test_s[mask], y_test[mask], device, args.batch_size
        )
        print(
            f"  test BREAKOUTS @ best (subset n={int(mask.sum())}): "
            f"{format_metrics(breakout_metrics)}"
        )

    print("-" * 72)
    print("")

    return (
        best_train_at_best,
        best_val_at_best,
        best_test_at_best,
        best_select_metrics,
        breakout_metrics,
        best_epoch,
        last_epoch_ran,
        model,
    )

def concat_ticker_sequences(
    df: pd.DataFrame,
    tickers: list[str],
    seq_len: int,
    use_split: str,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build sequences per ticker for one split ("train" or "test").
    # Third return value is the news_breakout flag aligned with each sequence label.
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    brs: list[np.ndarray] = []
    has_br = "news_breakout" in df.columns
    for ticker in sorted(tickers):
        chunk = df[df["ticker"].astype(str) == ticker].sort_values("timestamp_utc")
        chunk = chunk[~pd.isna(chunk["target_up_next_bar"])].reset_index(drop=True)
        split_ok = chunk["split"].astype(str) == use_split
        sub = chunk.loc[split_ok].reset_index(drop=True)
        if len(sub) < seq_len:
            continue
        feats, _ = build_feature_matrix(sub, tickers, feature_set)
        labs = sub["target_up_next_bar"].to_numpy(dtype=np.float64)
        Xi, yi = build_sequences(feats, labs, seq_len)
        if len(Xi):
            xs.append(Xi)
            ys.append(yi)
            if has_br:
                br_seq = [
                    int(sub["news_breakout"].iloc[i])
                    for i in range(seq_len - 1, len(feats))
                ]
                brs.append(np.asarray(br_seq, dtype=np.int64))
            else:
                brs.append(np.zeros(len(Xi), dtype=np.int64))
    if not xs:
        fd = len(build_feature_matrix(df.head(1).copy(), tickers, feature_set)[1])
        return np.empty((0, seq_len, fd)), np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(brs, axis=0)


def concat_ticker_fit_val_sequences(
    df: pd.DataFrame,
    tickers: list[str],
    seq_len: int,
    val_fraction: float,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    # Build sequences from the "train" split.
    # If val_fraction > 0, we take the tail end of each ticker's train timeline as validation.
    xs_fit: list[np.ndarray] = []
    ys_fit: list[np.ndarray] = []
    xs_val: list[np.ndarray] = []
    ys_val: list[np.ndarray] = []
    for ticker in sorted(tickers):
        chunk = df[df["ticker"].astype(str) == ticker].sort_values("timestamp_utc")
        chunk = chunk[~pd.isna(chunk["target_up_next_bar"])].reset_index(drop=True)
        split_ok = chunk["split"].astype(str) == "train"
        sub = chunk.loc[split_ok].reset_index(drop=True)
        if len(sub) < seq_len:
            continue
        feats, _ = build_feature_matrix(sub, tickers, feature_set)
        labs = sub["target_up_next_bar"].to_numpy(dtype=np.float64)
        Xi, yi = build_sequences(feats, labs, seq_len)
        if len(Xi) == 0:
            continue
        if val_fraction <= 0:
            xs_fit.append(Xi)
            ys_fit.append(yi)
            continue
        nv = max(1, int(len(Xi) * val_fraction))
        nv = min(nv, len(Xi) - 20)
        if nv <= 0 or len(Xi) - nv < 20:
            xs_fit.append(Xi)
            ys_fit.append(yi)
        else:
            xs_fit.append(Xi[:-nv])
            ys_fit.append(yi[:-nv])
            xs_val.append(Xi[-nv:])
            ys_val.append(yi[-nv:])

    if not xs_fit:
        fd = len(build_feature_matrix(df.head(1).copy(), tickers, feature_set)[1])
        return np.empty((0, seq_len, fd)), np.empty((0,)), None, None

    X_fit = np.concatenate(xs_fit, axis=0)
    y_fit = np.concatenate(ys_fit, axis=0)
    if not xs_val:
        return X_fit, y_fit, None, None
    X_val = np.concatenate(xs_val, axis=0)
    y_val = np.concatenate(ys_val, axis=0)
    return X_fit, y_fit, X_val, y_val


def main():
    # Inputs: cleaned stock bars in analysis_outputs/cleaned/stocks
    # If you use feature_set=extended_news, we merge analysis_outputs/cleaned/news/news_aligned_by_bar.csv.
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks-dir", type=Path, default=DEFAULT_STOCKS_DIR)
    parser.add_argument("--interval", type=int, default=60)
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
    parser.add_argument("--val-fraction", type=float, default=0.12)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--out-model", type=Path, default=None)
    parser.add_argument("--news-aligned", type=Path, default=DEFAULT_NEWS_ALIGNED)
    parser.add_argument("--no-news-merge", action="store_true")
    args = parser.parse_args()

    args.balance_loss = not args.no_balance_loss
    run_training(args)


def run_training(args: argparse.Namespace) -> dict[str, object]:
    # Core training + metrics; used by main and by the batch scripts.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_frames(args.stocks_dir, args.interval, list(args.tickers))
    df = df[~pd.isna(df["target_up_next_bar"])].reset_index(drop=True)
    if args.feature_set == "extended_news":
        if args.no_news_merge:
            raise ValueError("feature_set=extended_news requires news merge (do not use --no-news-merge).")
        df = merge_news_into_df(
            df, Path(args.news_aligned), args.interval, include_signal_columns=True
        )
    elif not args.no_news_merge:
        df = merge_news_into_df(df, Path(args.news_aligned), args.interval, include_signal_columns=False)
    _, feat_names = build_feature_matrix(df.head(1).copy(), list(args.tickers), args.feature_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LSTM_NEWS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "interval_minutes": args.interval,
        "tickers": list(args.tickers),
        "feature_set": args.feature_set,
        "feature_columns": feat_names,
        "seq_len": args.seq_len,
        "val_fraction": args.val_fraction,
        "early_stopping_patience": args.early_stopping_patience,
        "news_merged": (not args.no_news_merge) or (args.feature_set == "extended_news"),
    }

    X_train, y_train, X_val, y_val = concat_ticker_fit_val_sequences(
        df, list(args.tickers), args.seq_len, args.val_fraction, args.feature_set
    )
    X_test, y_test, test_br = concat_ticker_sequences(
        df, list(args.tickers), args.seq_len, "test", args.feature_set
    )
    test_breakout: np.ndarray | None = (
        test_br if (not args.no_news_merge or args.feature_set == "extended_news") else None
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train or test sequences empty; lower seq_len or check data.")

    X_train_s, X_val_s, X_test_s, _ = scale_sequences_fit_transform(X_train, X_val, X_test)
    y_val_np: np.ndarray | None = y_val if X_val_s is not None else None
    n_val_report = int(len(X_val_s)) if X_val_s is not None else 0

    mf, mv_r, mte, sel_metrics, mte_bo, best_epoch, last_epoch_ran, model = train_full_logging(
        X_train_s,
        y_train,
        X_val_s,
        y_val_np,
        X_test_s,
        y_test,
        test_breakout,
        args,
        device,
        feat_names,
        n_val_report,
    )

    report["best_epoch"] = best_epoch
    report["last_epoch_ran"] = last_epoch_ran
    report["best_train_metrics"] = mf
    report["best_val_metrics"] = mv_r
    report["best_test_metrics_at_best_epoch"] = mte
    report["best_test_breakout_metrics_at_best_epoch"] = mte_bo
    report["selection_metrics_at_best_epoch"] = sel_metrics
    report["train_sequences"] = int(len(X_train))
    report["train_sequences_val"] = n_val_report
    report["test_sequences"] = int(len(X_test))
    report["device"] = str(device)

    if args.save_model:
        if args.out_model is not None:
            out_model = Path(args.out_model)
        elif args.feature_set == "extended_news":
            out_model = LSTM_NEWS_MODEL_DIR / f"{args.interval}m_LSTM_NEWS_epoch_{best_epoch}.pt"
        else:
            out_model = LSTM_MODEL_DIR / f"{args.interval}m_LSTM_OHLCV_epoch_{best_epoch}.pt"
        out_model.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "interval": args.interval,
            "tickers": list(args.tickers),
            "feature_set": args.feature_set,
            "seq_len": args.seq_len,
            "feature_names": feat_names,
            "best_epoch": best_epoch,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "layernorm": not args.no_layernorm,
                "input_dim": X_train.shape[2],
            },
        }
        torch.save(payload, out_model)
        report["saved_model"] = str(out_model)
        print(f"Saved best checkpoint to {out_model}")
    else:
        report["saved_model"] = None

    return report


if __name__ == "__main__":
    main()
