import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from day13_attention_model import (  # noqa: E402
    TemporalAttention,
    TemporalAttentionWithPositional,
    count_parameters,
)
from day13_transformer_model import VolatilityTransformer  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 60
LR = 5e-4
PATIENCE = 12
WEIGHT_DECAY = 1e-4
TEST_SIZE = 500
TICKERS = ["SPY", "QQQ", "AAPL"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RVSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int = 32):
        self.x = features.astype(np.float32)
        self.y = target.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.x[idx : idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return torch.from_numpy(x_seq), torch.tensor([y_next], dtype=torch.float32)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rv_1d = (df["log_return"] ** 2) * 252

    feats = pd.DataFrame(index=df.index)
    feats["rv_1d"] = rv_1d.shift(1)
    feats["rv_5d_avg"] = rv_1d.shift(1).rolling(5, min_periods=5).mean()
    feats["rv_22d_avg"] = rv_1d.shift(1).rolling(22, min_periods=22).mean()
    feats["abs_ret"] = df["log_return"].abs().shift(1)

    if "park_vol_5d" in df.columns:
        feats["park_vol_5d"] = df["park_vol_5d"].shift(1)
    if "gk_vol_5d" in df.columns:
        feats["gk_vol_5d"] = df["gk_vol_5d"].shift(1)

    target = rv_1d.shift(-1)
    combined = pd.concat([feats, target.rename("target")], axis=1).dropna()
    x = combined.drop(columns=["target"]).values
    y = combined["target"].values
    return x, y


def make_loaders(x: np.ndarray, y: np.ndarray) -> tuple[DataLoader, DataLoader, DataLoader]:
    n = len(x)
    split = n - TEST_SIZE
    val_sz = max(1, int(split * 0.15))

    x_tr = x[: split - val_sz]
    x_va = x[split - val_sz : split]
    x_te = x[split:]
    y_tr = y[: split - val_sz]
    y_va = y[split - val_sz : split]
    y_te = y[split:]

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_va = scaler.transform(x_va)
    x_te = scaler.transform(x_te)

    dl_tr = DataLoader(RVSequenceDataset(x_tr, y_tr, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
    dl_va = DataLoader(RVSequenceDataset(x_va, y_va, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(RVSequenceDataset(x_te, y_te, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
    return dl_tr, dl_va, dl_te


def train_epoch(model: nn.Module, loader: DataLoader, opt, criterion, device: torch.device) -> float:
    model.train()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.squeeze(-1).to(device)
        opt.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        opt.step()
        bsz = len(yb)
        total += loss.item() * bsz
        count += bsz
    return total / max(1, count)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.squeeze(-1).to(device)
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        bsz = len(yb)
        total += loss.item() * bsz
        count += bsz
    return total / max(1, count)


@torch.no_grad()
def get_predictions_and_attention(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    actuals = []
    all_attn = []

    for xb, yb in loader:
        xb = xb.to(device)
        pred, attn_w = model(xb)
        preds.extend(pred.detach().cpu().numpy().tolist())
        actuals.extend(yb.squeeze(-1).numpy().tolist())

        if isinstance(attn_w, list) and len(attn_w) > 0:
            all_attn.extend(attn_w[-1].detach().cpu().numpy())
        elif attn_w is not None:
            all_attn.extend(attn_w.detach().cpu().numpy())

    attn = np.array(all_attn) if len(all_attn) > 0 else None
    return np.array(preds), np.array(actuals), attn


def train_model(model: nn.Module, ticker: str, dl_tr: DataLoader, dl_va: DataLoader, dl_te: DataLoader) -> dict:
    model_name = model.__class__.__name__
    print(f"\n  Training {model_name} for {ticker}")
    print(f"  Parameters: {count_parameters(model):,}")

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    criterion = nn.MSELoss()

    best_val = np.inf
    no_impr = 0
    best_wts = None
    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, dl_tr, opt, criterion, DEVICE)
        val_loss = eval_epoch(model, dl_va, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_wts = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_impr = 0
        else:
            no_impr += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d} | train={tr_loss:.6f} | val={val_loss:.6f}")

        if no_impr >= PATIENCE:
            print(f"    Early stop at epoch {epoch}, best_val={best_val:.6f}")
            break

    if best_wts is not None:
        model.load_state_dict(best_wts)

    ckpt_name = f"day13_{model_name}_{ticker}.pt"
    torch.save(
        {"model_state": model.state_dict(), "model_name": model_name, "ticker": ticker},
        os.path.join(OUT_DIR, ckpt_name),
    )

    preds, acts, attn = get_predictions_and_attention(model, dl_te, DEVICE)

    return {
        "model_name": model_name,
        "ticker": ticker,
        "history": history,
        "preds": preds,
        "actuals": acts,
        "attn": attn,
        "best_val": float(best_val),
    }


def run_training() -> dict:
    print("  DAY 13 - Attention Model Training")
    print(f"{'=' * 55}")
    print(f"  Device: {DEVICE}")

    all_results: dict[str, dict] = {}

    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found")
            continue

        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        x, y = build_features(df)
        n_features = x.shape[1]

        dl_tr, dl_va, dl_te = make_loaders(x, y)
        all_results[ticker] = {}

        m1 = TemporalAttention(n_features=n_features, d_model=32, dropout=0.1)
        r1 = train_model(m1, ticker, dl_tr, dl_va, dl_te)
        all_results[ticker]["TemporalAttention"] = r1

        m2 = TemporalAttentionWithPositional(n_features=n_features, d_model=32, seq_len=SEQ_LEN, dropout=0.1)
        r2 = train_model(m2, ticker, dl_tr, dl_va, dl_te)
        all_results[ticker]["TemporalAttentionPos"] = r2

        m3 = VolatilityTransformer(
            n_features=n_features,
            seq_len=SEQ_LEN,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            dropout=0.1,
        )
        r3 = train_model(m3, ticker, dl_tr, dl_va, dl_te)
        all_results[ticker]["Transformer"] = r3

    rows = []
    for ticker, arch_dict in all_results.items():
        for arch_name, res in arch_dict.items():
            for pred, actual in zip(res["preds"], res["actuals"]):
                rows.append(
                    {
                        "ticker": ticker,
                        "arch": arch_name,
                        "actual": float(actual),
                        "pred": float(pred),
                    }
                )
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "day13_predictions.csv"), index=False)
    print("\n  Saved day13_predictions.csv")

    h_rows = []
    for ticker, arch_dict in all_results.items():
        for arch_name, res in arch_dict.items():
            for ep, (tr, val) in enumerate(zip(res["history"]["train"], res["history"]["val"]), 1):
                h_rows.append(
                    {
                        "ticker": ticker,
                        "arch": arch_name,
                        "epoch": ep,
                        "train_loss": float(tr),
                        "val_loss": float(val),
                    }
                )

    pd.DataFrame(h_rows).to_csv(os.path.join(OUT_DIR, "day13_training_history.csv"), index=False)
    print("  Saved day13_training_history.csv")

    return all_results











