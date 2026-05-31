import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").is_dir() and (candidate / "reports").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _find_repo_root()
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from day05_harnet_model import HARNet, count_parameters

DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "plots" / "day4" / "day05" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 22
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3
PATIENCE = 20
WEIGHT_DECAY = 1e-4
N_FILTERS = 16
FC_HIDDEN = 32
DROPOUT = 0.2
TEST_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TICKERS = ["SPY", "QQQ", "AAPL"]


class RVSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 22):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.targets) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        # convert to (channels, seq_len) for Conv1d: (n_features, seq_len)
        x = x.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    rv_1d = df["log_return"].pow(2) * 252
    feats = pd.DataFrame(index=df.index)
    feats["rv_1d"] = rv_1d.shift(1)
    feats["rv_5d"] = rv_1d.shift(1).rolling(5, min_periods=5).mean()
    feats["rv_22d"] = rv_1d.shift(1).rolling(22, min_periods=22).mean()
    feats["abs_ret"] = df["log_return"].abs().shift(1)

    if "park_vol_5d" in df.columns:
        feats["park_vol_5d"] = df["park_vol_5d"].shift(1)
    if "gk_vol_5d" in df.columns:
        feats["gk_vol_5d"] = df["gk_vol_5d"].shift(1)

    target = rv_1d.shift(-1)
    combined = pd.concat([feats, target.rename("target")], axis=1).dropna()
    X = combined.drop(columns=["target"]).values
    y = combined["target"].values
    return X, y, combined.index


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    test_sz: int = 500,
    seq_len: int = 22,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    if len(X) <= test_sz + seq_len:
        raise ValueError("Not enough rows to create train/validation/test splits")

    split = len(X) - test_sz
    val_sz = max(seq_len + 1, int(split * 0.15))
    if split - val_sz <= seq_len:
        raise ValueError("Training split is too small for the requested sequence length")

    X_train = X[: split - val_sz]
    X_val = X[split - val_sz : split]
    X_test = X[split:]

    y_train = y[: split - val_sz]
    y_val = y[split - val_sz : split]
    y_test = y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(
        RVSequenceDataset(X_train, y_train, seq_len),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loader = DataLoader(
        RVSequenceDataset(X_val, y_val, seq_len),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        RVSequenceDataset(X_test, y_test, seq_len),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, scaler


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.squeeze(-1).to(device)

        optimiser.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        batch_size = len(y_batch)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.squeeze(-1).to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            batch_size = len(y_batch)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / max(1, total_items)


def get_predictions(model, loader, device):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            preds_all.extend(preds.tolist())
            targets_all.extend(y_batch.squeeze(-1).numpy().tolist())

    return np.asarray(preds_all), np.asarray(targets_all)


def train_harnet(ticker: str, df: pd.DataFrame) -> dict:
    print(f"\n{'─' * 50}")
    print(f"  Training HARNet — {ticker}")
    print(f"{'─' * 50}")

    X, y, _ = build_features(df)
    n_features = X.shape[1]
    dl_train, dl_val, dl_test, scaler = make_loaders(
        X, y, test_sz=TEST_SIZE, seq_len=SEQ_LEN
    )

    model = HARNet(
        n_features=n_features,
        seq_len=SEQ_LEN,
        n_filters=N_FILTERS,
        fc_hidden=FC_HIDDEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    print(f"  Parameters : {count_parameters(model):,}")
    print(f"  Device     : {DEVICE}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=8, factor=0.5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = np.inf
    no_improve = 0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_train, optimiser, criterion, DEVICE)
        val_loss = evaluate(model, dl_val, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {key: value.clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  train={tr_loss:.6f}  "
                f"val={val_loss:.6f}  lr={optimiser.param_groups[0]['lr']:.2e}"
            )

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  best_val={best_val:.6f}")
            break

    if best_weights is None:
        best_weights = {key: value.clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_weights)

    ckpt = OUT_DIR / f"day05_harnet_{ticker}.pt"
    torch.save(
        {
            "model_state": best_weights,
            "config": {
                "n_features": n_features,
                "seq_len": SEQ_LEN,
                "n_filters": N_FILTERS,
                "fc_hidden": FC_HIDDEN,
                "dropout": DROPOUT,
            },
            "scaler": scaler,
        },
        ckpt,
    )
    print(f"  Checkpoint → {ckpt}")

    preds, actuals = get_predictions(model, dl_test, DEVICE)
    return {
        "ticker": ticker,
        "history": history,
        "preds": preds,
        "actuals": actuals,
        "best_val": best_val,
    }


def main():
    print(f"\n{'=' * 55}")
    print("  DAY 5 — HARNet Training")
    print(f"{'=' * 55}")
    print(
        f"  Device={DEVICE}  SEQ_LEN={SEQ_LEN}  EPOCHS={EPOCHS}  PATIENCE={PATIENCE}"
    )

    all_results = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}_processed.csv"
        if not path.exists():
            print(f"  ⚠ {path} not found — run Day 2 first")
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        all_results[ticker] = train_harnet(ticker, df)

    if not all_results:
        raise RuntimeError("No tickers were trained; check the processed data inputs")

    rows = []
    for ticker, res in all_results.items():
        for epoch, (tr_loss, val_loss) in enumerate(
            zip(res["history"]["train_loss"], res["history"]["val_loss"]),
            1,
        ):
            rows.append(
                {"ticker": ticker, "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss}
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "day05_training_history.csv", index=False)

    pred_rows = []
    for ticker, res in all_results.items():
        for pred, actual in zip(res["preds"], res["actuals"]):
            pred_rows.append({"ticker": ticker, "actual": actual, "harnet_pred": pred})
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "day05_harnet_predictions.csv", index=False)

    print("\n  ✓ day05_training_history.csv")
    print("  ✓ day05_harnet_predictions.csv")
    return all_results


if __name__ == "__main__":
    main()




    
