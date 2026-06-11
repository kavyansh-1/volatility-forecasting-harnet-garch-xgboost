# ─────────────────────────────────────────────────────────────
# day05_evaluate.py
# Computes RMSE / MAE / MAPE / QLIKE / DirAcc for HARNet.
# Loads Day 4 metrics and appends HARNet to produce a
# unified 4-model comparison CSV.
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").is_dir() and (candidate / "reports").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _find_repo_root()
OUT_DIR = REPO_ROOT / "plots" / "day4" / "day05" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DAY4_METRICS = REPO_ROOT / "reports" / "day4_model_metrics.csv"


# ── Metric functions ────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred, eps=1e-10):
    return np.mean(np.abs(y_true - y_pred) /
                   (np.abs(y_true) + eps)) * 100

def qlike(y_true, y_pred, floor=1e-8):
    """
    mean( log(h_pred) + y_true / h_pred )
    Minimised when h_pred == y_true.
    More negative = better fit.
    """
    h = np.maximum(y_pred, floor)
    y = np.maximum(y_true, floor)
    return np.mean(np.log(h) + y / h)

def directional_accuracy(y_true, y_pred):
    """
    % of days where predicted direction of vol change matches actual.
    50% = random baseline.
    """
    return np.mean(
        np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    ) * 100

def compute_all_metrics(y_true, y_pred, model, ticker):
    return {
        "Ticker" : ticker,
        "Model"  : model,
        "RMSE"   : round(rmse(y_true, y_pred),                  8),
        "MAE"    : round(mae(y_true,  y_pred),                  8),
        "MAPE"   : round(mape(y_true, y_pred),                  4),
        "QLIKE"  : round(qlike(y_true, y_pred),                 6),
        "DirAcc" : round(directional_accuracy(y_true, y_pred),  2),
        "N"      : len(y_true),
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  DAY 5 — Evaluation: HARNet vs All Models")
    print(f"{'='*55}")

    pred_path = OUT_DIR / "day05_harnet_predictions.csv"
    if not pred_path.exists():
        print("  ⚠ Run day05_train.py first.")
        return

    pred_df    = pd.read_csv(pred_path)
    harnet_rows = []

    for ticker in pred_df["ticker"].unique():
        sub    = pred_df[pred_df["ticker"] == ticker]
        y_true = sub["actual"].values
        y_pred = sub["harnet_pred"].values
        row    = compute_all_metrics(y_true, y_pred, "HARNet", ticker)
        harnet_rows.append(row)
        print(f"\n  {ticker} HARNet:")
        for k, v in row.items():
            if k not in ["Ticker", "Model", "N"]:
                print(f"    {k:8s}: {v}")

    harnet_df = pd.DataFrame(harnet_rows)

    # Merge with Day 4 results if available
    if DAY4_METRICS.exists():
        day4_df  = pd.read_csv(DAY4_METRICS)
        combined = pd.concat([day4_df, harnet_df], ignore_index=True)
        print(f"\n  Merged with Day 4 metrics.")
    else:
        combined = harnet_df
        print("  ⚠ Day 4 metrics not found — HARNet only.")

    out = OUT_DIR / "day05_all_model_metrics.csv"
    combined.to_csv(out, index=False)
    print(f"\n  ✓ Saved → {out}")

    # Quick pivot tables
    print("\n  RMSE × 10^4:")
    print((combined.pivot_table(
        index="Model", columns="Ticker",
        values="RMSE", aggfunc="mean"
    ) * 1e4).round(1).to_string())

    print("\n  DirAcc %:")
    print(combined.pivot_table(
        index="Model", columns="Ticker",
        values="DirAcc", aggfunc="mean"
    ).round(1).to_string())

    return combined


if __name__ == "__main__":
    main()