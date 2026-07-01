# ─────────────────────────────────────────────────────────────
# day13_evaluate.py
# Evaluation: computes metrics for all three Day 13 architectures
# and compares them against Day 5's HARNet baseline.
# Also loads and analyses the attention patterns.
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
TICKERS  = ["SPY", "QQQ", "AAPL"]


def rmse(y, yhat):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))

def mae(y, yhat):
    return np.mean(np.abs(np.asarray(y) - np.asarray(yhat)))

def qlike(y, yhat, floor=1e-8):
    h = np.maximum(yhat, floor)
    v = np.maximum(y,    floor)
    return np.mean(np.log(h) + v / h)

def dir_acc(y, yhat):
    return np.mean(
        np.sign(np.diff(y)) == np.sign(np.diff(yhat))
    ) * 100

def compute_all_metrics(y, yhat, model, ticker, version="Day13"):
    return {
        "Ticker" : ticker,
        "Model"  : model,
        "Version": version,
        "RMSE"   : round(rmse(y, yhat),     8),
        "MAE"    : round(mae(y, yhat),      8),
        "QLIKE"  : round(qlike(y, yhat),    6),
        "DirAcc" : round(dir_acc(y, yhat),  2),
        "N"      : len(y),
    }


def load_day5_metrics() -> pd.DataFrame:
    """
    Try to load Day 5 HARNet predictions for comparison.
    Returns empty DataFrame if not found.
    """
    day5_path = os.path.join(
        BASE_DIR, "..", "day05", "output",
        "day05_harnet_predictions.csv"
    )
    if not os.path.exists(day5_path):
        print("  ⚠ Day 5 predictions not found — skipping HARNet baseline")
        return pd.DataFrame()

    df = pd.read_csv(day5_path)
    rows = []
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        rows.append(compute_all_metrics(
            sub["actual"].values,
            sub["harnet_pred"].values,
            model="HARNet_Day5",
            ticker=ticker,
            version="Day5_Baseline"
        ))
    return pd.DataFrame(rows)


def analyse_attention_patterns(all_results: dict) -> pd.DataFrame:
    """
    Compute average attention weight per lag position.

    For each model and ticker, the attention matrix is (T,T) where
    attn[i,j] = how much query position i attends to key position j.
    Averaging over all query positions and all test batches gives
    a 1D vector of length T showing which LAG POSITIONS receive
    the most attention on average.

    Interpretation:
        High attention at lag 1  → model focuses on yesterday (HAR daily)
        High attention at lag 5  → model focuses on last week (HAR weekly)
        High attention at lag 22 → model focuses on a month ago (HAR monthly)
        Diffuse attention        → model uses all timesteps equally
    """
    rows = []
    for ticker, arch_dict in all_results.items():
        for arch_name, res in arch_dict.items():
            attn = res.get("attn")
            if attn is None or len(attn) == 0:
                continue

            # attn shape: (n_batches_in_test, T, T) or similar
            # Average attention received BY each position
            # (sum over query dimension → which keys get attended to?)
            if attn.ndim == 3:
                avg_attn = attn.mean(axis=0).mean(axis=0)  # (T,)
            else:
                avg_attn = attn.mean(axis=0)

            for lag, w in enumerate(avg_attn):
                rows.append({
                    "Ticker"  : ticker,
                    "Arch"    : arch_name,
                    "Lag"     : lag + 1,
                    "Avg_Attn": float(w),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        out = os.path.join(OUT_DIR, "day13_attention_patterns.csv")
        df.to_csv(out, index=False)
        print(f"  ✓ day13_attention_patterns.csv")
    return df


def run_evaluation(all_results: dict) -> pd.DataFrame:
    """
    Compute all metrics for Day 13 models and merge with Day 5 baseline.
    """
    print(f"\n{'='*55}")
    print("  DAY 13 — Evaluation")
    print(f"{'='*55}")

    rows = []
    pred_df = pd.read_csv(
        os.path.join(OUT_DIR, "day13_predictions.csv")
    )

    for ticker in TICKERS:
        sub = pred_df[pred_df["ticker"] == ticker]
        for arch in sub["arch"].unique():
            a_sub = sub[sub["arch"] == arch]
            rows.append(compute_all_metrics(
                a_sub["actual"].values,
                a_sub["pred"].values,
                model=arch,
                ticker=ticker,
            ))

    day13_df = pd.DataFrame(rows)
    day5_df  = load_day5_metrics()

    if not day5_df.empty:
        combined = pd.concat([day13_df, day5_df], ignore_index=True)
    else:
        combined = day13_df

    out = os.path.join(OUT_DIR, "day13_all_metrics.csv")
    combined.to_csv(out, index=False)

    print("\n  RMSE × 10^4 Comparison:")
    pivot = combined.pivot_table(
        index="Model", columns="Ticker",
        values="RMSE", aggfunc="mean"
    ) * 1e4
    print(pivot.round(1).to_string())

    print("\n  DirAcc % Comparison:")
    pivot2 = combined.pivot_table(
        index="Model", columns="Ticker",
        values="DirAcc", aggfunc="mean"
    )
    print(pivot2.round(1).to_string())

    # Attention pattern analysis
    attn_df = analyse_attention_patterns(all_results)

    print(f"\n  ✓ day13_all_metrics.csv")
    return combined, attn_df