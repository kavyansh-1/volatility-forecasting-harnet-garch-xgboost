# ─────────────────────────────────────────────────────────────
# day14_evaluate.py
# Evaluation of regime-conditional models.
# Computes overall and PER-REGIME metrics to answer:
#   "Does regime conditioning help most in HIGH-vol periods?"
# ─────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR  = os.path.join(BASE_DIR, "output")
TICKERS  = ["SPY", "QQQ", "AAPL"]
REGIME_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def rmse(y, yhat):
    return np.sqrt(np.mean((np.asarray(y)-np.asarray(yhat))**2))

def mae(y, yhat):
    return np.mean(np.abs(np.asarray(y)-np.asarray(yhat)))

def qlike(y, yhat, floor=1e-8):
    h = np.maximum(yhat, floor)
    v = np.maximum(y,    floor)
    return np.mean(np.log(h) + v/h)

def dir_acc(y, yhat):
    return np.mean(
        np.sign(np.diff(y))==np.sign(np.diff(yhat))
    ) * 100 if len(y) > 1 else np.nan


def compute_metrics(y, yhat, label, ticker, regime="All"):
    return {
        "Ticker" : ticker,
        "Model"  : label,
        "Regime" : regime,
        "RMSE"   : round(rmse(y, yhat),    8),
        "MAE"    : round(mae(y, yhat),     8),
        "QLIKE"  : round(qlike(y, yhat),   6),
        "DirAcc" : round(dir_acc(y, yhat), 2),
        "N"      : len(y),
    }


def simple_dm_test(y, yhat1, yhat2) -> dict:
    """
    Simplified DM test comparing two forecasts.
    d_t = (y-yhat1)^2 - (y-yhat2)^2
    t-stat on mean(d): positive = yhat2 better.
    """
    d = (y-yhat1)**2 - (y-yhat2)**2
    t, p = stats.ttest_1samp(d, 0)
    return {
        "DM_tstat" : round(float(t), 4),
        "DM_pvalue": round(float(p), 4),
        "Better"   : "RegimeModel" if t > 0 else "GlobalHAR",
        "Sig"      : bool(p < 0.05),
    }


def run_evaluation(regime_results: dict) -> tuple:
    """
    Overall and per-regime metrics for all models × tickers.
    """
    print(f"\n{'='*55}")
    print("  DAY 14 — Regime-Conditional Evaluation")
    print(f"{'='*55}")

    pred_csv = os.path.join(OUT_DIR, "day14_regime_predictions.csv")
    if not os.path.exists(pred_csv):
        print("  ⚠ Predictions not found — run day14_regime_models.py first")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(pred_csv)
    all_rows = []
    dm_rows  = []

    model_cols = {
        "GlobalHAR"   : "pred_global_har",
        "RegimeHAR"   : "pred_regime_har",
        "RegimeXGB"   : "pred_xgb_regime",
    }

    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        y   = sub["actual"].values

        # Overall metrics
        for label, col in model_cols.items():
            yhat = sub[col].values
            all_rows.append(
                compute_metrics(y, yhat, label, ticker, "All")
            )

        # Per-regime metrics
        for regime_id, regime_name in REGIME_NAMES.items():
            mask = (sub["regime"] == regime_id)
            if mask.sum() < 5:
                continue
            y_r = sub.loc[mask, "actual"].values
            for label, col in model_cols.items():
                yhat_r = sub.loc[mask, col].values
                all_rows.append(
                    compute_metrics(y_r, yhat_r, label, ticker, regime_name)
                )

        # DM tests: RegimeHAR vs GlobalHAR, RegimeXGB vs GlobalHAR
        for label, col in [("RegimeHAR","pred_regime_har"),
                            ("RegimeXGB","pred_xgb_regime")]:
            dm = simple_dm_test(y, sub["pred_global_har"].values,
                                sub[col].values)
            dm["Ticker"]  = ticker
            dm["Vs"]      = f"GlobalHAR vs {label}"
            dm_rows.append(dm)

    metrics_df = pd.DataFrame(all_rows)
    dm_df      = pd.DataFrame(dm_rows)

    metrics_df.to_csv(os.path.join(OUT_DIR,"day14_metrics.csv"), index=False)
    dm_df.to_csv(os.path.join(OUT_DIR,"day14_dm_tests.csv"), index=False)

    # Print summary
    print("\n  RMSE × 10^4 by Model (All regimes):")
    overall = metrics_df[metrics_df["Regime"]=="All"]
    pivot = overall.pivot_table(
        index="Model", columns="Ticker", values="RMSE", aggfunc="mean"
    ) * 1e4
    print(pivot.round(1).to_string())

    print("\n  High-Vol Regime RMSE × 10^4:")
    high = metrics_df[metrics_df["Regime"]=="High"]
    if not high.empty:
        pivot2 = high.pivot_table(
            index="Model", columns="Ticker", values="RMSE", aggfunc="mean"
        ) * 1e4
        print(pivot2.round(1).to_string())

    print("\n  DM Tests (positive t-stat = regime model better):")
    print(dm_df[["Vs","Ticker","DM_tstat","DM_pvalue","Better","Sig"]]
          .to_string(index=False))

    print(f"\n  ✓ day14_metrics.csv")
    print(f"  ✓ day14_dm_tests.csv")
    return metrics_df, dm_df