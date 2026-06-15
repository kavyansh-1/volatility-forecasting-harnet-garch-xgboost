import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "output")

TICKERS  = ["SPY", "QQQ", "AAPL"]
TEST_SIZE = 500


def label_regimes(rv_series: pd.Series) -> pd.Series:
    
    p33 = rv_series.quantile(0.33)
    p67 = rv_series.quantile(0.67)

    regimes = pd.Series("medium", index=rv_series.index)
    regimes[rv_series <= p33] = "low"
    regimes[rv_series >= p67] = "high"

    return regimes


def load_test_regimes(ticker: str) -> pd.Series:
  
    path = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(path):
        print(f"  ⚠ {path} not found")
        return None

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if "rv_rolling_21d" not in df.columns:
        return None

    regimes = label_regimes(df["rv_rolling_21d"])
    return regimes.iloc[-TEST_SIZE:]


def rmse(y, yhat):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))

def mae(y, yhat):
    return np.mean(np.abs(np.asarray(y) - np.asarray(yhat)))

def qlike(y, yhat, floor=1e-8):
    h = np.maximum(np.asarray(yhat), floor)
    v = np.maximum(np.asarray(y),    floor)
    return np.mean(np.log(h) + v / h)


def evaluate_by_regime(y:        np.ndarray,
                        yhat:     np.ndarray,
                        regimes:  pd.Series,
                        model:    str,
                        ticker:   str) -> list:
    
    rows = []
    for regime in ["low", "medium", "high"]:
        mask = (regimes == regime).values

        
        if mask.sum() < 10:
            continue

        y_r    = y[mask]
        yhat_r = yhat[mask]

        rows.append({
            "Ticker"    : ticker,
            "Model"     : model,
            "Regime"    : regime,
            "N"         : int(mask.sum()),
            "RMSE"      : round(rmse(y_r,  yhat_r), 8),
            "MAE"       : round(mae(y_r,   yhat_r), 8),
            "QLIKE"     : round(qlike(y_r, yhat_r), 6),
        })
    return rows


def run_regime_analysis(predictions: dict) -> pd.DataFrame:
    print(f"\n{'─'*50}")
    print("  Regime Analysis")
    print(f"{'─'*50}")

    all_rows = []

    for ticker in TICKERS:
        regimes = load_test_regimes(ticker)
        if regimes is None:
            print(f"  ⚠ Regimes not available for {ticker}")
            continue

        if ticker not in predictions:
            continue

        y = predictions["y_test"].get(ticker)
        if y is None:
            continue

        
        n = min(len(y), len(regimes))
        regimes_aligned = regimes.iloc[:n]
        y_aligned       = y[:n]

       
        counts = regimes_aligned.value_counts()
        print(f"\n  {ticker} regime distribution:")
        for reg in ["low", "medium", "high"]:
            print(f"    {reg:8s}: {counts.get(reg, 0):4d} days")

        for model_key, yhat in predictions[ticker].items():
            if yhat is None:
                continue
            yhat_aligned = yhat[:n]
            rows = evaluate_by_regime(
                y_aligned, yhat_aligned,
                regimes_aligned, model_key, ticker
            )
            all_rows.extend(rows)

    if not all_rows:
        print("  ⚠ No regime results generated")
        return pd.DataFrame()

    regime_df = pd.DataFrame(all_rows)

    # Save
    out = os.path.join(OUT_DIR, "day08_regime_analysis.csv")
    regime_df.to_csv(out, index=False)
    print(f"\n  ✓ Regime analysis → {out}")

   
    print("\n  High-volatility regime RMSE (most critical):")
    high_df = regime_df[regime_df["Regime"] == "high"]
    if not high_df.empty:
        pivot = high_df.pivot_table(
            index="Model", columns="Ticker",
            values="RMSE", aggfunc="mean"
        ).round(6)
        print(pivot.to_string())

    return regime_df