#!/usr/bin/env python
"""prepare_data.py

Prepare data for the volatility forecasting pipeline.

- Loads raw market price data from `data/raw/market/<TICKER>.parquet`.
- Generates HAR features using `generate_har_features.py` (already available).
- Loads daily sentiment aggregates produced by `build_sentiment_dataset.py`.
- Merges market and sentiment features, creates target (next‑day volatility),
  performs a chronological train/test split, scales the features, and writes
  the resulting CSVs under `data/processed/`.

The resulting files (`har_features.csv` and `sentiment/daily_<TICKER>.csv`)
are exactly what `train_ridge.py` expects.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_MARKET_DIR = BASE_DIR / "data" / "raw" / "market"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_DIR = PROCESSED_DIR / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Number of samples to hold out for the final test set (as used in the
# original notebook). Adjust if required.
TEST_SIZE = int(os.getenv("TEST_SIZE", "500"))

# Tickers – can be overridden via the TICKERS environment variable.
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,NVDA,TSLA").split(",")
TICKERS = [t.strip() for t in TICKERS if t]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_market(ticker: str) -> pd.DataFrame:
    """Load raw market parquet for a ticker and ensure a clean dataframe.

    Returns a DataFrame with columns ``date`` (datetime) and ``close`` (float).
    """
    path = RAW_MARKET_DIR / f"{ticker}.parquet"
    if not path.exists():
        logging.error(f"Market file not found for {ticker}: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Ensure required columns
    if "date" not in df.columns or "close" not in df.columns:
        logging.error(f"Missing required columns in {path}")
        return pd.DataFrame()
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker
    return df


def load_sentiment(ticker: str) -> pd.DataFrame:
    """Load the daily sentiment aggregation produced by ``build_sentiment_dataset.py``.
    """
    path = SENTIMENT_DIR / f"daily_{ticker}.csv"
    if not path.exists():
        logging.warning(f"Sentiment file missing for {ticker}: {path}. Using placeholder zeros.")
        # Create an empty placeholder with the expected columns
        return pd.DataFrame(columns=["date", "average_sentiment", "sentiment_std", "sentiment_shock", "article_count"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def compute_har_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the classic HAR (heterogeneous autoregressive) features.

    The implementation mirrors the one in ``generate_har_features.py`` but
    operates on the already‑loaded ``df`` to keep the pipeline self‑contained.
    """
    df = df.copy()
    # Daily realized volatility (close‑to‑close)
    df["rv_daily"] = df["close"].pct_change().apply(lambda x: np.sqrt(x ** 2) if pd.notnull(x) else np.nan)
    # Weekly (5 business days) and monthly (22 business days) rolling averages
    df["rv_weekly"] = df["rv_daily"].rolling(window=5, min_periods=1).mean()
    df["rv_monthly"] = df["rv_daily"].rolling(window=22, min_periods=1).mean()
    # Lagged daily volatility for the HAR model
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"rv_lag_{lag}"] = df["rv_daily"].shift(lag)
    # Keep only the columns needed for the regression model
    cols = ["date", "ticker"] + [c for c in df.columns if c.startswith("rv_")]
    return df[cols]


def merge_features(market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge market HAR features with sentiment.
    The merge is on ``date``; ticker column is added afterwards.
    """
    har = compute_har_features(market_df)
    merged = pd.merge(har, sentiment_df, on="date", how="left")
    # Forward‑fill any missing sentiment values (e.g., weekends) and then replace any remaining NaNs with zeros
    merged[["average_sentiment", "sentiment_std", "sentiment_shock", "article_count"]] = \
        merged[["average_sentiment", "sentiment_std", "sentiment_shock", "article_count"]].ffill().fillna(0)
    return merged

def create_train_test(df: pd.DataFrame) -> tuple:
    """Create train/test splits and scale the features.

    Returns ``(X_train, X_test, y_train, y_test, scaler)`` where ``X`` are the
    feature matrices and ``y`` is the next‑day volatility target.
    """
    # Target is next‑day realized volatility (using the daily rv column)
    df["target"] = df["rv_daily"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        logging.warning("DataFrame is empty after preparing target – skipping scaling.")
        # Return empty structures
        scaler = StandardScaler()
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        return empty_X, empty_X, empty_y, empty_y, scaler
    # Chronological split (80% train, 20% test) – same heuristic as the original script
    split_idx = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c not in {"date", "ticker", "target", "rv_daily"}]
    if not feature_cols:
        logging.warning("No feature columns identified – returning empty splits.")
        scaler = StandardScaler()
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=float)
        return empty_X, empty_X, empty_y, empty_y, scaler
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["target"]
    y_test = test_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if not X_train.empty else np.empty((0, len(feature_cols)))
    X_test_scaled = scaler.transform(X_test) if not X_test.empty else np.empty((0, len(feature_cols)))

    # Convert back to DataFrames with original column names for easier downstream use
    X_train_scaled = pd.DataFrame(X_train_scaled, index=train_df.index, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=test_df.index, columns=feature_cols)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    logging.info("Starting data‑preparation pipeline for volatility forecasting.")
    for ticker in TICKERS:
        logging.info(f"Processing ticker {ticker}")
        market_df = load_market(ticker)
        if market_df.empty:
            logging.error(f"Skipping {ticker} – market data unavailable.")
            continue
        sentiment_df = load_sentiment(ticker)
        merged = merge_features(market_df, sentiment_df)
        # Save HAR‑only features for the Ridge script (the script expects a single CSV)
        # We will concatenate across tickers later.
        har_path = PROCESSED_DIR / f"har_{ticker}.csv"
        merged.to_csv(har_path, index=False)
        logging.info(f"Saved merged features for {ticker} to {har_path}")

    # Concatenate all tickers into one HAR file as required by train_ridge.py
    har_frames = []
    for ticker in TICKERS:
        p = PROCESSED_DIR / f"har_{ticker}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["ticker"] = ticker
            har_frames.append(df)
    if har_frames:
        har_all = pd.concat(har_frames, ignore_index=True)
        har_all_path = PROCESSED_DIR / "har_features.csv"
        har_all.to_csv(har_all_path, index=False)
        logging.info(f"Combined HAR features written to {har_all_path}")
    else:
        logging.error("No HAR features were created – aborting.")
        return

    # Create train/test CSVs for each ticker (optional, useful for quick checks)
    for ticker in TICKERS:
        p = PROCESSED_DIR / f"har_{ticker}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        X_train, X_test, y_train, y_test, _ = create_train_test(df)
        out_dir = PROCESSED_DIR / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(out_dir / f"{ticker}_train_X.csv", index=False)
        X_test.to_csv(out_dir / f"{ticker}_test_X.csv", index=False)
        y_train.to_csv(out_dir / f"{ticker}_train_y.csv", index=False)
        y_test.to_csv(out_dir / f"{ticker}_test_y.csv", index=False)
        logging.info(f"Saved train/test split for {ticker} under {out_dir}")

    logging.info("Data‑preparation pipeline completed.")

if __name__ == "__main__":
    main()
