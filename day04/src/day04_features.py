"""Builds lag features."""
import os, pandas as pd

if __name__ == "__main__":
    os.makedirs("day04/output", exist_ok=True)
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{ticker}_processed.csv", index_col="Date", parse_dates=True)
        df['target_rv_1d'] = df['rv_1d'].shift(-1)
        for k in [1, 5, 21]:
            df[f'rv_lag_{k}'] = df['rv_1d'].shift(k)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{ticker}_features.csv")
