"""Computes log returns and simple returns."""
import os, numpy as np, pandas as pd

if __name__ == "__main__":
    os.makedirs("day02/output", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/raw/{ticker}_daily.csv", index_col="Date", parse_dates=True)
        df['simple_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv_1d'] = (df['log_return'] ** 2) * 252
        df.to_csv(f"data/processed/{ticker}_returns.csv")
