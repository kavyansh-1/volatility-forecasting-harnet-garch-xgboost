"""Computes rolling volatility estimators."""
import os, numpy as np, pandas as pd

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{ticker}_returns.csv", index_col="Date", parse_dates=True)
        for w in [5, 21]:
            df[f'vol_c2c_{w}d'] = df['log_return'].rolling(w).std() * np.sqrt(252)
            # Parkinson
            hl = (np.log(df['High']/df['Low']))**2
            df[f'vol_park_{w}d'] = np.sqrt((1/(4*np.log(2))) * hl.rolling(w).mean()) * np.sqrt(252)
        df.dropna(inplace=True)
        df.to_csv(f"data/processed/{ticker}_processed.csv")
