"""Downloads daily OHLCV data."""
import os
import yfinance as yf
import pandas as pd

def create_folders():
    for f in ["data/raw", "data/processed", "day01/src", "day01/output"]:
        os.makedirs(f, exist_ok=True)

if __name__ == "__main__":
    create_folders()
    for t in ["SPY", "QQQ", "AAPL"]:
        df = yf.download(t, start="2015-01-01", end="2025-01-01", auto_adjust=True)
        df.to_csv(f"data/raw/{t}_daily.csv")
