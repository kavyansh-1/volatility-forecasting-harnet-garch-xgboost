"""Mocks or fetches news data."""
import os, pandas as pd

if __name__ == "__main__":
    os.makedirs("day06/output", exist_ok=True)
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.DataFrame({"Date": ["2024-01-01"], "Headline": ["Market goes up"]})
        df.to_csv(f"data/raw/{t}_news.csv", index=False)
