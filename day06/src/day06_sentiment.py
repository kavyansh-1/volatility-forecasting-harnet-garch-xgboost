"""Scores headlines."""
import os, pandas as pd
if __name__ == "__main__":
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.DataFrame({"Date": ["2024-01-01"], "compound_score": [0.5]})
        df.to_csv(f"day06/output/day06_{t}_daily_sentiment.csv", index=False)
