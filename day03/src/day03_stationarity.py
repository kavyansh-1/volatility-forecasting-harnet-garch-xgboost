"""Tests for stationarity."""
import os, pandas as pd
from statsmodels.tsa.stattools import adfuller

if __name__ == "__main__":
    os.makedirs("day03/output", exist_ok=True)
    res = []
    for t in ["SPY", "QQQ", "AAPL"]:
        df = pd.read_csv(f"data/processed/{t}_processed.csv", index_col="Date", parse_dates=True)
        ret = df['log_return'].dropna()
        stat, p, _, _, _, _ = adfuller(ret)
        res.append({"Ticker": t, "ADF_p_value": p, "Stationary": p < 0.05})
    pd.DataFrame(res).to_csv("day03/output/day03_stationarity.csv", index=False)
